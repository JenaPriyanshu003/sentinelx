"""
SentinelX Inference API v3
- Loads the fine-tuned xception_finetuned.pth
- Face detection via OpenCV Haar Cascade (no build-dep required,
  same pipeline used during training)
- Three endpoints: /predict/image, /predict/video, /predict/frame
"""

import io
import os
import base64
import tempfile
import cv2
import torch
import timm
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
app = FastAPI(title="SentinelX Inference API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)

FINETUNED = os.path.join(_ROOT, "models", "xception_finetuned.pth")
PRETRAINED = os.path.join(_ROOT, "models", "xception_pretrained.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server device: {device}")

# ---------------------------------------------------------------------------
# Model
def _load_model():
    m = timm.create_model("xception", pretrained=False, num_classes=2)
    # Prefer the fine-tuned weights
    weight_path = FINETUNED if os.path.exists(FINETUNED) else PRETRAINED
    state = torch.load(weight_path, map_location="cpu")
    m.load_state_dict(state, strict=False)
    m.to(device)
    m.eval()
    print(f"Loaded model weights: {os.path.basename(weight_path)}")
    return m

try:
    model = _load_model()
except Exception as e:
    print(f"Model load error: {e}")
    model = None

# ---------------------------------------------------------------------------
# Face detector (Haar Cascade — no C++ build deps, ships with opencv-python)
HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR)

# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# ---------------------------------------------------------------------------
def _crop_face(pil_img: Image.Image) -> Image.Image:
    """Return the largest face crop, or the full frame if no face found."""
    cv_img = cv2.cvtColor(
        cv2.cvtColor(
            cv2.cvtColor(
                cv2.resize(
                    cv2.cvtColor(
                        # PIL -> numpy RGB -> BGR
                        cv2.cvtColor(__import__("numpy").array(pil_img), cv2.COLOR_RGB2BGR),
                        cv2.COLOR_BGR2GRAY
                    ), (0, 0), fx=1, fy=1
                ),
                cv2.COLOR_GRAY2BGR
            ), cv2.COLOR_BGR2GRAY
        ), cv2.COLOR_GRAY2BGR
    )
    # Simpler path:
    import numpy as np
    np_img = np.array(pil_img)  # RGB
    gray   = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    faces  = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces):
        x, y, w, h = faces[0]
        pad = int(0.1 * min(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(np_img.shape[1], x + w + pad), min(np_img.shape[0], y + h + pad)
        return Image.fromarray(np_img[y1:y2, x1:x2])
    return pil_img  # fallback: full frame


def _infer(pil_img: Image.Image):
    """Crop face → transform → forward pass. Returns (prob_fake, prob_real)."""
    face   = _crop_face(pil_img)
    tensor = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]
    # Class 0 = Fake, Class 1 = Real  (matches training label convention)
    return probs[0].item(), probs[1].item()


def _verdict(prob_fake: float, prob_real: float, analyzed_type: str) -> dict:
    conf    = max(prob_fake, prob_real)
    is_fake = prob_fake > prob_real
    return {
        "prediction":    "Fake" if is_fake else "Real",
        "confidence":    round(conf * 100, 2),
        "score_fake":    round(prob_fake * 100, 2),
        "score_real":    round(prob_real * 100, 2),
        "analyzed_type": analyzed_type,
    }

# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "model": os.path.basename(FINETUNED if os.path.exists(FINETUNED) else PRETRAINED)}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(500, "Model not loaded.")
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pf, pr = _infer(img)
    return _verdict(pf, pr, "image")


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(500, "Model not loaded.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap         = cv2.VideoCapture(tmp_path)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = min(15, max(1, total))
    step        = max(1, total // num_samples)

    fakes, reals = [], []
    for i in range(num_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if ok:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pf, pr = _infer(pil)
            fakes.append(pf)
            reals.append(pr)

    cap.release()
    os.unlink(tmp_path)

    if not fakes:
        raise HTTPException(400, "Could not extract usable frames.")

    return _verdict(
        sum(fakes) / len(fakes),
        sum(reals) / len(reals),
        "video"
    )


@app.post("/predict/frame")
async def predict_frame(
    file: UploadFile = File(None),
    frame_base64: str = Form(None)
):
    """Low-latency endpoint for the browser extension (zero disk writes)."""
    if not model:
        raise HTTPException(500, "Model not loaded.")

    try:
        if file:
            raw = await file.read()
        elif frame_base64:
            b64 = frame_base64.split(",")[-1]   # strip data URI prefix if present
            raw = base64.b64decode(b64)
        else:
            raise HTTPException(400, "No frame payload.")

        img     = Image.open(io.BytesIO(raw)).convert("RGB")
        pf, pr  = _infer(img)
        result  = _verdict(pf, pr, "frame")
        result["status"] = "success"
        return result

    except Exception as exc:
        return {"status": "no_face", "prediction": "Unknown", "confidence": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
