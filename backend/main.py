"""
SentinelX Inference API v4
- MongoDB Atlas integration (Motor async driver)
- Auth: Register / Login with JWT tokens + bcrypt password hashing
- Scan history: every analysis is saved to DB per user
- Endpoints: /auth/register, /auth/login, /scans/history, /predict/image, /predict/video, /predict/frame
"""

import io, os, base64, tempfile, cv2, torch, timm
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from torchvision import transforms
from pydantic import BaseModel, EmailStr

# Auth / DB libs
from passlib.context import CryptContext
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI", "mongodb+srv://rajkumarpadhy2006_db_user:vHuvCUuVbheo3vF5@cluster0.eeltwtb.mongodb.net/sentinelx?retryWrites=true&w=majority&appName=Cluster0")
SECRET_KEY  = os.getenv("SECRET_KEY", "sentinelx-super-secret-key-change-in-prod")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 72

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SentinelX API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB ───────────────────────────────────────────────────────────────────
client = AsyncIOMotorClient(MONGO_URI)
db     = client["sentinelx"]
users_col = db["users"]
scans_col = db["scans"]

# ── Security ──────────────────────────────────────────────────────────────────
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer   = HTTPBearer(auto_error=False)

def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds:
        return None
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")   # returns email
    except JWTError:
        return None

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class RegisterSchema(BaseModel):
    name: str
    email: str
    password: str

class LoginSchema(BaseModel):
    email: str
    password: str

# ── AI Model ──────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
FINETUNED  = os.path.join(_ROOT, "models", "xception_finetuned.pth")
PRETRAINED = os.path.join(_ROOT, "models", "xception_pretrained.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    m = timm.create_model("xception", pretrained=False, num_classes=2)
    weight_path = FINETUNED if os.path.exists(FINETUNED) else PRETRAINED
    state = torch.load(weight_path, map_location="cpu")
    m.load_state_dict(state, strict=False)
    m.to(device).eval()
    print(f"Loaded: {os.path.basename(weight_path)}")
    return m

try:
    model = _load_model()
except Exception as e:
    print(f"Model load error: {e}")
    model = None

HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def _crop_face(pil_img):
    import numpy as np
    np_img = np.array(pil_img)
    gray   = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    faces  = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces):
        x, y, w, h = faces[0]
        pad = int(0.1 * min(w, h))
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(np_img.shape[1], x+w+pad), min(np_img.shape[0], y+h+pad)
        return Image.fromarray(np_img[y1:y2, x1:x2])
    return pil_img

def _infer(pil_img):
    face   = _crop_face(pil_img)
    tensor = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]
    return probs[0].item(), probs[1].item()

def _verdict(prob_fake, prob_real, analyzed_type):
    conf    = max(prob_fake, prob_real)
    is_fake = prob_fake > prob_real
    return {
        "prediction":    "Fake" if is_fake else "Real",
        "confidence":    round(conf * 100, 2),
        "score_fake":    round(prob_fake * 100, 2),
        "score_real":    round(prob_real * 100, 2),
        "analyzed_type": analyzed_type,
    }

# ── Auth Endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/register")
async def register(body: RegisterSchema):
    existing = await users_col.find_one({"email": body.email})
    if existing:
        raise HTTPException(400, "Email already registered.")
    user = {
        "name":       body.name,
        "email":      body.email,
        "password":   hash_password(body.password),
        "created_at": datetime.utcnow(),
    }
    await users_col.insert_one(user)
    token = create_token({"sub": body.email, "name": body.name})
    return {"token": token, "name": body.name, "email": body.email}

@app.post("/auth/login")
async def login(body: LoginSchema):
    user = await users_col.find_one({"email": body.email})
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(401, "Invalid email or password.")
    token = create_token({"sub": body.email, "name": user["name"]})
    return {"token": token, "name": user["name"], "email": body.email}

# ── Scan History ──────────────────────────────────────────────────────────────
@app.get("/scans/history")
async def scan_history(user_email: Optional[str] = Depends(get_current_user)):
    if not user_email:
        raise HTTPException(401, "Login required to view history.")
    cursor = scans_col.find({"user_email": user_email}).sort("timestamp", -1).limit(20)
    scans  = []
    async for s in cursor:
        s["_id"] = str(s["_id"])
        scans.append(s)
    return scans

# ── Predict Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": os.path.basename(FINETUNED if os.path.exists(FINETUNED) else PRETRAINED)}

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    user_email: Optional[str] = Depends(get_current_user)
):
    if not model:
        raise HTTPException(500, "Model not loaded.")
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pf, pr = _infer(img)
    result = _verdict(pf, pr, "image")
    if user_email:
        await scans_col.insert_one({**result, "user_email": user_email, "timestamp": datetime.utcnow()})
    return result

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    user_email: Optional[str] = Depends(get_current_user)
):
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
            fakes.append(pf); reals.append(pr)
    cap.release()
    os.unlink(tmp_path)
    if not fakes:
        raise HTTPException(400, "Could not extract usable frames.")
    result = _verdict(sum(fakes)/len(fakes), sum(reals)/len(reals), "video")
    if user_email:
        await scans_col.insert_one({**result, "user_email": user_email, "timestamp": datetime.utcnow()})
    return result

@app.post("/predict/frame")
async def predict_frame(file: UploadFile = File(None), frame_base64: str = Form(None)):
    if not model:
        raise HTTPException(500, "Model not loaded.")
    try:
        if file:
            raw = await file.read()
        elif frame_base64:
            raw = base64.b64decode(frame_base64.split(",")[-1])
        else:
            raise HTTPException(400, "No frame payload.")
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        pf, pr = _infer(img)
        result = _verdict(pf, pr, "frame")
        result["status"] = "success"
        return result
    except Exception:
        return {"status": "no_face", "prediction": "Unknown", "confidence": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
