"""
SentinelX - Xception Fine-tuning Script (Phase 6: Augmentation)
Adds JPEG compression, blur, and noise augmentations to generalize
against screenshot and ROOP-style deepfakes.
"""

import os
import io
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# Resolve paths relative to THIS script file, regardless of cwd
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)   # parent = SentinelX/

# =============================================================================
# CONFIG
# =============================================================================
REAL_DIR   = os.path.join(_ROOT, "dataset", "raw", "real")
FAKE_DIR   = os.path.join(_ROOT, "dataset", "raw", "fake")
FF_REAL_DIR= os.path.join(_ROOT, "dataset", "raw", "ff_real")
FF_FAKE_DIR= os.path.join(_ROOT, "dataset", "raw", "ff_fake")
MODEL_BASE = os.path.join(_ROOT, "models",  "xception_pretrained.pth")
MODEL_OUT  = os.path.join(_ROOT, "models",  "xception_finetuned.pth")

FRAMES_PER_VIDEO = 8
IMG_SIZE        = 299
BATCH_SIZE      = 16
TARGET_EPOCHS   = 25      # Automated Phase: 10 more epochs (16-25)
RESUME_LR       = 5e-7    # Low LR for fine-tuning on new data
WEIGHT_DECAY    = 1e-5
UNFREEZE_EPOCH  = 3
VAL_SPLIT       = 0.15
PATIENCE        = 5       # Early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)

# =============================================================================
# AUGMENTATIONS  (simulate real-world screenshot / social-media deepfakes)
# =============================================================================
class RandomJPEGCompression:
    """Simulate lossy JPEG compression (as seen in screenshots & social posts)."""
    def __init__(self, quality_range=(30, 90)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianNoise:
    """Add pixel-level Gaussian noise to simulate camera sensor artifacts."""
    def __init__(self, std_range=(0.0, 0.05)):
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_range)
        return torch.clamp(tensor + torch.randn_like(tensor) * std, -1.0, 1.0)

# =============================================================================
# EARLY STOPPING
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# =============================================================================
# FACE EXTRACTION
# =============================================================================
def extract_faces_from_video(video_path, n_frames=FRAMES_PER_VIDEO):
    crops = []
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return crops

    step = max(1, total // n_frames)
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            continue

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces):
            x, y, w, h = faces[0]
            pad  = int(0.1 * min(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]
            pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(pil)
        else:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crops.append(pil)

    cap.release()
    return crops

# =============================================================================
# DATASET
# =============================================================================
class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        crops = extract_faces_from_video(path, n_frames=FRAMES_PER_VIDEO)
        if crops:
            img = random.choice(crops)
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), label

# =============================================================================
# MODEL
# =============================================================================
def build_model():
    print("Building Xception model…")
    m = timm.create_model("xception", pretrained=False, num_classes=2)
    
    # Priority 1: Resume from the fine-tuned weights (Epoch 10)
    if os.path.exists(MODEL_OUT):
        state = torch.load(MODEL_OUT, map_location="cpu")
        m.load_state_dict(state, strict=False)
        print(f"  [RESUME] Loaded weights from {MODEL_OUT}")
        return m, True # Resuming
    
    # Priority 2: Standard pre-trained base
    if os.path.exists(MODEL_BASE):
        state = torch.load(MODEL_BASE, map_location="cpu")
        m.load_state_dict(state, strict=False)
        print(f"  Loaded base weights from {MODEL_BASE}")
    else:
        print("  Warning: No weights found, starting from scratch.")
    
    return m, False # Not resuming

def freeze_backbone(m):
    for p in m.parameters():
        p.requires_grad = False
    for p in m.get_classifier().parameters():
        p.requires_grad = True
    print("Backbone frozen.")

def unfreeze_all(m):
    for p in m.parameters():
        p.requires_grad = True
    print("All layers unfrozen.")

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train():
    # Gather videos from BOTH Celeb-DF and FaceForensics++
    real_videos = glob.glob(os.path.join(REAL_DIR, "*.mp4")) + \
                  glob.glob(os.path.join(FF_REAL_DIR, "**", "*.mp4"), recursive=True)
    
    fake_videos = glob.glob(os.path.join(FAKE_DIR, "*.mp4")) + \
                  glob.glob(os.path.join(FF_FAKE_DIR, "**", "*.mp4"), recursive=True)

    if not real_videos and not fake_videos:
        print("ERROR: No dataset found.")
        return

    print(f"Dataset — REAL: {len(real_videos)}, FAKE: {len(fake_videos)}")
    all_samples = [(v, 1) for v in real_videos] + [(v, 0) for v in fake_videos]
    random.shuffle(all_samples)

    val_n = max(1, int(len(all_samples) * VAL_SPLIT))
    val_samples   = all_samples[:val_n]
    train_samples = all_samples[val_n:]

    train_tf = transforms.Compose([
        RandomJPEGCompression(quality_range=(30, 92)),   # simulate screenshots
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomApply([RandomGaussianNoise(std_range=(0.0, 0.04))], p=0.4),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = DeepfakeDataset(train_samples, train_tf)
    val_ds   = DeepfakeDataset(val_samples,   val_tf)

    labels       = [s[1] for s in train_samples]
    class_count  = [labels.count(0), labels.count(1)]
    class_weight = [1.0 / max(c, 1) for c in class_count]
    sample_w     = [class_weight[l] for l in labels]
    sampler      = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, is_resumed = build_model()
    
    if is_resumed:
        unfreeze_all(model)
        start_epoch = 11
        current_lr = RESUME_LR
    else:
        freeze_backbone(model)
        start_epoch = 1
        current_lr = 1e-4

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr, weight_decay=WEIGHT_DECAY)
    
    # Slower decay for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining session: Epoch {start_epoch} to {TARGET_EPOCHS}\n")

    for epoch in range(start_epoch, TARGET_EPOCHS + 1):
        if not is_resumed and epoch == UNFREEZE_EPOCH + 1:
            unfreeze_all(model)
            optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=WEIGHT_DECAY)

        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{TARGET_EPOCHS} [Train]")
        
        last_log_p = -1
        for i, (imgs, labels) in enumerate(bar):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)
            
            # Progress reporting (Keep file log but remove disruptive terminal print)
            p = int(100. * (i + 1) / len(train_loader))
            if p % 2 == 0 and p != last_log_p:
                with open(os.path.join(_ROOT, "training_progress.txt"), "w") as f:
                    f.write(f"Epoch {epoch}: {p}% Complete\n")
                last_log_p = p

            bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*t_correct/t_total:.1f}%")

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{TARGET_EPOCHS} [Val]  "):
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        e_val_loss = v_loss / max(v_total, 1)
        val_acc    = 100. * v_correct / max(v_total, 1)
        
        print(f"\nEpoch {epoch}: Train Loss={t_loss/t_total:.4f} | Val Loss={e_val_loss:.4f} Val Acc={val_acc:.2f}%")

        scheduler.step(e_val_loss)
        early_stopping(e_val_loss)
        
        if e_val_loss < best_val_loss:
            best_val_loss = e_val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✓ Best model updated → {MODEL_OUT}")

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    print(f"Session complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
