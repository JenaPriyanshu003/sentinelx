import os
import subprocess

_ROOT = r"c:\Users\krish\Desktop\MyProjects\SentinelX"
PYTHON_EXE = os.path.join(_ROOT, "venv_gpu", "Scripts", "python.exe")
DOWNLOAD_SCRIPT = os.path.join(_ROOT, "training", "download_ff.py")
TRAIN_SCRIPT = os.path.join(_ROOT, "training", "train.py")

def run_cmd(cmd):
    print(f"\n{'='*60}\nRunning: {cmd}\n{'='*60}")
    subprocess.run(cmd, env=os.environ.copy(), cwd=_ROOT)

def main():
    print("🚀 SentinelX Fully Automated Data Pipeline & Training 🚀")
    
    # 1. Complete Deepfakes if it somehow was stopped (script skips existing files)
    run_cmd(f'"{PYTHON_EXE}" "{DOWNLOAD_SCRIPT}" dataset/raw/ff_fake -d Deepfakes -c c40 -t videos --server EU2')
    
    # 2. Download FaceSwap (1,000 videos)
    print("\n[STEP 2/4] Downloading FaceSwap dataset...")
    run_cmd(f'"{PYTHON_EXE}" "{DOWNLOAD_SCRIPT}" dataset/raw/ff_fake -d FaceSwap -c c40 -t videos --server EU2')
    
    # 3. Download Original/Real videos (1,000 videos to balance)
    print("\n[STEP 3/4] Downloading Original (Real) dataset...")
    run_cmd(f'"{PYTHON_EXE}" "{DOWNLOAD_SCRIPT}" dataset/raw/ff_real -d original -c c40 -t videos --server EU2')
    
    # 4. Trigger Training (15 Epochs on combined dataset)
    print("\n[STEP 4/4] Starting 15-Epoch Training Run on Full Combined Dataset...")
    run_cmd(f'"{PYTHON_EXE}" "{TRAIN_SCRIPT}"')
    
    print("\n✅ All Automated Tasks Completed! ✅")

if __name__ == "__main__":
    main()
