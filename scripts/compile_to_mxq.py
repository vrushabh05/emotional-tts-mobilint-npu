"""
EmotionalTTS - HiFi-GAN ONNX to MXQ conversion (first, two-input version)
=========================================================================
Run inside the Mobilint qubee Docker container.

Usage (inside Docker):
    cd /NPU_Projects
    python compile_to_mxq.py

Input files needed:
    - hifigan_generator.onnx (about 26 MB)
    - calibration_mels.npy   (about 2 MB)

Output:
    - hifigan_generator.mxq

Note: this script calibrates a model with two inputs (mel and emotion_id).
The model that was benchmarked in npu_inference.py was a later variant with the
neutral emotion embedding baked in as a constant, so it takes only the mel.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ONNX_PATH = SCRIPT_DIR / 'hifigan_generator.onnx'
CALIB_PATH = SCRIPT_DIR / 'calibration_mels.npy'
MXQ_PATH = SCRIPT_DIR / 'hifigan_generator.mxq'

# qubee needs calibration as a folder of .npz files
CALIB_DIR = SCRIPT_DIR / 'calib_data'
SAVE_DIR = SCRIPT_DIR / 'mxq_output'

EMOTION_NAMES = ['angry', 'calm', 'disgust', 'fearful',
                 'happy', 'neutral', 'sad', 'surprised']


def build_calibration_folder():
    """
    Save calibration data as a folder of .npz files.
    Each .npz has keys matching the model input names:
        'mel'        : (1, 80, 65) float32
        'emotion_id' : (1,) int64
    """
    print("\nSTEP 2 - Building calibration folder...")

    CALIB_DIR.mkdir(exist_ok=True)
    for f in CALIB_DIR.glob('*.npz'):
        f.unlink()

    calib = np.load(str(CALIB_PATH)).astype(np.float32)
    print(f"  Loaded: {calib.shape} samples")

    for i in range(len(calib)):
        mel_sample = calib[i:i + 1]                          # (1, 80, 65)
        emotion_id = np.array([i % 8], dtype=np.int64)       # cycle 0-7
        npz_path = CALIB_DIR / f'sample_{i:04d}.npz'
        np.savez(str(npz_path), mel=mel_sample, emotion_id=emotion_id)

    saved = len(list(CALIB_DIR.glob('*.npz')))
    print(f"Saved {saved} calibration files to: {CALIB_DIR}")
    print("  Each file: mel=(1,80,65) float32 + emotion_id=(1,) int64")
    return str(CALIB_DIR)


def main():
    print("\n" + "=" * 55)
    print("EmotionalTTS - HiFi-GAN ONNX to MXQ Compilation")
    print("Using Mobilint qubee.mxq_compile v0.12.0.0")
    print("=" * 55 + "\n")

    # Step 1: verify input files
    print("STEP 1 - Checking input files...")

    if not ONNX_PATH.exists():
        print(f"ONNX not found: {ONNX_PATH}")
        sys.exit(1)
    print(f"ONNX  : {ONNX_PATH.name} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")

    if not CALIB_PATH.exists():
        print(f"Calibration not found: {CALIB_PATH}")
        sys.exit(1)
    calib = np.load(str(CALIB_PATH))
    print(f"Calib : {CALIB_PATH.name} shape={calib.shape} "
          f"range=[{calib.min():.2f}, {calib.max():.2f}]")

    # Step 2: calibration folder
    calib_dir_path = build_calibration_folder()

    # Step 3: compile
    print("\nSTEP 3 - Running qubee.mxq_compile...")
    print(f"  ONNX      : {ONNX_PATH}")
    print(f"  Calib dir : {calib_dir_path}")
    print(f"  Save dir  : {SAVE_DIR}")
    print("  Running... (may take a few minutes)\n")

    SAVE_DIR.mkdir(exist_ok=True)

    try:
        import qubee
        print(f"qubee version: {qubee.__version__}")

        qubee.mxq_compile(
            model=str(ONNX_PATH),
            save_path=str(SAVE_DIR),
            calib_data_path=calib_dir_path,
        )

        mxq_files = list(SAVE_DIR.glob('*.mxq'))
        if not mxq_files:
            mxq_files = list(SAVE_DIR.rglob('*.mxq'))

        if mxq_files:
            shutil.copy(str(mxq_files[0]), str(MXQ_PATH))
            size_mb = MXQ_PATH.stat().st_size / 1e6
            print("\nMXQ compiled successfully")
            print(f"  File : {MXQ_PATH}")
            print(f"  Size : {size_mb:.1f} MB")
        elif MXQ_PATH.exists():
            size_mb = MXQ_PATH.stat().st_size / 1e6
            print("\nMXQ compiled")
            print(f"  File : {MXQ_PATH}")
            print(f"  Size : {size_mb:.1f} MB")
        else:
            print(f"\nMXQ not found, check {SAVE_DIR}")
            print("  Contents:")
            for f in SAVE_DIR.rglob('*'):
                print(f"    {f}")

    except ImportError:
        print("qubee not installed")
        print("  pip install qubee-0.12.0.0+aries2-py3-none-any.whl")
        sys.exit(1)
    except Exception as e:
        print(f"Compilation failed: {e}")
        print("\n  To see the full signature run:")
        print("  python3 -c \"import qubee; help(qubee.mxq_compile)\"")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("DONE - exit Docker, then run npu_inference.py")
    print("=" * 55)


if __name__ == '__main__':
    main()
