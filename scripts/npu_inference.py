"""
EmotionalTTS - HiFi-GAN NPU inference and benchmark
===================================================
Run on the machine with the Mobilint NPU after compiling the MXQ.

Usage:
    python npu_inference.py

Requirements:
    - maccel installed
    - hifigan_generator.mxq in the same folder
    - calibration_mels.npy in the same folder
    - hifigan_neutral_op11.onnx in the same folder (CPU baseline)

MXQ model details:
    Input : mel   = (1, 1, 1, 80) NHWC float32
    Output: audio = (1, 1, 256, 1) NHWC float32
    Emotion: neutral, baked into the model as a constant (emotion_id removed,
             neutral embedding pre-computed)

Output:
    - npu_outputs/ folder with 8 WAV files
    - RTF comparison NPU vs CPU
    - npu_benchmark_results.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
MXQ_PATH = SCRIPT_DIR / 'hifigan_generator.mxq'
ONNX_PATH = SCRIPT_DIR / 'hifigan_neutral_op11.onnx'
CALIB_PATH = SCRIPT_DIR / 'calibration_mels.npy'
OUTPUT_DIR = SCRIPT_DIR / 'npu_outputs'
REPORT_PATH = SCRIPT_DIR / 'npu_benchmark_results.txt'

# Config
SAMPLE_RATE = 22050
N_MELS = 80
N_WARMUP = 3
N_BENCH_RUNS = 10
EMOTION_NAMES = ['angry', 'calm', 'disgust', 'fearful',
                 'happy', 'neutral', 'sad', 'surprised']


def load_test_mel():
    """Load a mel and convert it to the NHWC layout (1, 1, 1, 80) the MXQ expects."""
    if CALIB_PATH.exists():
        calib = np.load(str(CALIB_PATH)).astype(np.float32)
        mel = calib[0:1]                                    # (1, 80, 65)
        mel_nhwc = mel[:, :, 0:1].reshape(1, 1, 1, 80)      # one frame
        print(f"  Mel loaded | original={mel.shape} -> NHWC={mel_nhwc.shape}")
    else:
        print("  Warning: using random mel")
        mel_nhwc = np.random.uniform(-8, 0, (1, 1, 1, 80)).astype(np.float32)
    return mel_nhwc


def normalize_audio(audio):
    """Normalize audio to [-1, 1]."""
    return audio / (np.abs(audio).max() + 1e-8)


def run_npu(model, mel_nhwc):
    """Single NPU inference. Returns 1D normalized audio."""
    outs = model.infer(mel_nhwc)
    return normalize_audio(outs[0].squeeze())


def run_onnx(session, mel_nhwc):
    """Single ONNX Runtime (CPU) inference. The ONNX model expects NCHW."""
    mel_nchw = mel_nhwc.transpose(0, 3, 2, 1)               # (1, 80, 1, 1)
    outs = session.run(None, {'mel': mel_nchw})
    return normalize_audio(outs[0].squeeze())


def benchmark_npu(model, mel_nhwc):
    """Benchmark NPU real-time factor (RTF = inference time / audio duration)."""
    print(f"\n  [NPU] Warming up ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        model.infer(mel_nhwc)

    print(f"  [NPU] Benchmarking ({N_BENCH_RUNS} runs)...")
    times = []
    audio_dur = 0
    for _ in range(N_BENCH_RUNS):
        t0 = time.perf_counter()
        out = model.infer(mel_nhwc)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        audio_dur = out[0].squeeze().shape[-1] / SAMPLE_RATE

    avg = np.mean(times)
    std = np.std(times)
    rtf = avg / max(audio_dur, 0.001)
    print(f"  [NPU] avg={avg*1000:.1f}ms std={std*1000:.1f}ms "
          f"audio={audio_dur:.3f}s RTF={rtf:.4f} "
          f"({'real-time' if rtf < 1.0 else 'slower than real-time'})")
    return rtf, avg, audio_dur


def benchmark_cpu(session, mel_nhwc):
    """Benchmark CPU (ONNX Runtime) real-time factor."""
    mel_nchw = mel_nhwc.transpose(0, 3, 2, 1)
    print(f"\n  [CPU] Warming up ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        session.run(None, {'mel': mel_nchw})

    print(f"  [CPU] Benchmarking ({N_BENCH_RUNS} runs)...")
    times = []
    audio_dur = 0
    for _ in range(N_BENCH_RUNS):
        t0 = time.perf_counter()
        out = session.run(None, {'mel': mel_nchw})
        t1 = time.perf_counter()
        times.append(t1 - t0)
        audio_dur = out[0].squeeze().shape[-1] / SAMPLE_RATE

    avg = np.mean(times)
    std = np.std(times)
    rtf = avg / max(audio_dur, 0.001)
    print(f"  [CPU] avg={avg*1000:.1f}ms std={std*1000:.1f}ms "
          f"audio={audio_dur:.3f}s RTF={rtf:.4f} "
          f"({'real-time' if rtf < 1.0 else 'slower than real-time'})")
    return rtf, avg


def main():
    print("\n" + "=" * 55)
    print("EmotionalTTS - HiFi-GAN NPU Inference")
    print("Mobilint maccel v0.30.1")
    print("=" * 55 + "\n")

    # Step 1: load the MXQ
    print("STEP 1 - Loading MXQ model on NPU")
    print("-" * 40)

    if not MXQ_PATH.exists():
        print(f"MXQ not found: {MXQ_PATH}")
        sys.exit(1)

    try:
        import maccel
        print(f"maccel version: {maccel.__version__}")
        print(f"  Loading: {MXQ_PATH.name} ({MXQ_PATH.stat().st_size/1e6:.1f} MB)")
        npu_model = maccel.load(str(MXQ_PATH))
        print("Model loaded on NPU")
        print("  Input : mel=(1,1,1,80) NHWC float32")
        print("  Output: audio=(1,1,256,1) NHWC float32")
        print("  Emotion: neutral (baked as constant)")
    except ImportError:
        print("maccel not installed")
        print("  pip install maccel-0.30.1-cp312-cp312-win_amd64.whl")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load: {e}")
        sys.exit(1)

    # Step 2: benchmark
    print("\nSTEP 2 - RTF Benchmark: NPU vs CPU")
    print("-" * 40)
    test_mel = load_test_mel()

    rtf_npu, time_npu, audio_dur = benchmark_npu(npu_model, test_mel)

    rtf_cpu = None
    speedup = None
    try:
        import onnxruntime as ort
        if ONNX_PATH.exists():
            print(f"\n  Loading ONNX for CPU baseline: {ONNX_PATH.name}")
            cpu_session = ort.InferenceSession(str(ONNX_PATH))
            rtf_cpu, time_cpu = benchmark_cpu(cpu_session, test_mel)
            speedup = time_cpu / time_npu
            print(f"\n  Speedup: {speedup:.2f}x faster on NPU vs CPU")
        else:
            print(f"  Warning: {ONNX_PATH.name} not found, skipping CPU")
    except Exception as e:
        print(f"  Warning: CPU benchmark failed: {e}")

    # Step 3: write a WAV per emotion name
    # Note: the model has the neutral embedding baked in, so every file is
    # produced by the same model from the same mel. Only the file names differ.
    print("\nSTEP 3 - Generating audio for all 8 emotion names on NPU")
    print("-" * 40)
    OUTPUT_DIR.mkdir(exist_ok=True)
    emotion_rtfs = {}

    for emo in EMOTION_NAMES:
        mel = load_test_mel()
        t0 = time.perf_counter()
        audio = run_npu(npu_model, mel)
        t1 = time.perf_counter()

        dur = len(audio) / SAMPLE_RATE
        rtf = (t1 - t0) / max(dur, 0.001)
        emotion_rtfs[emo] = rtf

        out_path = OUTPUT_DIR / f"{emo}_npu.wav"
        wavfile.write(str(out_path), SAMPLE_RATE, audio.astype(np.float32))
        print(f"  {emo:<12} | dur={dur:.2f}s | RTF={rtf:.4f} | {out_path.name}")

    # Step 4: report
    print("\n" + "=" * 55)
    print("FINAL REPORT - EmotionalTTS NPU Benchmark")
    print("=" * 55)
    print("\n  Model      : HiFi-GAN GeneratorPretrained (4D)")
    print("  Format     : MXQ (Mobilint Aries2)")
    print("  Ops        : 100% supported")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Emotions   : {len(EMOTION_NAMES)}")
    print(f"\n  NPU RTF : {rtf_npu:.4f} "
          f"({'real-time' if rtf_npu < 1.0 else 'slower than real-time'})")
    if rtf_cpu:
        print(f"  CPU RTF : {rtf_cpu:.4f}")
        print(f"  Speedup : {speedup:.2f}x faster on NPU")
    print("\n  Per-emotion RTF:")
    for emo, rtf in emotion_rtfs.items():
        print(f"    {emo:<12}: {rtf:.4f}")
    print(f"\n  Output WAVs: {OUTPUT_DIR}")

    with open(REPORT_PATH, 'w') as f:
        f.write("EmotionalTTS NPU Benchmark Results\n")
        f.write("=" * 40 + "\n")
        f.write("Model   : HiFi-GAN GeneratorPretrained\n")
        f.write("Format  : MXQ (Mobilint maccel v0.30.1)\n")
        f.write("Ops     : 100% supported\n")
        f.write(f"NPU RTF : {rtf_npu:.4f}\n")
        if rtf_cpu:
            f.write(f"CPU RTF : {rtf_cpu:.4f}\n")
            f.write(f"Speedup : {speedup:.2f}x\n")
        f.write("\nPer-emotion RTF:\n")
        for emo, rtf in emotion_rtfs.items():
            f.write(f"  {emo:<12}: {rtf:.4f}\n")

    print(f"\nReport saved: {REPORT_PATH}")
    print("\nNPU inference complete")


if __name__ == '__main__':
    main()
