import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def wav_to_rgb_spectrogram(
    wav_path, output_path,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8000,
    cmap_name="plasma"
):
    """Convert one .wav file to a 224×224 RGB spectrogram PNG."""
    # 1) Load audio
    y, _ = librosa.load(wav_path, sr=sr)

    # 2) Log-Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin, fmax=fmax,
        power=2.0
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    # 3) Normalize to [0,255]
    norm = 255 * (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-6)
    norm = norm.astype(np.uint8)

    # 4) Apply colormap
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm / 255.0)            # (n_mels, T, 4)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)

    # 5) Save as 224×224 PNG
    img = Image.fromarray(rgb)
    img = img.resize((224, 224), Image.BILINEAR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

if __name__ == "__main__":
    # Base directory containing machine folders
    base_dir   = "training_data"
    categories = ["train", "supplemental"]

    for machine in sorted(os.listdir(base_dir)):
        machine_dir = os.path.join(base_dir, machine)
        if not os.path.isdir(machine_dir):
            continue

        print(f"\nProcessing machine: {machine}")
        for cat in categories:
            wav_dir = os.path.join(machine_dir, cat)
            # output into e.g. "trainRGB" or "supplementalRGB"
            rgb_dir = os.path.join(machine_dir, f"{cat}RGB")

            if not os.path.isdir(wav_dir):
                print(f"  [skip] no folder: {wav_dir}")
                continue

            print(f"  • Converting '{cat}' → '{cat}RGB'")
            os.makedirs(rgb_dir, exist_ok=True)
            for fname in tqdm(os.listdir(wav_dir), desc=f"{machine}/{cat}", ncols=80):
                if not fname.lower().endswith(".wav"):
                    continue
                in_path  = os.path.join(wav_dir, fname)
                out_name = os.path.splitext(fname)[0] + ".png"
                out_path = os.path.join(rgb_dir, out_name)
                wav_to_rgb_spectrogram(in_path, out_path)

    print("\n✅ Spectrogram conversion complete!")
