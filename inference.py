"""
inference.py
------------
Generate speech from Hindi text using a trained Tacotron2 checkpoint.

Usage:
    python inference.py --text "आपका स्वागत है" --checkpoint checkpoints/checkpoint_0099.pt
    python inference.py --text "आपका स्वागत है" --checkpoint checkpoints/checkpoint_0019.pt

    # Or run interactively:
    python inference.py --interactive
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import Tacotron2, Tacotron2Config
from tokenizer_hindi import HindiTokenizer
from dataset import AudioMelConversions, denormalize


def load_model(checkpoint_path, device):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    # ckpt = torch.load(checkpoint_path, map_location=device)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    
    tokenizer = HindiTokenizer(vocab_path=config.VOCAB_PATH)

    # Try to restore config from checkpoint, fall back to config.py
    if "config" in ckpt and isinstance(ckpt["config"], Tacotron2Config):
        cfg = ckpt["config"]
    else:
        cfg = Tacotron2Config(
            num_mels                  = config.N_MELS,
            num_chars                 = tokenizer.vocab_size,
            character_embed_dim       = config.EMBEDDING_DIM,
            pad_token_id              = tokenizer.pad_token_id,
            encoder_kernel_size       = config.ENCODER_KERNEL_SIZE,
            encoder_n_convolutions    = config.ENCODER_N_CONV,
            encoder_embed_dim         = config.EMBEDDING_DIM,
            encoder_dropout_p         = config.ENCODER_DROPOUT,
            decoder_embed_dim         = config.DECODER_RNN_DIM,
            decoder_prenet_dim        = config.PRENET_DIM,
            decoder_prenet_depth      = config.PRENET_DEPTH,
            decoder_prenet_dropout_p  = config.PRENET_DROPOUT,
            decoder_postnet_num_convs = config.POSTNET_N_CONV,
            decoder_postnet_n_filters = config.POSTNET_EMBED_DIM,
            decoder_postnet_kernel_size = config.POSTNET_KERNEL_SIZE,
            decoder_postnet_dropout_p = config.POSTNET_DROPOUT,
            decoder_dropout_p         = config.DECODER_DROPOUT,
            attention_dim             = config.ATTENTION_DIM,
            attention_location_n_filters  = config.ATTENTION_LOC_FILTERS,
            attention_location_kernel_size = config.ATTENTION_LOC_KERNEL,
            attention_dropout_p       = 0.1,
        )

    model = Tacotron2(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, tokenizer, cfg


def synthesize(text, model, tokenizer, device, max_steps=1000, griffin_lim_iters=60):
    """
    Run Tacotron2 inference + Griffin-Lim vocoder.
    Returns (audio_int16, mel_postnet, attn_weights, stats_dict)
    """
    # Tokenize
    token_ids = tokenizer.encode(text).unsqueeze(0).to(device)
    n_tokens  = token_ids.shape[1]
    n_chars   = len(text)

    t0 = time.time()

    # Forward pass
    with torch.inference_mode():
        mel_postnet, attn_weights = model.inference(token_ids, max_decode_steps=max_steps)

    t_infer = time.time() - t0

    mel_frames = mel_postnet.shape[1]
    speech_len = mel_frames * config.HOP_LENGTH / config.SAMPLE_RATE   # seconds
    rtf        = t_infer / max(speech_len, 1e-6)

    stats = {
        "Characters":    n_chars,
        "Tokens":        n_tokens,
        "Mel Frames":    mel_frames,
        "Speech Length": f"{speech_len:.2f} sec",
        "Inference Time":f"{t_infer:.2f} sec",
        "RTF":           f"{rtf:.2f}",
    }

    # Vocoder: Griffin-Lim
    mel_np = mel_postnet.squeeze(0).T.cpu()   # (n_mels, T)
    audio_proc = AudioMelConversions(
        num_mels=config.N_MELS, sampling_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT, window_size=config.WIN_SIZE, hop_size=config.HOP_LENGTH,
        fmin=config.FMIN, fmax=config.FMAX, min_db=config.MIN_DB,
        max_scaled_abs=config.MAX_SCALED_ABS,
    )
    audio_int16 = audio_proc.mel2audio(mel_np, do_denorm=True,
                                       griffin_lim_iters=griffin_lim_iters)

    return audio_int16, mel_postnet.squeeze(0).cpu(), attn_weights.squeeze(0).cpu(), stats


def save_outputs(audio, mel, attn, stats, text, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Sanitize filename
    slug = text[:30].replace(" ", "_").replace("/", "-")
    wav_path  = os.path.join(out_dir, f"{slug}.wav")
    plot_path = os.path.join(out_dir, f"{slug}_analysis.png")

    # Save audio
    wavfile.write(wav_path, config.SAMPLE_RATE, audio)

    # Save analysis plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Input: "{text}"', fontsize=11)

    mel_display = denormalize(mel.T).numpy()
    im0 = axes[0].imshow(mel_display, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Predicted Mel Spectrogram")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    attn_display = attn.numpy()   # (T_dec, T_enc)
    im1 = axes[1].imshow(attn_display.T, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Attention Alignment (encoder steps x decoder steps)")
    axes[1].set_xlabel("Decoder step")
    axes[1].set_ylabel("Encoder step")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    return wav_path, plot_path


def print_stats(stats, text):
    print("\n" + "="*45)
    print("Generation Statistics")
    print("="*45)
    print(f"  Input text : {text}")
    for k, v in stats.items():
        print(f"  {k:<16}: {v}")
    print("="*45 + "\n")


def get_latest_checkpoint():
    if not os.path.isdir(config.CHECKPOINT_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0,
    )
    return os.path.join(config.CHECKPOINT_DIR, files[-1]) if files else None


def main():
    parser = argparse.ArgumentParser(description="Tacotron2 Hindi Inference")
    parser.add_argument("--text",        type=str, default=None,
                        help="Hindi text to synthesize")
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to .pt checkpoint (defaults to latest in checkpoints/)")
    parser.add_argument("--out_dir",     type=str, default="inference_out",
                        help="Directory to save wav + plots")
    parser.add_argument("--max_steps",   type=int, default=config.MAX_DECODER_STEPS)
    parser.add_argument("--gl_iters",    type=int, default=60,
                        help="Griffin-Lim iterations (more = better quality, slower)")
    parser.add_argument("--interactive", action="store_true",
                        help="Loop and synthesize multiple sentences")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or get_latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found. Train the model first.")
        return

    model, tokenizer, _ = load_model(ckpt_path, device)

    if args.interactive:
        print("\nInteractive mode. Type Hindi text and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("Text: ").strip()
            if not text or text.lower() == "quit":
                break
            audio, mel, attn, stats = synthesize(
                text, model, tokenizer, device, args.max_steps, args.gl_iters
            )
            print_stats(stats, text)
            wav_path, plot_path = save_outputs(audio, mel, attn, stats, text, args.out_dir)
            print(f"Audio saved : {wav_path}")
            print(f"Plot saved  : {plot_path}\n")
    else:
        text = args.text
        if not text:
            text = "नमस्ते, आपका स्वागत है।"
            print(f"No text provided, using default: {text}")
        audio, mel, attn, stats = synthesize(
            text, model, tokenizer, device, args.max_steps, args.gl_iters
        )
        print_stats(stats, text)
        wav_path, plot_path = save_outputs(audio, mel, attn, stats, text, args.out_dir)
        print(f"Audio saved : {wav_path}")
        print(f"Plot saved  : {plot_path}")


if __name__ == "__main__":
    main()
