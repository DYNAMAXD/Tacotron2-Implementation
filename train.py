"""
train.py
--------
Train Tacotron2 on IISc SYSPIN Hindi dataset.

Run after prep_data.py:
    python train.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import Tacotron2, Tacotron2Config
from dataset import TTSDataset, TTSCollator, BatchSampler, denormalize
from tokenizer_hindi import HindiTokenizer


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def save_log(message):
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "train_mel_loss":    [],
        "train_postnet_loss":[],
        "train_gate_loss":   [],
        "train_total_loss":  [],
        "val_mel_loss":      [],
        "val_postnet_loss":  [],
        "val_gate_loss":     [],
        "val_total_loss":    [],
        "epochs":            [],
    }


def plot_metrics(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = metrics["epochs"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tacotron2 Hindi Training Metrics", fontsize=14)

    pairs = [
        ("train_mel_loss",    "val_mel_loss",     "Mel Loss",         axes[0, 0]),
        ("train_postnet_loss","val_postnet_loss",  "PostNet Mel Loss", axes[0, 1]),
        ("train_gate_loss",   "val_gate_loss",     "Gate Loss",        axes[1, 0]),
        ("train_total_loss",  "val_total_loss",    "Total Loss",       axes[1, 1]),
    ]

    for train_key, val_key, title, ax in pairs:
        ax.plot(epochs, metrics[train_key], label="Train", color="steelblue")
        ax.plot(epochs, metrics[val_key],   label="Val",   color="tomato", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_alignment_plot(mel_true, mel_pred, attention, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Epoch {epoch} - Validation Sample", fontsize=13)

    im0 = axes[0].imshow(mel_true.numpy(), aspect="auto", origin="lower")
    axes[0].set_title("True Mel")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mel_pred.numpy(), aspect="auto", origin="lower")
    axes[1].set_title("Predicted Mel (PostNet)")
    axes[1].set_ylabel("Mel bins")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(attention.numpy(), aspect="auto", origin="lower")
    axes[2].set_title("Attention Alignment")
    axes[2].set_ylabel("Encoder step")
    axes[2].set_xlabel("Decoder step")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    path = os.path.join(save_dir, f"alignment_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=100)
    plt.close()


def alignment_score(attn):
    """
    Diagonal alignment metric: higher is better.
    Measures how monotonic the attention is (score close to 1 = perfect diagonal).
    """
    T_dec, T_enc = attn.shape
    if T_enc == 0 or T_dec == 0:
        return 0.0
    t_steps = torch.arange(T_dec, dtype=torch.float32)
    s_steps = torch.arange(T_enc, dtype=torch.float32)
    expected_pos = (t_steps / T_dec) * T_enc
    peak_pos     = torch.argmax(attn, dim=1).float()
    error        = torch.abs(peak_pos - expected_pos).mean().item()
    score        = max(0.0, 1.0 - error / T_enc)
    return score


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.AUDIO_OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    save_log(f"Device: {device}")

    # ── Tokenizer ──
    assert os.path.exists(config.VOCAB_PATH), \
        f"Vocab not found at {config.VOCAB_PATH}. Run prep_data.py first."
    tokenizer = HindiTokenizer(vocab_path=config.VOCAB_PATH)

    # ── Model ──
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    save_log(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        eps=config.ADAM_EPS,
    )

    # ── Resume from checkpoint ──
    start_epoch = 0
    ckpt_files  = sorted(
        [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0
    )
    if ckpt_files:
        latest = os.path.join(config.CHECKPOINT_DIR, ckpt_files[-1])
        print(f"Resuming from checkpoint: {latest}")
        save_log(f"Resuming from checkpoint: {latest}")

        ckpt = torch.load(
            latest,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")
        save_log(f"Resuming from epoch {start_epoch}")

    # ── Datasets ──
    train_csv = os.path.join(config.DATA_DIR, "train_metadata.csv")
    val_csv   = os.path.join(config.DATA_DIR, "val_metadata.csv")

    trainset = TTSDataset(
        train_csv, tokenizer,
        sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
        window_size=config.WIN_SIZE,    hop_size=config.HOP_LENGTH,
        fmin=config.FMIN,              fmax=config.FMAX,
        num_mels=config.N_MELS,        min_db=config.MIN_DB,
        max_scaled_abs=config.MAX_SCALED_ABS,
    )
    valset = TTSDataset(
        val_csv, tokenizer,
        sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
        window_size=config.WIN_SIZE,    hop_size=config.HOP_LENGTH,
        fmin=config.FMIN,              fmax=config.FMAX,
        num_mels=config.N_MELS,        min_db=config.MIN_DB,
        max_scaled_abs=config.MAX_SCALED_ABS,
    )

    collate_fn  = TTSCollator(tokenizer) 
    # batch_sampler = BatchSampler(trainset, batch_size=config.BATCH_SIZE, drop_last=True)

    trainloader = DataLoader(
        trainset,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        shuffle=False,          # no shuffle
        drop_last=False,
        pin_memory=device.type == "cuda",
    )


    valloader = DataLoader(
        valset,
        batch_size=max(1, config.BATCH_SIZE // 2),
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    metrics_path = os.path.join(config.METRICS_DIR, "metrics.json")
    metrics = load_metrics(metrics_path)

    print(f"\nStarting training — epochs {start_epoch} to {config.NUM_EPOCHS - 1}\n")
    save_log(f"\nStarting training — epochs {start_epoch} to {config.NUM_EPOCHS - 1}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        running = {"mel": 0.0, "postnet": 0.0, "gate": 0.0, "total": 0.0}
        n_steps = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch:04d} [Train]", leave=True, dynamic_ncols=True)
        for texts, text_lens, mels, stops, enc_mask, dec_mask in pbar:
            texts    = texts.to(device)
            mels     = mels.to(device)
            stops    = stops.to(device)
            enc_mask = enc_mask.to(device)
            dec_mask = dec_mask.to(device)

            mel_out, mel_postnet, stop_preds, _ = model(
                texts, text_lens, mels, enc_mask, dec_mask
            )

            mel_loss    = F.mse_loss(mel_out,     mels)
            postnet_loss = F.mse_loss(mel_postnet, mels)
            gate_loss   = F.binary_cross_entropy_with_logits(
                stop_preds.reshape(-1, 1), stops.reshape(-1, 1)
            )
            total_loss = mel_loss + postnet_loss + gate_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_THRESH)
            optimizer.step()

            running["mel"]     += mel_loss.item()
            running["postnet"] += postnet_loss.item()
            running["gate"]    += gate_loss.item()
            running["total"]   += total_loss.item()
            n_steps += 1

            pbar.set_postfix({
                "total": f"{total_loss.item():.4f}",
                "mel":   f"{mel_loss.item():.4f}",
                "pnet":  f"{postnet_loss.item():.4f}",
                "gate":  f"{gate_loss.item():.4f}",
            })

        avg_train = {k: v / n_steps for k, v in running.items()}

        # ── Validate ──
        model.eval()
        val_running = {"mel": 0.0, "postnet": 0.0, "gate": 0.0, "total": 0.0}
        val_align_scores = []
        n_val = 0
        first_batch = True

        with torch.no_grad():
            for texts, text_lens, mels, stops, enc_mask, dec_mask in tqdm(
                valloader, desc=f"Epoch {epoch:04d} [Val]  ", leave=False, dynamic_ncols=True
            ):
                texts    = texts.to(device)
                mels     = mels.to(device)
                stops    = stops.to(device)
                enc_mask = enc_mask.to(device)
                dec_mask = dec_mask.to(device)

                mel_out, mel_postnet, stop_preds, attn_weights = model(
                    texts, text_lens, mels, enc_mask, dec_mask
                )
                mel_loss     = F.mse_loss(mel_out,     mels)
                postnet_loss = F.mse_loss(mel_postnet, mels)
                gate_loss    = F.binary_cross_entropy_with_logits(
                    stop_preds.reshape(-1, 1), stops.reshape(-1, 1)
                )
                total_loss = mel_loss + postnet_loss + gate_loss

                val_running["mel"]     += mel_loss.item()
                val_running["postnet"] += postnet_loss.item()
                val_running["gate"]    += gate_loss.item()
                val_running["total"]   += total_loss.item()

                # Alignment score for first sample in batch
                attn_sample = attn_weights[0].cpu()   # (T_dec, T_enc)
                val_align_scores.append(alignment_score(attn_sample))

                if first_batch:
                    save_alignment_plot(
                        mel_true=denormalize(mels[0].T.cpu()),
                        mel_pred=denormalize(mel_postnet[0].T.cpu()),
                        attention=attn_weights[0].T.cpu(),
                        epoch=epoch,
                        save_dir=config.AUDIO_OUT_DIR,
                    )
                    first_batch = False

                n_val += 1

        avg_val   = {k: v / n_val for k, v in val_running.items()}
        avg_align = sum(val_align_scores) / len(val_align_scores)
        elapsed   = time.time() - epoch_start

        
        print(
            f"Epoch {epoch:04d} | "
            f"Train Total={avg_train['total']:.4f}  Mel={avg_train['mel']:.4f}  "
            f"PNet={avg_train['postnet']:.4f}  Gate={avg_train['gate']:.4f} | "
            f"Val Total={avg_val['total']:.4f}  Align={avg_align:.3f} | "
            f"Time={elapsed:.1f}s"
        )
        save_log(
            f"Epoch {epoch:04d} | "
            f"Train Total={avg_train['total']:.4f}  Mel={avg_train['mel']:.4f}  "
            f"PNet={avg_train['postnet']:.4f}  Gate={avg_train['gate']:.4f} | "
            f"Val Total={avg_val['total']:.4f}  Align={avg_align:.3f} | "
            f"Time={elapsed:.1f}s"
        )

        # ── Record metrics ──
        metrics["epochs"].append(epoch)
        metrics["train_mel_loss"].append(avg_train["mel"])
        metrics["train_postnet_loss"].append(avg_train["postnet"])
        metrics["train_gate_loss"].append(avg_train["gate"])
        metrics["train_total_loss"].append(avg_train["total"])
        metrics["val_mel_loss"].append(avg_val["mel"])
        metrics["val_postnet_loss"].append(avg_val["postnet"])
        metrics["val_gate_loss"].append(avg_val["gate"])
        metrics["val_total_loss"].append(avg_val["total"])
        save_metrics(metrics, metrics_path)

        # ── Plot ──
        graph_path = plot_metrics(metrics, config.METRICS_DIR)

        # ── Save checkpoint ──
        if (epoch + 1) % config.SAVE_EVERY_N == 0 or epoch == config.NUM_EPOCHS - 1:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_{epoch:04d}.pt")
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":          cfg,
                "metrics": {
                    "val_total_loss":  avg_val["total"],
                    "alignment_score": avg_align,
                },
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            save_log(f"Checkpoint saved: {ckpt_path}")

    print("\nTraining complete.")
    save_log("\nTraining complete.")
    print(f"Metrics : {metrics_path}")
    save_log(f"Metrics : {metrics_path}")
    print(f"Graphs  : {graph_path}")
    save_log(f"Graphs  : {graph_path}")
    print(f"Checkpoints: {config.CHECKPOINT_DIR}/")
    save_log(f"Checkpoints: {config.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    train()
