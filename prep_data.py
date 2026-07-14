"""
prep_data.py
------------
Converts the IISc SYSPIN Hindi JSON transcript file into
train/val CSV splits that dataset.py can read.

Run this FIRST before training.

Usage:
    python prep_data.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import random
import pandas as pd
from tqdm import tqdm

import config
from tokenizer_hindi import HindiTokenizer


def build_metadata(wav_dir, json_path):
    """Read JSON transcripts and match with wav files."""
    print(f"Reading transcripts from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcripts = data["Transcripts"]

    rows = [] 
    missing = 0


    # for utt_id, info in tqdm(transcripts.items(), desc="Scanning utterances"):
    #     wav_file = os.path.join(wav_dir, f"{utt_id}.wav")
    #     if not os.path.isfile(wav_file):
    #         missing += 1
    #         continue
    #     text = info["Transcript"].strip()
    #     if not text:
    #         continue
    #     rows.append({"file_path": wav_file, "transcript": text})

    # if config.MAX_SAMPLES is not None:
    #     rows = rows[:config.MAX_SAMPLES]
    #     print(f"Using {len(rows)} samples (MAX_SAMPLES limit)")
    #     missing = 0

    for utt_id, info in tqdm(transcripts.items(), desc="Scanning utterances"):

        if config.MAX_SAMPLES is not None and len(rows) >= config.MAX_SAMPLES:
            break

        wav_file = os.path.join(wav_dir, f"{utt_id}.wav")

        if not os.path.isfile(wav_file):
            missing += 1
            continue

        text = info["Transcript"].strip()

        if text:
            rows.append({
                "file_path": wav_file,
                "transcript": text
            })

    print(f"Found {len(rows)} valid utterances  |  Missing wav files: {missing}")
    return rows


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    rows = build_metadata(config.WAV_DIR, config.JSON_PATH)

    if len(rows) == 0:
        print("ERROR: No data found. Check WAV_DIR and JSON_PATH in config.py")
        return

    # Shuffle and split
    random.seed(config.SEED)
    # random.shuffle(rows)

    n_val = max(1, int(len(rows) * config.TEST_SPLIT_PCT))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)

    train_path = os.path.join(config.DATA_DIR, "train_metadata.csv")
    val_path   = os.path.join(config.DATA_DIR, "val_metadata.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Saved to      : {train_path}, {val_path}")

    # Build and save vocabulary
    print("\nBuilding vocabulary from training transcripts...")
    tokenizer = HindiTokenizer()
    tokenizer.build_from_texts(train_df["transcript"].tolist())
    #tokenizer.save(config.VOCAB_PATH)
    print(f"\nVOCAB_SIZE = {tokenizer.vocab_size}")
    print(">>> Update VOCAB_SIZE in config.py to this value before training! <<<")


if __name__ == "__main__":
    main()
