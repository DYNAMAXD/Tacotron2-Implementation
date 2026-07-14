Tacotron2 for Hindi (IISc SYSPIN Dataset)
=========================================

Files
-----
config.py          -- all hyperparameters and paths
tokenizer_hindi.py -- akshar-based Hindi syllable tokenizer
dataset.py         -- audio loading, mel conversion, dataloader
model.py           -- Tacotron2 architecture
prep_data.py       -- convert IISc JSON to train/val CSVs + build vocab
train.py           -- training loop with metrics, checkpoints, graphs
inference.py       -- inference + Griffin-Lim + generation stats
requirements.txt   -- pip dependencies


BEFORE YOU DO ANYTHING -- READ THIS
-------------------------------------

1. INSTALL DEPENDENCIES

   pip install torch torchaudio librosa numpy pandas scikit-learn scipy matplotlib tqdm akshar

   If akshar fails to install:
       pip install akshar
   If it still fails, the tokenizer falls back to character-level automatically.
   Character-level still works for Hindi -- akshar just gives slightly better tokens.

2. FIX PATHS IN config.py

   Open config.py and set:
       WAV_DIR   = path to the folder containing your .wav files
       JSON_PATH = path to your IISc SYSPIN transcript JSON file

   Example (Windows):
       WAV_DIR   = r"C:\data\IISc_SYSPINProject_Hindi_Male_Spk001_HC\wav"
       JSON_PATH = r"C:\data\IISc_SYSPINProject_Hindi_Male_Spk001_HC\transcripts.json"

   Example (Linux/Mac):
       WAV_DIR   = "/data/IISc_SYSPIN/wav"
       JSON_PATH = "/data/IISc_SYSPIN/transcripts.json"

3. RUN PREP_DATA FIRST

       python prep_data.py

   This will:
   - Scan all wav files and match with transcripts
   - Create data/train_metadata.csv and data/val_metadata.csv
   - Build the Hindi vocabulary and save to vocab.pt
   - Print the VOCAB_SIZE number you need

4. UPDATE VOCAB_SIZE IN config.py

   After prep_data.py prints "VOCAB_SIZE = XXXX", go to config.py and set:
       VOCAB_SIZE = XXXX   (the number it printed)

5. TUNE BATCH SIZE FOR YOUR 3050

   The RTX 3050 has 4-8GB VRAM. Start with:
       BATCH_SIZE = 16
   If you get CUDA out-of-memory, reduce to:
       BATCH_SIZE = 8   or   BATCH_SIZE = 4

6. TUNE NUM_EPOCHS

   50 hours of data is a good amount. Recommended:
   - Epoch 50:  model starts learning basic prosody
   - Epoch 200: intelligible speech usually starts
   - Epoch 500: good quality (if alignment is diagonal)

   Set in config.py:
       NUM_EPOCHS = 200  (start here, extend later)


TRAINING
--------

    python train.py

Output:
- checkpoints/checkpoint_NNNN.pt  (saved every SAVE_EVERY_N epochs)
- metrics/metrics.json             (all loss values)
- metrics/training_metrics.png     (loss graphs, updated every epoch)
- audio_out/alignment_epoch_NNNN.png  (true mel vs predicted mel vs attention)


HOW TO KNOW IF TRAINING IS WORKING
------------------------------------

Watch the alignment plot (audio_out/alignment_epoch_XXXX.png).
The bottom panel shows the attention matrix.

- Early epochs (1-50):  attention looks like noise or a blurry blob -- normal
- Middle epochs (50-150): a diagonal stripe should start forming
- Good training (150+):  a clear diagonal line from top-left to bottom-right

If after 200 epochs the attention is STILL completely random/noisy:
- Reduce learning rate to 5e-4
- Try batch size 8
- Check that your wav files load at the right sample rate (22050)

The total loss should decrease from ~1.5-2.0 down toward 0.2-0.5 over training.


INFERENCE
---------

After training, synthesize speech:

    # Synthesize one sentence (uses latest checkpoint automatically)
    python inference.py --text "नमस्ते दुनिया"

    # Interactive mode
    python inference.py --interactive

    # Specify checkpoint and output folder
    python inference.py --text "आपका स्वागत है" \
                        --checkpoint checkpoints/checkpoint_0199.pt \
                        --out_dir my_outputs \
                        --gl_iters 80

Output per synthesis:
- inference_out/XXXX.wav              (the audio)
- inference_out/XXXX_analysis.png     (mel + attention plot)
- Console prints generation statistics

Griffin-Lim quality tips:
  --gl_iters 60   fast, acceptable quality
  --gl_iters 100  slower, better quality
  --gl_iters 200  best quality from Griffin-Lim


COMMON PROBLEMS
---------------

"Vocab not found"
  -> Run prep_data.py first

CUDA out of memory
  -> Reduce BATCH_SIZE in config.py (try 8, then 4)

Audio sounds like noise after 100+ epochs
  -> This is usually an alignment problem. Check the attention plots.
     Try lowering LEARNING_RATE to 5e-4 and retraining from scratch.

Audio is cut off too early
  -> Lower GATE_THRESHOLD from 0.5 to 0.3 in config.py

Audio never stops (max steps hit)
  -> Raise GATE_THRESHOLD from 0.5 to 0.6

UNK tokens everywhere in tokenizer
  -> Your text contains characters not seen in training.
     Make sure test sentences use vocabulary from training data.

Windows multiprocessing errors in DataLoader
  -> NUM_WORKERS is already set to 0 in config.py for Windows safety.


DIRECTORY STRUCTURE AFTER SETUP
---------------------------------

tacotron2_hindi/
    config.py
    tokenizer_hindi.py
    dataset.py
    model.py
    prep_data.py
    train.py
    inference.py
    requirements.txt
    vocab.pt                    (created by prep_data.py)
    data/
        train_metadata.csv      (created by prep_data.py)
        val_metadata.csv        (created by prep_data.py)
    checkpoints/
        checkpoint_0009.pt
        checkpoint_0019.pt
        ...
    metrics/
        metrics.json
        training_metrics.png
    audio_out/
        alignment_epoch_0000.png
        ...
    inference_out/
        नमस्ते_दुनिया.wav
        नमस्ते_दुनिया_analysis.png
