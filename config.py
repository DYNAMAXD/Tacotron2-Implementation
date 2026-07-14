import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# THINGS TO CHANGE BEFORE RUNNING
# ============================================================
# 1. WAV_DIR        -> path to your wav folder
# 2. JSON_PATH      -> path to your transcripts JSON file
# 3. VOCAB_SIZE     -> run prep_data.py first, it will print the correct value
# 4. NUM_EPOCHS     -> start with 100, increase if needed
# 5. BATCH_SIZE     -> reduce to 8 or 4 if you get CUDA out-of-memory
# ============================================================

# ------- PATHS --------
WAV_DIR = r"../IISc_SYSPINProject_Hindi_Male_Spk001_HC/IISc_SYSPIN_Data/IISc_SYSPINProject_Hindi_Male_Spk001_HC/wav"
JSON_PATH = r"../IISc_SYSPINProject_Hindi_Male_Spk001_HC/IISc_SYSPIN_Data/IISc_SYSPINProject_Hindi_Male_Spk001_HC/IISc_SYSPINProject_Hindi_Male_Spk001_HC_Transcripts.json"
LOG_FILE = "Training_Logs.txt"
DATA_DIR = "data"
CHECKPOINT_DIR  = "checkpoints"
METRICS_DIR     = "metrics"
AUDIO_OUT_DIR   = "audio_out"
VOCAB_PATH      = "vocab.pt"

# ------- AUDIO -------
SAMPLE_RATE     = 22050
N_FFT           = 1024
WIN_SIZE        = 1024
HOP_LENGTH      = 256
FMIN            = 0
FMAX            = 8000
N_MELS          = 80
MIN_DB          = -100.0
MAX_SCALED_ABS  = 4.0
#sdcr
# ------- MODEL -------
# Set VOCAB_SIZE after running prep_data.py -- it prints the correct number
VOCAB_SIZE              = 4685     # <- CHANGE after running prep_data.py
EMBEDDING_DIM           = 512
ENCODER_N_CONV          = 3
ENCODER_KERNEL_SIZE     = 5
ENCODER_DROPOUT         = 0.5
DECODER_RNN_DIM         = 1024
DECODER_DROPOUT         = 0.1
ATTENTION_RNN_DIM       = 1024
ATTENTION_DIM           = 128
ATTENTION_LOC_FILTERS   = 32
ATTENTION_LOC_KERNEL    = 31
PRENET_DIM              = 256
PRENET_DEPTH            = 2
PRENET_DROPOUT          = 0.5
POSTNET_EMBED_DIM       = 512
POSTNET_N_CONV          = 5
POSTNET_KERNEL_SIZE     = 5
POSTNET_DROPOUT         = 0.5
MAX_DECODER_STEPS       = 1000
GATE_THRESHOLD          = 0.5

# ------- TRAINING -------
NUM_EPOCHS          = 30        # <- CHANGE: more is better, start at 100
BATCH_SIZE          = 24          # <- CHANGE: reduce to 8 or 4 if OOM on 3050
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-6
ADAM_EPS            = 1e-6
GRAD_CLIP_THRESH    = 1.0
LOG_INTERVAL        = 50           # steps
SAVE_EVERY_N        = 3          # epochs
NUM_WORKERS         = 0            # keep 0 on Windows to avoid multiprocessing issues
TEST_SPLIT_PCT      = 0.1
SEED                = 42
MAX_SAMPLES         = None             # set to None to use full dataset  