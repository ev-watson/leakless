from sys import platform

# ── Environment ───────────────────────────────────────────────────────
SEED = 42
MAC = platform == 'darwin'
RUN_NAME = 'leakless_hrm'

# ── Resolution ────────────────────────────────────────────────────────
# Changing NSIDE requires regenerating data (data_construction)
NSIDE = 32                          # HEALPix resolution
NSIDE_EFF = 32                      # effective resolution (16 <= NSIDE_EFF <= NSIDE)
LMAX = 3 * NSIDE_EFF - 1           # maximum multipole

# ── Data generation ───────────────────────────────────────────────────
ROLLING = False
SCALE = not ROLLING and False    # only turns on if True and ROLLING is not on
STACK_SIZE = 20000
NUM_SAMPLES = 12000
REPLACE_FRAC = 0.05     # percent of existing training data to replace with newly generated data (0.1 means 10% replaced)
MASK_APOSCALE = 10      # apodization scale for mask
MASK_APOTYPE = 'C2'     # nmt apotype

# ── Data handling ─────────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_WORKERS = 16
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# ── HRM architecture ─────────────────────────────────────────────────
INPUT_DIM = 4                       # [Re_E, Im_E, Re_B, Im_B]
HIDDEN_SIZE = 128                   # transformer hidden dimension
H_LAYERS = 4                        # transformer blocks in H-module
L_LAYERS = 4                        # transformer blocks in L-module
H_CYCLES = 2                        # high-level reasoning cycles (N)
L_CYCLES = 3                        # low-level steps per cycle (T)
NUM_HEADS = 4                       # attention heads
EXPANSION = 4.0                     # SwiGLU expansion ratio
ROPE_THETA = 10000.0                # RoPE base frequency
MAX_SEQ_LEN = 8192                  # RoPE cache (>= alm_len)
N_SUPERVISION = 4                   # deep supervision segments (M)

# ── Training ──────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
ENABLE_EARLY_STOPPING = True
PATIENCE = 7
GRADIENT_CLIP_VAL = 1.0
WEIGHT_DECAY = 0.01
CONTINUOUS = False      # resume from last.ckpt

# ── Auxiliary files ───────────────────────────────────────────────────
DATA_FILE = 'stacks.npy'
SCALER_FILE = 'scaler.pkl'      # None to turn off saving
MASK_FILE = "binary_GAL_mask_N1024.fits"

# ── Logging ───────────────────────────────────────────────────────────
ON_STEP = False

# ── Spectral analysis bands ──────────────────────────────────────────
BANDS = [(0, 20), (20, 40), (40, 66), (66, LMAX + 1)]

# ── Mac overrides (local development) ────────────────────────────────
if MAC:
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    PREFETCH_FACTOR = None
    PIN_MEMORY = False
    STACK_SIZE = 5000
    NUM_SAMPLES = 3000
    HIDDEN_SIZE = 64
    H_LAYERS = 2
    L_LAYERS = 2
    N_SUPERVISION = 2

# ── Runtime hyperparameter store ─────────────────────────────────────
hparams = {}


def update_hparams(new_hps):
    """Merge *new_hps* into the global hparams dictionary."""
    hparams.update(new_hps)
