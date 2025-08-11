from sys import platform

# ENVIRONMENT
SEED = 42
MAC = platform == 'darwin'  # True if running on mac, False if anything else
RUN_NAME = 'herm_proto'

# CHANGING NSIDE MEANS RERUNNING DATA_INIT
NSIDE = 32    # Project computation scales as O(NSIDE**2) right now
NSIDE_EFF = 32  # cannot be more than NSIDE and cannot be less than 16
# CHANGING NSIDE MEANS RERUNNING DATA_INIT
LMAX = 3 * NSIDE_EFF - 1
BANDS = [(0, 20), (20, 40), (40, 66), (66, LMAX + 1)]

# DATA GENERATION
ROLLING = False          # enable rolling buffer training
SCALE = not ROLLING and False    # only turns on if True and ROLLING is not on
STACK_SIZE = 20000      # total dataset available to pull from
NUM_SAMPLES = 12000     # initial number of pulls to use during training/val/testing
REPLACE_FRAC = 0.05     # percent of existing training data to replace with newly generated data (0.1 means 10% replaced)
MASK_APOSCALE = 10      # apodization scale for mask
MASK_APOTYPE = 'C2'     # nmt apotype

# DATA HANDLING
BATCH_SIZE = 32
NUM_WORKERS = 16
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# ENCODING/DECODING
INPUT_DIM = 4
HIDDEN_DIM = 64

# GENERAL ARCHITECTURE
NUM_LEVELS = 2
DROP_RATE = 0.35

# TRAINING
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
ENABLE_EARLY_STOPPING = True
PATIENCE = 7
GRADIENT_CLIP_VAL = 0.5
WEIGHT_DECAY = 7e-3
CONTINUOUS = False   # resume from last.ckpt

# AUX
DATA_FILE = 'stacks.npy'
SCALER_FILE = 'scaler.pkl'     # None to turn off saving
MASK_FILE = "binary_GAL_mask_N1024.fits"

# LOGGING
ON_STEP = False

# MAC SETTINGS
if MAC:
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    PREFETCH_FACTOR = None
    PIN_MEMORY = False
    STACK_SIZE = 5000
    NUM_SAMPLES = 3000


# HPARAMS
hparams = {}


def update_hparams(new_hps):
    """
    Updates the hparams dictionary with new hparams values
    :param new_hps: dict
    :return: None
    """
    hparams.update(new_hps)
    return None




