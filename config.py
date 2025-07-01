from sys import platform

# ENVIRONMENT
SEED = 42
MAC = platform == 'darwin'  # True if running on mac, False if anything else

# CHANGING NSIDES MEANS RERUNNING DATA_INIT
NSIDES = 64    # Project computation scales as O(NSIDES**2) right now
# CHANGING NSIDES MEANS RERUNNING DATA_INIT

# DATA GENERATION
ROLLING = not MAC and True      # does not work with scaling or mac
SCALE = not ROLLING and True    # only turns on if True and ROLLING is not on
STACK_SIZE = 50000      # total dataset available to pull from
NUM_SAMPLES = 4500     # initial number of pulls to use during training/val/testing
REPLACE_FRAC = 0.10     # percent of existing training data to replace with newly generated data (0.1 means 10% replaced)

# DATA HANDLING
BATCH_SIZE = 4
NUM_WORKERS = 16
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# ENCODING/DECODING
IN_CHANNELS = 4
BASE_CHANNELS = 2*NSIDES
KERNEL_LIST = [7, 5, 3, 1]
SAMPLE_FACTOR = 2
BIAS = True

# GENERAL ARCHITECTURE
NUM_LEVELS = 4
N_CONV_LAYERS_IN_ONE_BLOCK = 1
DROP_RATE = 0.35

# TRAINING
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
ENABLE_EARLY_STOPPING = True
PATIENCE = 7
GRADIENT_CLIP_VAL = 0.5
WEIGHT_DECAY = 7e-3

# AUX
DATA_FILE = 'stacks.npy'
SCALER_FILE = 'scaler.pkl'     # None to turn off saving

# LOGGING
ON_STEP = False

# MAC SETTINGS
if MAC:
    NUM_WORKERS = 0
    PREFETCH_FACTOR = None
    PIN_MEMORY = False
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




