# ENVIRONMENT
SEED = 42
MAC = True

# CHANGING NSIDES MEANS RERUNNING DATA_INIT
NSIDES = 64    # Project computation scales as O(NSIDES**2) right now
# CHANGING NSIDES MEANS RERUNNING DATA_INIT

# DATA GENERATION
SCALE = True
STACK_SIZE = 5000
NUM_SAMPLES = 1000

# DATA HANDLING
BATCH_SIZE = 4
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# ENCODING/DECODING
IN_CHANNELS = 4
BASE_CHANNELS = 2*NSIDES
KERNEL_SIZE = 5
SAMPLE_FACTOR = 2
BIAS = True
GATE_RESOLUTION = NSIDES//4

# GENERAL ARCHITECTURE
NUM_LEVELS = 4
N_CONV_LAYERS_IN_ONE_BLOCK = 1
DROP_RATE = 0.2
DENSE_DIM = 3 * NSIDES

# TRAINING
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
ENABLE_EARLY_STOPPING = True
PATIENCE = 7
GRADIENT_CLIP_VAL = 1.5
WEIGHT_DECAY = 7e-3
PRECISION = '16-mixed' if not MAC else None      # lightning Trainer precision argument

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




