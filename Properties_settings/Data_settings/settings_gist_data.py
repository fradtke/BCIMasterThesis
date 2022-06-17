# DATA PREPARATION
DATA_SET_NAME = 'GIST'
GENERAL_DATA_SET_SAMPLE_SPLIT = 39
DATA_FOLDER_PATH = 'Data_Sets/GIST/Data/'
DATA_READ_METHOD = 'data_processing_gist'

# COMMON CLASSIFIER SETTINGS
EXPAND_DIMENSIONS = -1
IS_FLOAT64 = False
NUMBER_OF_CLASSES = 2

# CW-GAN SETTINGS
BATCH_SIZE = 32
EEG_CHANNEL_NUMBER = 68
EEG_TIME_STEPS = 3584

LATENT_DIMENSIONS = 3584
BUFFER_SIZE = 1024

NUMBER_OF_SAMPLES_PER_CLASS = 2000

# RESULT SETTINGS
NETWORK_SETTINGS = 'network_settings_default'

# DATA SET SPECIFIC VALUES
LOWER_EVENT_TIME_VALUE = 2 
UPPER_EVENT_TIME_VALUE = 5 
