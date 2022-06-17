# DATA PREPARATION
DATA_SET_NAME = 'ISC2020'
GENERAL_DATA_SET_SAMPLE_SPLIT = 2  # 12 TODO: Should be 12 current just for testing
DATA_FOLDER_PATH = 'Data_Sets/test_sample_for_procedure/'  # 'Imagined_speech_challenge_2020/'
DATA_READ_METHOD = 'data_processing_imagined_speech'

# COMMON CLASSIFIER SETTINGS
EXPAND_DIMENSIONS = -1  # TODO: maybe 3???
IS_FLOAT64 = True
NUMBER_OF_CLASSES = 5

# CW-GAN SETTINGS
BATCH_SIZE = 32
EEG_CHANNEL_NUMBER = 64
EEG_TIME_STEPS = 795

LATENT_DIMENSIONS = 795
BUFFER_SIZE = 1024

NUMBER_OF_SAMPLES_PER_CLASS = 240  # TODO: Should be more

# RESULT SETTINGS
# TODO: Probably some results have to be customized
NETWORK_SETTINGS = 'network_settings_default'

# DATA SET SPECIFIC VALUES
TRAIN_DATA_FOLDER = 'Training_set/'
VALIDATION_DATA_FOLDER = 'Validation_set/'
