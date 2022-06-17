from Data_Reading_Scripts import gist_data_reader
from Data_Reading_Scripts import imagined_speech_reader


def data_processing_gist(data_settings):
    return gist_data_reader.process_data_set(data_settings)


def data_processing_imagined_speech(data_settings):
    return imagined_speech_reader.process_data_set(data_settings)


