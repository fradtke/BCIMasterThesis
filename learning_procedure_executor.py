import tensorflow as tf
import datetime

from data_preparator import DataPreparator
from network_generator_manager import NetworkGeneratorManager


class LearningProcedureExecutor:
    def __init__(self):
        print('I Bims am executoren')

    @staticmethod
    def conduct_network_learning_procedure_for_data_sample(
            parameter_settings,
            network_settings
    ):
        #TODO: Hier muss ich die auskommentierten elemente wieder hinzufügen
        #TODO: ADD to Settings
        if parameter_settings.IS_FLOAT64:
            tf.keras.backend.set_floatx('float64')
        data_prep = DataPreparator(general_data_set_sample_split=parameter_settings.GENERAL_DATA_SET_SAMPLE_SPLIT)
        network_generator = NetworkGeneratorManager(
            # DATA PREPARATION
            file_storing_path=f'{datetime.datetime.now().strftime("%m%d-%H%M%S")}',
            data_set_name=parameter_settings.DATA_SET_NAME,
            number_of_classes=parameter_settings.NUMBER_OF_CLASSES,
            # GAN SETTINGS
            batch_size=parameter_settings.BATCH_SIZE,
            eeg_channel_number=parameter_settings.EEG_CHANNEL_NUMBER,
            eeg_time_steps=parameter_settings.EEG_TIME_STEPS,
            latent_dimensions=parameter_settings.LATENT_DIMENSIONS,
            # CLASSIFIER GENERATOR SETTINGS
            network_settings=network_settings
        )
        individual_data, grouped_data = data_prep.load_eeg_data_set(
            data_settings=parameter_settings
        )

        # network_generator.create_simple_cnn_models(
        #     individual_data=individual_data,
        #     general_data=grouped_data
        # )
        #
        network_generator.create_cnn_rnn_models(
            individual_data=individual_data,
            general_data=grouped_data
        )

        # Execute GAN Procedure
        # augmented_data = network_generator.create_cw_gan_and_artificial_data(
        #     data=grouped_data[0],
        #     name=parameter_settings.DATA_SET_NAME,
        #     gen_sample_number=parameter_settings.NUMBER_OF_SAMPLES_PER_CLASS,
        #     is_float_64=parameter_settings.IS_FLOAT64
        # )

        # Classifier with augmented data

        # network_generator.create_models_with_artificial_data(
        #     real_data=grouped_data,
        #     artificial_data=augmented_data
        # )
        # network_generator.combine_result_data()
