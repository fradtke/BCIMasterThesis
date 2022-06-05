import importlib
import os
import sys

from Properties_settings import bci_utility as utility


# TODO: write demonstration execution
def write_script(script_corpus, script_name_and_path):
    new_file = open(script_name_and_path, 'w')
    new_file.write(script_corpus)
    new_file.close()


def read_modules_in_folder(folder_path):
    module_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.py'):
            module_list.append(importlib.import_module(f'{folder_path.replace("/", ".")}.{file.split(".")[0]}'))
    return module_list


# TODO: Dieser Code Snippet muss noch sinnvoll ausgeführt werden
def construct_bci_master_executor(
        parameter_settings_folder=utility.PARAMETER_SETTINGS_FOLDER
):
    parameter_settings_list = read_modules_in_folder(parameter_settings_folder)
    script_corpus = f"# Import area \n" \
                    f"from learning_procedure_executor import LearningProcedureExecutor \n\n" \
                    f"# Area of settings\n"

    # Settings import
    for setting in parameter_settings_list:
        script_corpus += f"# {setting.DATA_SET_NAME} \n" \
                         f"from Properties_settings.Data_settings import {setting.__name__.split('.')[-1]}\n" \
                         f"from Properties_settings.Network_settings import {setting.NETWORK_SETTINGS}\n\n"

    script_corpus += f"\n"

    # data settings list
    script_corpus += f"# Data area \n" \
                     f"setting_list = [\n"
    for setting in parameter_settings_list:
        script_corpus += f"    # {setting.DATA_SET_NAME} \n" \
                         f"    ({setting.__name__.split('.')[-1]},\n" \
                         f"     {setting.NETWORK_SETTINGS}),\n"

    script_corpus += f"]\n\n"
    script_corpus += f"# Execution area \n" \
                     f"if __name__ == '__main__': \n" \
                     f"    for settings_pair in setting_list: \n" \
                     f"        LearningProcedureExecutor.conduct_network_learning_procedure_for_data_sample( \n" \
                     f"            parameter_settings=settings_pair[0],\n" \
                     f"            network_settings=settings_pair[1]\n" \
                     f"        )\n"

    write_script(script_corpus=script_corpus,
                 script_name_and_path='bci_master_executor.py')


def write_data_settings_file(
        data_set_name,
        general_data_set_sample_split,
        data_folder_path,
        # train_data_folder,
        # validation_data_folder,
        data_read_method,
        expand_dimensions,
        number_of_classes,
        batch_size,
        eeg_channel_number,
        eeg_time_steps,
        latent_dimensions,
        buffer_size,
        number_of_samples_per_class,
        file_naming,
        network_settings='network_settings_default',
        is_float64=False,
        data_specific_values=None
):
    script_corpus = f"# DATA PREPARATION\n" \
                    f"DATA_SET_NAME = '{data_set_name}'\n" \
                    f"GENERAL_DATA_SET_SAMPLE_SPLIT = {general_data_set_sample_split}\n" \
                    f"DATA_FOLDER_PATH = '{data_folder_path}'\n" \
                    f"DATA_READ_METHOD = '{data_read_method}'\n\n" \
                    f"# COMMON CLASSIFIER SETTINGS\n" \
                    f"EXPAND_DIMENSIONS = {expand_dimensions}\n" \
                    f"IS_FLOAT64 = {is_float64}\n" \
                    f"NUMBER_OF_CLASSES = {number_of_classes}\n\n" \
                    f"# CW-GAN SETTINGS\n" \
                    f"BATCH_SIZE = {batch_size}\n" \
                    f"EEG_CHANNEL_NUMBER = {eeg_channel_number}\n" \
                    f"EEG_TIME_STEPS = {eeg_time_steps}\n\n" \
                    f"LATENT_DIMENSIONS = {latent_dimensions}\n" \
                    f"BUFFER_SIZE = {buffer_size}\n\n" \
                    f"NUMBER_OF_SAMPLES_PER_CLASS = {number_of_samples_per_class}\n\n" \
                    f"# RESULT SETTINGS\n" \
                    f"NETWORK_SETTINGS = '{network_settings}'\n\n"

    if data_specific_values is not None:
        script_corpus += f"# DATA SET SPECIFIC VALUES\n"
        for specific in data_specific_values:
            if type(data_specific_values.get(specific)) is str:
                script_corpus += f"{str(specific).upper()} = '{data_specific_values.get(specific)}' \n"
            else:
                script_corpus += f"{str(specific).upper()} = {data_specific_values.get(specific)} \n"

    write_script(script_corpus=script_corpus,
                 script_name_and_path=f'{utility.PARAMETER_SETTINGS_FOLDER}/settings_{file_naming}.py')


def write_network_settings_file(
        hyperband_objective,
        hyperband_max_epochs,
        hyperband_factor,
        hyperband_iterations,
        hyperband_trainings_epochs,
        hyperband_tuner_epochs,
        hyperband_tuner_validation_split,
        cnn_filters_layer_1,
        cnn_kernels_layer_1,
        cnn_first_layers_lower_limit,
        cnn_first_layers_upper_limit,
        cnn_first_layers_filters_choice_values,
        cnn_first_layers_filters_choice_default,
        cnn_first_layers_kernel_size,
        cnn_second_layers_lower_limit,
        cnn_second_layers_upper_limit,
        cnn_second_layers_units_choice_min,
        cnn_second_layers_units_choice_max,
        cnn_second_layers_units_choice_step,
        cnn_activation_functions_values,
        cnn_activation_functions_default,
        cnn_learning_rate_choice,
        cnn_loss_function,
        cnn_lstm_filters_layer_1,
        cnn_lstm_kernels_layer_1,
        cnn_lstm_first_layers_lower_limit,
        cnn_lstm_first_layers_upper_limit,
        cnn_lstm_first_layers_filters_choice_values,
        cnn_lstm_first_layers_filters_choice_default,
        cnn_lstm_first_layers_kernel_size,
        cnn_lstm_second_layers_lower_limit,
        cnn_lstm_second_layers_upper_limit,
        cnn_lstm_second_layers_units_choice_min,
        cnn_lstm_second_layers_units_choice_max,
        cnn_lstm_second_layers_units_choice_step,
        cnn_lstm_activation_functions_values,
        cnn_lstm_activation_functions_default,
        cnn_lstm_learning_rate_choice,
        cnn_lstm_loss_function,
        cw_gan_trainings_epochs,
        generator_filters_layer_1,
        generator_filters_layer_2,
        generator_learning_rate,
        generator_beta_1,
        generator_beta_2,
        discriminator_filters_layer_1,
        discriminator_filters_layer_2,
        discriminator_filters_layer_3,
        discriminator_filters_layer_4,
        discriminator_learning_rate,
        discriminator_beta_1,
        discriminator_beta_2,
        discriminator_extra_steps,
        file_naming
):
    script_corpus = f"#Hyperband Settings\n" \
                    f"HYPERBAND_OBJECTIVE = '{hyperband_objective}'\n" \
                    f"HYPERBAND_MAX_EPOCHS = {hyperband_max_epochs}\n" \
                    f"HYPERBAND_FACTOR = {hyperband_factor}\n" \
                    f"HYPERBAND_ITERATIONS = {hyperband_iterations}\n\n" \
                    f"HYPERBAND_TRAININGS_EPOCHS = {hyperband_trainings_epochs}\n" \
                    f"HYPERBAND_TUNER_EPOCHS = {hyperband_tuner_epochs}\n" \
                    f"HYPERBAND_TUNER_VALIDATION_SPLIT = {hyperband_tuner_validation_split}\n\n" \
                    f"# Classifier CNN Settings\n" \
                    f"CNN_FILTERS_LAYER_1 = {cnn_filters_layer_1}\n" \
                    f"CNN_KERNELS_LAYER_1 = {cnn_kernels_layer_1}\n\n" \
                    f"CNN_FIRST_LAYERS_LOWER_LIMIT = {cnn_first_layers_lower_limit}\n" \
                    f"CNN_FIRST_LAYERS_UPPER_LIMIT = {cnn_first_layers_upper_limit}\n\n" \
                    f"CNN_FIRST_LAYERS_FILTERS_CHOICE_VALUES = {cnn_first_layers_filters_choice_values}\n" \
                    f"CNN_FIRST_LAYERS_FILTERS_CHOICE_DEFAULT = {cnn_first_layers_filters_choice_default}\n" \
                    f"CNN_FIRST_LAYERS_KERNEL_SIZE = {cnn_first_layers_kernel_size}\n\n" \
                    f"CNN_SECOND_LAYERS_LOWER_LIMIT = {cnn_second_layers_lower_limit}\n" \
                    f"CNN_SECOND_LAYERS_UPPER_LIMIT = {cnn_second_layers_upper_limit}\n\n" \
                    f"CNN_SECOND_LAYERS_UNITS_CHOICE_MIN = {cnn_second_layers_units_choice_min}\n" \
                    f"CNN_SECOND_LAYERS_UNITS_CHOICE_MAX = {cnn_second_layers_units_choice_max}\n" \
                    f"CNN_SECOND_LAYERS_UNITS_CHOICE_STEP = {cnn_second_layers_units_choice_step}\n\n" \
                    f"CNN_ACTIVATION_FUNCTIONS_VALUES = {cnn_activation_functions_values}\n" \
                    f"CNN_ACTIVATION_FUNCTIONS_DEFAULT = '{cnn_activation_functions_default}'\n\n" \
                    f"CNN_LEARNING_RATE_CHOICE = {cnn_learning_rate_choice}\n" \
                    f"CNN_LOSS_FUNCTION = '{cnn_loss_function}'\n\n" \
                    f"# Classifier CNN+LSTM Settings\n" \
                    f"CNN_LSTM_FILTERS_LAYER_1 = {cnn_lstm_filters_layer_1}\n" \
                    f"CNN_LSTM_KERNELS_LAYER_1 = {cnn_lstm_kernels_layer_1}\n\n" \
                    f"CNN_LSTM_FIRST_LAYERS_LOWER_LIMIT = {cnn_lstm_first_layers_lower_limit}\n" \
                    f"CNN_LSTM_FIRST_LAYERS_UPPER_LIMIT = {cnn_lstm_first_layers_upper_limit}\n\n" \
                    f"CNN_LSTM_FIRST_LAYERS_FILTERS_CHOICE_VALUES = {cnn_lstm_first_layers_filters_choice_values}\n" \
                    f"CNN_LSTM_FIRST_LAYERS_FILTERS_CHOICE_DEFAULT = {cnn_lstm_first_layers_filters_choice_default}\n" \
                    f"CNN_LSTM_FIRST_LAYERS_KERNEL_SIZE = {cnn_lstm_first_layers_kernel_size}\n\n" \
                    f"CNN_LSTM_SECOND_LAYERS_LOWER_LIMIT = {cnn_lstm_second_layers_lower_limit}\n" \
                    f"CNN_LSTM_SECOND_LAYERS_UPPER_LIMIT = {cnn_lstm_second_layers_upper_limit}\n\n" \
                    f"CNN_LSTM_SECOND_LAYERS_UNITS_CHOICE_MIN = {cnn_lstm_second_layers_units_choice_min}\n" \
                    f"CNN_LSTM_SECOND_LAYERS_UNITS_CHOICE_MAX = {cnn_lstm_second_layers_units_choice_max}\n" \
                    f"CNN_LSTM_SECOND_LAYERS_UNITS_CHOICE_STEP = {cnn_lstm_second_layers_units_choice_step}\n\n" \
                    f"CNN_LSTM_ACTIVATION_FUNCTIONS_VALUES = {cnn_lstm_activation_functions_values}\n" \
                    f"CNN_LSTM_ACTIVATION_FUNCTIONS_DEFAULT = '{cnn_lstm_activation_functions_default}'\n\n" \
                    f"CNN_LSTM_LEARNING_RATE_CHOICE = {cnn_lstm_learning_rate_choice}\n" \
                    f"CNN_LSTM_LOSS_FUNCTION = '{cnn_lstm_loss_function}'\n\n\n" \
                    f"# Conditional Wasserstein GAN Settings\n" \
                    f"CW_GAN_TRAININGS_EPOCHS = {cw_gan_trainings_epochs}\n\n" \
                    f"# -Generator\n" \
                    f"GENERATOR_FILTERS_LAYER_1 = {generator_filters_layer_1}\n" \
                    f"GENERATOR_FILTERS_LAYER_2 = {generator_filters_layer_2}\n\n" \
                    f"GENERATOR_LEARNING_RATE = {generator_learning_rate}\n" \
                    f"GENERATOR_BETA_1 = {generator_beta_1}\n" \
                    f"GENERATOR_BETA_2 = {generator_beta_2}\n\n" \
                    f"# -Discriminator\n" \
                    f"DISCRIMINATOR_FILTERS_LAYER_1 = {discriminator_filters_layer_1}\n" \
                    f"DISCRIMINATOR_FILTERS_LAYER_2 = {discriminator_filters_layer_2}\n" \
                    f"DISCRIMINATOR_FILTERS_LAYER_3 = {discriminator_filters_layer_3}\n" \
                    f"DISCRIMINATOR_FILTERS_LAYER_4 = {discriminator_filters_layer_4}\n\n" \
                    f"DISCRIMINATOR_LEARNING_RATE = {discriminator_learning_rate}\n" \
                    f"DISCRIMINATOR_BETA_1 = {discriminator_beta_1}\n" \
                    f"DISCRIMINATOR_BETA_2 = {discriminator_beta_2}\n\n" \
                    f"DISCRIMINATOR_EXTRA_STEPS = {discriminator_extra_steps}\n"

    write_script(script_corpus=script_corpus,
                 script_name_and_path=f'{utility.NETWORK_SETTINGS_FOLDER}/network_settings_{file_naming}.py')


def update_data_reader_functions(
        data_reading_scripts_folder=utility.DATA_READING_SCRIPTS_FOLDER
):
    module_list = read_modules_in_folder(data_reading_scripts_folder)

    script_corpus = f""
    for module in module_list:
        script_corpus += f"from {utility.DATA_READING_SCRIPTS_FOLDER} import {module.__name__.split('.')[-1]}\n"
    script_corpus += f"\n\n"
    for module in module_list:
        script_corpus += \
            f"def {module.FUNCTION_NAME}(data_settings):\n " \
            f"   return {module.__name__.split('.')[-1]}.process_data_set(data_settings)\n\n\n"

    write_script(script_corpus=script_corpus,
                 script_name_and_path='data_reader_functions.py')


def create_data_settings_and_update_executor_and_read(
        data_set_name, general_data_set_sample_split, data_folder_path, train_data_folder, validation_data_folder,
        data_read_method, expand_dimensions, number_of_classes, batch_size, eeg_channel_number, eeg_time_steps,
        latent_dimensions, buffer_size, number_of_samples_per_class, network_settings, file_naming):
    write_data_settings_file(
        data_set_name, general_data_set_sample_split, data_folder_path, train_data_folder, validation_data_folder,
        data_read_method, expand_dimensions, number_of_classes, batch_size, eeg_channel_number, eeg_time_steps,
        latent_dimensions, buffer_size, number_of_samples_per_class, network_settings, file_naming)
    update_data_reader_functions()
    construct_bci_master_executor()
