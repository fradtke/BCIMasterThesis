from scipy.io import loadmat

import numpy as np
import os

FUNCTION_NAME = 'data_processing_imagined_speech'


# Imagined Speech BCI Contest from 2020???
def read_data_set(folder_path, train_folder_name, validate_folder_name):
    train_file_folder = f'{folder_path}{train_folder_name}'
    validate_file_folder = f'{folder_path}{validate_folder_name}'

    data_set = {}

    # This procedure requires that in each is data sample is represent in both folders,
    # just for training and validation procedure
    for file in os.listdir(train_file_folder):
        print(f'Is reading File: {file.split(".")[0]}')
        data_set[file.split('.')[0]] = [
            loadmat(f'{train_file_folder}{file}', mat_dtype=True).get('epo_train'),
            loadmat(f'{validate_file_folder}{file}', mat_dtype=True).get('epo_validation')
        ]

    return data_set


def relabel_values(labels):
    # The imaginary speech is labeled with a binary system,
    # a one in a specific position represents the related label -> one-hot labels
    # (should be just a single one)
    _labels = np.array(np.zeros(labels.shape[0]))
    for trial in range(len(labels)):
        value = 0
        for i in range(len(labels[trial])):
            value += i * labels[trial][i]
        _labels[trial] = value
    return _labels


def construct_individual_data_set(data_set):
    output_data_set = {}
    for data_sample in data_set:
        trainings_data_set = data_set.get(data_sample)[0]
        validation_data_set = data_set.get(data_sample)[1]
        # Extracting the data from the reading process
        # [0][0] -> is requiered to get the first elements from the ndarrays to reach finally the array
        # we need to transpose the data
        training_data = np.transpose(trainings_data_set[0][0][4])
        validation_data = np.transpose(validation_data_set[0][0][4])

        # The labels need to be rewritten,
        # which allows us to have a single number as a label an not a one in a cell in a table
        training_labels = relabel_values(np.transpose(trainings_data_set[0][0][5]))
        validation_labels = relabel_values(np.transpose(validation_data_set[0][0][5]))

        output_data_set[data_sample] = [
            (training_data, training_labels),
            (validation_data, validation_labels)
        ]

    return output_data_set


def process_data_set(data_settings):
    return construct_individual_data_set(
        read_data_set(
            folder_path=data_settings.DATA_FOLDER_PATH,
            train_folder_name=data_settings.TRAIN_DATA_FOLDER,
            validate_folder_name=data_settings.VALIDATION_DATA_FOLDER
        )
    )
