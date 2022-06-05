import random
import numpy as np
import sklearn

import os
from scipy.io import loadmat

# Publication for the data set: http://dx.doi.org/10.5524/100295

FUNCTION_NAME = 'data_processing_gist'


def read_data_set(data_folder_path, identifier='eeg'):
    # TODO: hieraus könnte ich dann eine generelle methode machen, bei der ich dann die identifyer mit übergebe!!!

    data_set = {}
    # This procedure requires that in each is data sample is represent in both folders,
    # just for training and validation procedure
    for file in os.listdir(data_folder_path):
        print(f'Is reading File: {file.split(".")[0]}')
        data_set[file.split('.')[0]] = [
            loadmat(f'{data_folder_path}{file}', mat_dtype=True).get(identifier)
        ]
    return data_set


def create_labeled_data_set(data_sample, label_value):
    """
    This methods adds the label for the data sample
    :param data_sample: The already chunked data set
    :param label_value: The label value of the current data class
    :return: data_sample and the labels
    """
    label = [np.asarray(label_value)] * len(data_sample)
    return data_sample, np.asarray(label)


def create_train_and_val_set(data_set_list, split_value, label, shuffle=True):
    """
    This methods splits a list containing the complete data set to a trainings and a validation set.
    The data split is based on the split_value.
    :param data_set_list: The original data, which has to be chunked
    :param split_value: The split position in the data set list
    :param label: The label value of the current data class
    :param shuffle: Should be 'True' if the data has to be shuffled
    :return: trainings data set, validation data set
    """
    _data_set_list = data_set_list.copy()
    if shuffle:
        random.shuffle(_data_set_list)
    split_position_value = int(len(_data_set_list) * split_value)

    return (create_labeled_data_set(_data_set_list[:split_position_value], label)), \
           (create_labeled_data_set(_data_set_list[split_position_value:], label))


def shuffle_data_and_labels(data, labels):
    # TODO Add description
    return sklearn.utils.shuffle(data, labels)


def generate_shuffled_data_sets(data_dict):
    # TODO Add description
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    for data_label in data_dict:
        element = create_train_and_val_set(data_set_list=data_dict.get(data_label),
                                           label=data_label,
                                           split_value=0.8
                                           )
        train_data.extend(element[0][0])
        train_labels.extend(element[0][1])
        val_data.extend(element[1][0])
        val_labels.extend(element[1][1])

    return shuffle_data_and_labels(np.asarray(train_data), np.asarray(train_labels)), \
           shuffle_data_and_labels(np.asarray(val_data), np.asarray(val_labels))


def chunk_data_sample_to_sets(data_samples, event_idx, s_rate, lower_value, upper_value, time_points):
    data_sets = []
    for data_sample in data_samples:
        data_set_list = []
        for event_index in event_idx:
            # It could happen that the record of the first on-set ist smaller then 'event_index - 2 * s_rate - 1'
            # and for this we need the maximum
            appending_chunk = data_sample[:, int(max(0, int(event_index - lower_value * s_rate - 1))):
                                             int(event_index + upper_value * s_rate - 1)]
            if appending_chunk.shape[-1] < time_points:
                for i in range(time_points - appending_chunk.shape[-1]):
                    # Filling up the missing values from the start
                    appending_chunk = np.insert(appending_chunk, 0, 0, axis=-1)

            data_set_list.append(appending_chunk)
        data_sets.append(data_set_list)

    return data_sets


def extract_data_from_gist(data, data_settings):
    imagery_event = data.get('imagery_event')

    event_indices = np.where(imagery_event[0] == 1)[0]  # Finds all position where a 1 is in the data
    im_le_list, im_ri_list = chunk_data_sample_to_sets(
        data_samples=[data.get('imagery_left'), data.get('imagery_right')],
        event_idx=event_indices,
        s_rate=data.get('srate')[0][0],
        lower_value=data_settings.LOWER_EVENT_TIME_VALUE,
        upper_value=data_settings.UPPER_EVENT_TIME_VALUE,
        time_points=data_settings.EEG_TIME_STEPS
    )

    data_sample_dict = {0: im_le_list,
                        1: im_ri_list}

    return generate_shuffled_data_sets(data_sample_dict)


def construct_individual_data_set(data_set, data_settings):
    output_data_set = {}
    for data_sample in data_set:
        gist_data = data_set.get(data_sample)[0]
        gist_dtype = gist_data.dtype
        gist_data = {n: gist_data[n][0, 0] for n in gist_dtype.names}
        train_data, val_data = extract_data_from_gist(gist_data, data_settings)

        output_data_set[data_sample] = [
            (train_data[0], train_data[1]),
            (val_data[0], val_data[1])
        ]

    return output_data_set


def process_data_set(data_settings):
    data_set = read_data_set(data_folder_path=data_settings.DATA_FOLDER_PATH)
    return construct_individual_data_set(data_set, data_settings)
