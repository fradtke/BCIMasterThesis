from scipy.io import loadmat

import numpy as np
import os

import data_reader_functions


class DataPreparator:
    def __init__(self, general_data_set_sample_split):
        self.general_data_split = general_data_set_sample_split

    def construct_general_data_sample(self, data_sample):
        _train_data = []
        _train_label = []
        _val_data = []
        _val_label = []
        for index, data_set in enumerate(data_sample):
            trainings_data_set = data_sample.get(data_set)[0]
            # Hier findet aktuell der Datensplit statt, mit dem die Einzelnen Versuchspersonen dann aufgetrennt werden
            # Hier könnte ich dann eine Methode mit übergeben,
            # die dann vorher vielleicht sogar den Datensatz irgendwie randomieisiert
            if index < self.general_data_split:
                _train_data.append(trainings_data_set[0])
                _train_label.append(trainings_data_set[1])
            else:
                _val_data.append(trainings_data_set[0])
                _val_label.append(trainings_data_set[1])

        train_data = np.concatenate(_train_data, axis=0)
        train_label = np.concatenate(_train_label, axis=0)
        val_data = np.concatenate(_val_data, axis=0)
        val_label = np.concatenate(_val_label, axis=0)

        return [(train_data, train_label), (val_data, val_label)]

    def extract_individual_data(self, data_settings):
        result = getattr(data_reader_functions, data_settings.DATA_READ_METHOD)(data_settings)
        return result

    def load_eeg_data_set(self, data_settings):
        # Procedure:
        # 1. Read Data
        # 2. Construct data sample for individual subject
        # 3. Construct the general data sample
        individual_data_set = self.extract_individual_data(
            # method_name=data_settings.DATA_READ_METHOD,
            # file_path=data_settings.DATA_FOLDER_PATH,
            data_settings=data_settings
        )

        general_data_set = self.construct_general_data_sample(individual_data_set)

        return individual_data_set, general_data_set
