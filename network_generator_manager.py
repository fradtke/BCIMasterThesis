import os
import pickle

import pandas as pd

import numpy as np

import Properties_settings.bci_utility as utility
from classifier_networks import ClassifierGenerator
from conditional_wasserstein_gan import ConditionalWassersteinGanGenerator


def create_store_path(file_path, date_folder, name):
    _store_dir = os.path.abspath(
        os.path.join(os.sep, os.path.dirname(os.path.abspath(file_path)), date_folder, name))
    _store_dir = os.path.normpath(_store_dir)
    os.makedirs(_store_dir)
    return _store_dir


class NetworkGeneratorManager:
    def __init__(self,
                 # DATA PREPARATION
                 file_storing_path,
                 data_set_name,
                 number_of_classes,
                 # GAN SETTINGS
                 batch_size,
                 eeg_channel_number,
                 eeg_time_steps,
                 # num_classes,
                 latent_dimensions,
                 # CLASSIFIER GENERATOR SETTINGS
                 network_settings,
                 expand_dimension=-1,
                 ):

        self.data_set_name = data_set_name
        self.file_storing_path = f'Data/{file_storing_path}/{data_set_name}'
        self.store_temp_dir = create_store_path(self.file_storing_path, data_set_name, 'temp')
        self.result_values_path = create_store_path(self.file_storing_path, data_set_name, 'result_values')
        self.generated_model_path = create_store_path(self.file_storing_path, data_set_name, 'generated_models')
        self.cnn_rnn_name = 'CNN_RNN'
        self.cnn_name = 'CNN'
        self.augmentation_identifier = 'AUG'
        self.train_history_dict = 'history'

        self.classifier_network_generator = ClassifierGenerator(
            # TODO: Hier noch mal die Settings dateien beide überprüfen
            number_of_classes=number_of_classes,
            expand_dimension=expand_dimension,
            network_settings=network_settings
        )

        self.cw_gan_generator = ConditionalWassersteinGanGenerator(
            batch_size=batch_size,
            eeg_channel_number=eeg_channel_number,
            eeg_time_steps=eeg_time_steps,
            num_classes=number_of_classes,
            latent_dimensions=latent_dimensions,
            network_settings=network_settings
        )

    def generate_result_path(self, dir_name):
        store_path = f'{self.result_values_path}/{dir_name}'
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        return store_path

    def store_result_data(self, storing_data, naming, dir_name):
        _model = storing_data.get(utility.HYPER_MODEL)
        _model.save(f'{self.generated_model_path}/{naming}')

        _result_values = storing_data.get(utility.BEST_HYPER_PARAMETERS).values
        _result_values[utility.BEST_EPOCH] = storing_data.get(utility.BEST_EPOCH)
        _result_values[utility.LOSS] = storing_data.get(utility.EVAL_RESULT)[0]
        _result_values[utility.ACCURACY] = storing_data.get(utility.EVAL_RESULT)[1]
        _result_values_df = pd.DataFrame.from_dict(_result_values, orient='index')

        json_store_path = self.generate_result_path(dir_name)
        history_store_path = self.generate_result_path(self.train_history_dict)

        with open(f'{json_store_path}/{naming}.json', mode='w') as f:
            _result_values_df.to_json(f)

        with open(f'{history_store_path}/{naming}', mode='wb') as history_data:
            pickle.dump(storing_data.get(utility.HISTORY).history, history_data)

    def store_network_results(self, individual_models, general_model, naming):

        for result in individual_models:
            self.store_result_data(
                storing_data=individual_models.get(result),
                naming=f'{naming}_{result}',
                dir_name=naming
            )

        self.store_result_data(
            storing_data=general_model,
            naming=f'{naming}_General',
            dir_name=naming
        )

    def store_aug_network_results(self, model_data, naming):
        self.store_result_data(
            storing_data=model_data,
            naming=f'{self.augmentation_identifier}_{naming}',
            dir_name=self.augmentation_identifier
        )

    def create_simple_cnn_models(self, individual_data, general_data):
        individual_models, general_model = self.classifier_network_generator.create_simple_cnn_models(
            individual_data=individual_data,
            general_data=general_data,
            store_dir=self.store_temp_dir,
            name=self.cnn_name
        )
        self.store_network_results(individual_models, general_model, self.cnn_name)
        return individual_models, general_model

    def create_cnn_rnn_models(self, individual_data, general_data):

        individual_models, general_model = self.classifier_network_generator.create_cnn_rnn_models(
            individual_data=individual_data,
            general_data=general_data,
            store_dir=self.store_temp_dir,
            name=self.cnn_rnn_name
        )
        self.store_network_results(individual_models, general_model, self.cnn_rnn_name)
        return individual_models, general_model

    def create_cw_gan_and_artificial_data(self, data, name, gen_sample_number, is_float_64):
        self.cw_gan_generator.generate_cw_gan(naming=name, data_sample=data, is_float_64=is_float_64)
        generated_data = self.cw_gan_generator.generate_augmented_data(number_of_samples_per_class=gen_sample_number)
        return generated_data

    def create_models_with_artificial_data(self, real_data, artificial_data):
        simple_cnn_aug_model, cnn_rnn_aug_model = self.classifier_network_generator.create_networks_with_aug(
            real_data=real_data,
            artificial_data=artificial_data,
            store_dir=self.store_temp_dir,
            cnn_name=self.cnn_name,
            cnn_rnn_name=self.cnn_rnn_name,
            augmentation_identifier=self.augmentation_identifier
        )
        self.store_aug_network_results(
            model_data=simple_cnn_aug_model,
            naming=self.cnn_name
        )

        self.store_aug_network_results(
            model_data=cnn_rnn_aug_model,
            naming=self.cnn_rnn_name
        )

        return simple_cnn_aug_model, cnn_rnn_aug_model

    def combine_result_data(self):
        data_frame_list = {}
        _result_values_df = pd.DataFrame(columns=['Name',
                                                  'Type',
                                                  'Data',
                                                  'Num_f_layers',
                                                  'Num_s_layers',
                                                  'Num_Best_Epoch',
                                                  'Accuracy'])
        for root, dirs, files in os.walk(self.result_values_path):
            for file in files:
                if file.endswith('.json'):
                    data_frame_list[file.split('.')[0]] = pd.read_json(root + os.sep + file)

        for ele in data_frame_list:
            _hdf = data_frame_list.get(ele)
            print(ele)
            if ele.startswith(self.cnn_rnn_name):
                ann_type = 'cnn_rnn'
            else:
                ann_type = 'cnn'

            if ele.__contains__(self.augmentation_identifier):
                data_type = 'aug'
            else:
                data_type = 'normal'

            _result_values_df = _result_values_df.append({'Name': ele,
                                                          'Type': ann_type,
                                                          'Data': data_type,
                                                          'Num_f_layers': _hdf.loc[utility.FIRST_LAYERS][0],
                                                          'Num_s_layers': _hdf.loc[utility.SECOND_LAYERS][0],
                                                          'Num_Best_Epoch': _hdf.loc[utility.BEST_EPOCH],
                                                          'Accuracy': _hdf.loc[utility.ACCURACY][0]},
                                                         ignore_index=True)

        _store_path = f'{self.result_values_path}'
        if not os.path.exists(_store_path):
            os.makedirs(_store_path)

        with open(f'{_store_path}/{self.data_set_name}.json', mode='w') as f:
            _result_values_df.to_json(f)

        print(_result_values_df)


