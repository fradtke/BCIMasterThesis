from tensorflow import keras
from kerastuner.tuners import Hyperband

import numpy as np

import Properties_settings.bci_utility as utility


class ClassifierGenerator:
    def __init__(self,
                 number_of_classes,
                 network_settings,
                 expand_dimension=-1
                 ):

        self.trainings_epochs = network_settings.HYPERBAND_TRAININGS_EPOCHS
        self.tuner_epochs = network_settings.HYPERBAND_TUNER_EPOCHS
        self.tuner_validation_split = network_settings.HYPERBAND_TUNER_VALIDATION_SPLIT
        self.expand_dimension = expand_dimension
        self.number_of_classes = number_of_classes
        self.network_settings = network_settings

    def basic_cnn_model(self, hp):
        # activation_functions = ['relu']
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(
            filters=self.network_settings.CNN_FILTERS_LAYER_1,
            kernel_size=self.network_settings.CNN_KERNELS_LAYER_1,
            activation=hp.Choice(
                'conv_activation_0',
                # values=['relu', 'tanh', 'sigmoid'],
                values=self.network_settings.CNN_ACTIVATION_FUNCTIONS_VALUES,
                default=self.network_settings.CNN_ACTIVATION_FUNCTIONS_DEFAULT
            )
        ))
        if self.network_settings.CNN_POOLING_LAYER == utility.MAX_POLLING_LAYER:
            model.add(keras.layers.MaxPooling1D(
                pool_size=self.network_settings.CNN_POOLING_SIZE,
                strides=self.network_settings.CNN_POOLING_STRIDES
            ))
        elif self.network_settings.CNN_POOLING_LAYER == utility.AVERAGE_POLLING_LAYER:
            model.add(keras.layers.AveragePooling1D(
                pool_size=self.network_settings.CNN_POOLING_SIZE,
                strides=self.network_settings.CNN_POOLING_STRIDES
            ))
        for i in range(hp.Int(
                utility.FIRST_LAYERS,
                self.network_settings.CNN_FIRST_LAYERS_LOWER_LIMIT,
                self.network_settings.CNN_FIRST_LAYERS_UPPER_LIMIT
        )):
            model.add(keras.layers.Conv1D(
                filters=hp.Choice(
                    f'conv_filters_{i + 1}',
                    values=self.network_settings.CNN_FIRST_LAYERS_FILTERS_CHOICE_VALUES,
                    default=self.network_settings.CNN_FIRST_LAYERS_FILTERS_CHOICE_DEFAULT
                ),
                kernel_size=self.network_settings.CNN_FIRST_LAYERS_KERNEL_SIZE,
                activation=hp.Choice(
                    f'conv_activation_{i + 1}',
                    # values=['relu', 'tanh', 'sigmoid'],
                    values=self.network_settings.CNN_ACTIVATION_FUNCTIONS_VALUES,
                    default=self.network_settings.CNN_ACTIVATION_FUNCTIONS_DEFAULT
                )))
            if self.network_settings.CNN_POOLING_LAYER == utility.MAX_POLLING_LAYER:
                model.add(keras.layers.MaxPooling1D(
                    pool_size=self.network_settings.CNN_POOLING_SIZE,
                    strides=self.network_settings.CNN_POOLING_STRIDES
                ))
            elif self.network_settings.CNN_POOLING_LAYER == utility.AVERAGE_POLLING_LAYER:
                model.add(keras.layers.AveragePooling1D(
                    pool_size=self.network_settings.CNN_POOLING_SIZE,
                    strides=self.network_settings.CNN_POOLING_STRIDES
                ))
        model.add(keras.layers.Flatten())

        for i in range(hp.Int(
                utility.SECOND_LAYERS,
                self.network_settings.CNN_SECOND_LAYERS_LOWER_LIMIT,
                self.network_settings.CNN_SECOND_LAYERS_UPPER_LIMIT
        )):
            model.add(keras.layers.Dense(
                units=hp.Int(
                    f'dense_units_{i + 1}',
                    min_value=self.network_settings.CNN_SECOND_LAYERS_UNITS_CHOICE_MIN,
                    max_value=self.network_settings.CNN_SECOND_LAYERS_UNITS_CHOICE_MAX,
                    step=self.network_settings.CNN_SECOND_LAYERS_UNITS_CHOICE_STEP),
                activation=hp.Choice(
                    f'dense_activation_{i + 1}',
                    # values=['relu', 'tanh', 'sigmoid'],
                    values=self.network_settings.CNN_ACTIVATION_FUNCTIONS_VALUES,
                    default=self.network_settings.CNN_ACTIVATION_FUNCTIONS_DEFAULT)
            ))
        model.add(keras.layers.Dense(self.number_of_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', self.network_settings.CNN_LEARNING_RATE_CHOICE)),
            loss=self.network_settings.CNN_LOSS_FUNCTION,
            metrics=['accuracy'])

        return model

    def cnn_rnn_model(self, hp):
        # activation_functions = ['relu']
        model = keras.Sequential()

        model.add(keras.layers.TimeDistributed(
            keras.layers.Conv1D(
                filters=self.network_settings.CNN_LSTM_FILTERS_LAYER_1,
                kernel_size=self.network_settings.CNN_LSTM_KERNELS_LAYER_1,
                activation=hp.Choice(
                    'conv_activation_0',
                    # values=['relu', 'tanh', 'sigmoid'],
                    values=self.network_settings.CNN_LSTM_ACTIVATION_FUNCTIONS_VALUES,
                    default=self.network_settings.CNN_LSTM_ACTIVATION_FUNCTIONS_DEFAULT)
            )
        ))
        if self.network_settings.CNN_LSTM_POOLING_LAYER == utility.MAX_POLLING_LAYER:
            model.add(keras.layers.TimeDistributed(
                keras.layers.MaxPooling1D(
                    pool_size=self.network_settings.CNN_LSTM_POOLING_SIZE,
                    strides=self.network_settings.CNN_LSTM_POOLING_STRIDES
                )))
        elif self.network_settings.CNN_LSTM_POOLING_LAYER == utility.AVERAGE_POLLING_LAYER:
            model.add(keras.layers.TimeDistributed(
                keras.layers.AveragePooling1D(
                    pool_size=self.network_settings.CNN_LSTM_POOLING_SIZE,
                    strides=self.network_settings.CNN_LSTM_POOLING_STRIDES
                )))
        for i in range(hp.Int(
                utility.FIRST_LAYERS,
                self.network_settings.CNN_LSTM_FIRST_LAYERS_LOWER_LIMIT,
                self.network_settings.CNN_LSTM_FIRST_LAYERS_UPPER_LIMIT
        )):
            model.add(keras.layers.TimeDistributed(
                keras.layers.Conv1D(
                    filters=
                    hp.Choice(
                        f'conv_filters_{i + 1}',
                        values=self.network_settings.CNN_LSTM_FIRST_LAYERS_FILTERS_CHOICE_VALUES,
                        default=self.network_settings.CNN_LSTM_FIRST_LAYERS_FILTERS_CHOICE_DEFAULT
                    ),
                    kernel_size=self.network_settings.CNN_LSTM_FIRST_LAYERS_KERNEL_SIZE,
                    activation=hp.Choice(
                        f'conv_activation_{i + 1}',
                        # values=['relu', 'tanh', 'sigmoid'],
                        values=self.network_settings.CNN_LSTM_ACTIVATION_FUNCTIONS_VALUES,
                        default=self.network_settings.CNN_LSTM_ACTIVATION_FUNCTIONS_DEFAULT
                    )
                )
            ))
            if self.network_settings.CNN_LSTM_POOLING_LAYER == utility.MAX_POLLING_LAYER:
                model.add(keras.layers.TimeDistributed(
                    keras.layers.MaxPooling1D(
                        pool_size=self.network_settings.CNN_LSTM_POOLING_SIZE,
                        strides=self.network_settings.CNN_LSTM_POOLING_STRIDES
                    )))
            elif self.network_settings.CNN_LSTM_POOLING_LAYER == utility.AVERAGE_POLLING_LAYER:
                model.add(keras.layers.TimeDistributed(
                    keras.layers.AveragePooling1D(
                        pool_size=self.network_settings.CNN_LSTM_POOLING_SIZE,
                        strides=self.network_settings.CNN_LSTM_POOLING_STRIDES
                    )))
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

        for i in range(hp.Int(
                utility.SECOND_LAYERS,
                self.network_settings.CNN_LSTM_SECOND_LAYERS_LOWER_LIMIT,
                self.network_settings.CNN_LSTM_SECOND_LAYERS_UPPER_LIMIT
        )):
            model.add(keras.layers.LSTM(
                units=hp.Int(
                    f'lstm_units_{i + 1}',
                    min_value=self.network_settings.CNN_LSTM_SECOND_LAYERS_UNITS_CHOICE_MIN,
                    max_value=self.network_settings.CNN_LSTM_SECOND_LAYERS_UNITS_CHOICE_MAX,
                    step=self.network_settings.CNN_LSTM_SECOND_LAYERS_UNITS_CHOICE_STEP
                ),
                activation=hp.Choice(
                    f'lstm_activation_{i + 1}',
                    # values=['relu', 'tanh', 'sigmoid'],
                    values=self.network_settings.CNN_LSTM_ACTIVATION_FUNCTIONS_VALUES,
                    default=self.network_settings.CNN_LSTM_ACTIVATION_FUNCTIONS_DEFAULT),
                return_sequences=True
            ))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.number_of_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice(
                    'learning_rate',
                    self.network_settings.CNN_LSTM_LEARNING_RATE_CHOICE
                )),
            loss=self.network_settings.CNN_LSTM_LOSS_FUNCTION,
            metrics=['accuracy'])
        # print(model.summary())
        return model

    def train_the_network(self,
                          model_architecture,
                          store_dir,
                          network_name,
                          trainings_data,
                          trainings_label,
                          validation_data,
                          validation_label
                          ):

        tuner = Hyperband(
            model_architecture,
            objective=self.network_settings.HYPERBAND_OBJECTIVE,
            max_epochs=self.network_settings.HYPERBAND_MAX_EPOCHS,
            factor=self.network_settings.HYPERBAND_FACTOR,
            hyperband_iterations=self.network_settings.HYPERBAND_ITERATIONS,
            directory=store_dir,
            project_name=network_name
        )

        tuner.search(trainings_data, trainings_label,
                     epochs=self.tuner_epochs, validation_split=self.tuner_validation_split,
                     callbacks=[keras.callbacks.EarlyStopping(monitor=utility.VAL_LOSS, patience=8)])

        best_hyper_parameters = tuner.get_best_hyperparameters()[0]

        # Build the model with the optimal hyperparameters and train it on the data
        model = tuner.hypermodel.build(best_hyper_parameters)
        history = model.fit(
            trainings_data,
            trainings_label,
            epochs=self.network_settings.HYPERBAND_TUNER_FIT_EPOCHS,
            validation_split=self.tuner_validation_split)

        val_acc_per_epoch = history.history[utility.VAL_ACCURACY]
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hyper_model = tuner.hypermodel.build(best_hyper_parameters)

        if self.trainings_epochs == 0:
            train_epoch = best_epoch
        else:
            train_epoch = self.trainings_epochs

        # Retrain the model
        trainings_history = hyper_model.fit(
            trainings_data,
            trainings_label,
            epochs=train_epoch,
            validation_split=self.network_settings.HYPERBAND_TUNER_TRAININGS_SPLIT
        )

        eval_result = hyper_model.evaluate(validation_data, validation_label)
        print("[test loss, test accuracy]:", eval_result)

        result_set = {
            utility.HYPER_MODEL: hyper_model,
            utility.BEST_HYPER_PARAMETERS: best_hyper_parameters,
            utility.EVAL_RESULT: eval_result,
            utility.BEST_EPOCH: best_epoch,
            utility.HISTORY: trainings_history
        }

        return result_set

    def create_individual_classifier(self, prepared_data_set, store_dir, network, network_name):
        _results = {}
        for data_sample in prepared_data_set:
            _trainings_data_set = prepared_data_set.get(data_sample)[0]
            _validation_data_set = prepared_data_set.get(data_sample)[1]

            train_data = _trainings_data_set[0]
            train_label = _trainings_data_set[1]
            val_data = _validation_data_set[0]
            val_label = _validation_data_set[1]

            _results[data_sample] = self.train_the_network(network,
                                                           store_dir,
                                                           f'{network_name}_{data_sample}',
                                                           train_data,
                                                           train_label,
                                                           val_data,
                                                           val_label
                                                           )

        return _results

    def create_general_classifier(self, data_sample_for_general_network, store_dir, network, network_name):
        train_data = data_sample_for_general_network[0][0]
        train_label = data_sample_for_general_network[0][1]
        val_data = data_sample_for_general_network[1][0]
        val_label = data_sample_for_general_network[1][1]

        # (model_architecture, store_dir, network_name, trainings_data, trainings_label,
        #  validation_data, validation_label):
        return self.train_the_network(network,
                                      store_dir,
                                      network_name,
                                      train_data,
                                      train_label,
                                      val_data,
                                      val_label
                                      )

    def create_network_models(self, individual_data, general_data, store_dir, network, naming):
        individual_models = self.create_individual_classifier(
            individual_data, store_dir, network, naming)
        general_model = self.create_general_classifier(
            general_data, store_dir, network, f'{naming}_general')

        return individual_models, general_model

    def expand_data_dimension(self, data_tuple):
        # This step is necessary to make it possible to read the data as the time sequential data:
        # https://stackoverflow.com/questions/48140989/keras-lstm-input-dimension-setting
        output = [(np.expand_dims(data_tuple[0][0], self.expand_dimension), data_tuple[0][1]),
                  (np.expand_dims(data_tuple[1][0], self.expand_dimension), data_tuple[1][1])
                  ]
        return output

    def create_simple_cnn_models(self, individual_data, general_data, store_dir, name):
        return self.create_network_models(individual_data, general_data, store_dir, self.basic_cnn_model,
                                          name)

    def create_cnn_rnn_models(self, individual_data, general_data, store_dir, name):
        exp_individual_data = {}
        for sample in individual_data:
            exp_individual_data[sample] = self.expand_data_dimension(individual_data.get(sample))

        exp_general_data = self.expand_data_dimension(general_data)

        return self.create_network_models(exp_individual_data, exp_general_data, store_dir,
                                          self.cnn_rnn_model,
                                          name)

    def create_networks_with_aug(self, real_data, artificial_data, store_dir, cnn_name, cnn_rnn_name,
                                 augmentation_identifier):
        # Data preparation
        _con_data = np.concatenate((real_data[0][0], artificial_data[0]))
        _con_label = np.concatenate((real_data[0][1], artificial_data[1]))
        data_for_training_aug = [
            (_con_data, _con_label),
            real_data[1]
        ]
        exp_data_for_training_aug = self.expand_data_dimension(data_for_training_aug)

        # Network generation
        simple_cnn_aug_model = self.create_general_classifier(
            data_for_training_aug, store_dir, self.basic_cnn_model, f'{augmentation_identifier}_{cnn_name}')

        cnn_rnn_aug_model = self.create_general_classifier(
            exp_data_for_training_aug, store_dir, self.cnn_rnn_model, f'{augmentation_identifier}_{cnn_rnn_name}')

        return simple_cnn_aug_model, cnn_rnn_aug_model
