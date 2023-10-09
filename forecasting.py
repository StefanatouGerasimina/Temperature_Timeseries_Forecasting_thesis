import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from clustering import Normalizer


def prepare_data_for_lstm(cluster: int):
    # load data from joblib obj and normalize data
    mySeries = joblib.load("joblib_objects/mySeries.joblib")
    norm = Normalizer()
    norm.__int__(scaler="minmax", mySeries=mySeries)
    norm_ma_Series, Series_arrays = norm.normalize_data()
    data_for_stations = {}
    # load namesofmyseries object
    namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")

    df = pd.read_csv(f"som/som_tuning_results_daily_som_xy_combos.csv")
    shaped_df = df[df['shape_method'] == 'manhattan']
    epoched_df = shaped_df[shaped_df['epochs'] == 100]
    daily_df = epoched_df.drop(columns=['epochs', 'shape_method', 'silhouete_score', 'quantization_error'])
    missing_cols = daily_df.columns[daily_df.isna().any()]
    daily_df = daily_df.drop(columns=missing_cols)

    # get data for stations in the cluster
    cl = f'cluster{cluster}'
    stations = daily_df[cl]
    stations_string = s = ''.join(str(x) for x in stations)
    stations = stations_string.split(" ")
    print(f"Stations for cluster {cluster}: {stations}")
    for station in stations:
        data_for_stations[station] = Series_arrays[station]
    data = []
    # first_key, data = next(iter(data_for_stations.items()))
    for key, value in data_for_stations.items():
        a = value['Temperature'].tolist()
        data.append(a)
    if len(data) > 1:
        data = np.mean(data, axis=0)
    else:
        data = data[0]

    # set the number of time steps to look back
    look_back = 3
    # split the data into training and testing sets
    train_size = int(len(data) * 0.95)
    test_size = len(data) - train_size
    data.index = pd.DatetimeIndex(data.index.values,
                                  freq=data.index.inferred_freq)
    data = data.drop(columns=['Date', 'Normalised'])
    train = data.head(train_size)
    test_size = len(data) - train_size
    test = data.tail(test_size)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # train = scaler.fit_transform(train)
    # test = scaler.transform(test)
    train = train.values.reshape(-1, 1)
    test = test.values.reshape(-1, 1)
    return train, test


def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def prepare_train_and_test_dataset(train, test, time_steps):
    # Prepare training dataset
    trainX, trainY = create_dataset(dataset=train, look_back=time_steps)
    testX, testY = create_dataset(dataset=test, look_back=time_steps)
    # Normalization
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(trainX)
    # trainX = scaler.transform(trainX)
    # trainY = scaler.fit_transform(trainY.reshape(-1, 1)).flatten()
    # scaler.fit(testX)
    # testX = scaler.transform(testX)
    # testY = scaler.fit_transform(testY.reshape(-1, 1)).flatten()

    # Reshape data
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY


def build_model(hp, Xtrain):
    model = Sequential()
    for i in range(hp.get('num_layers')):
        if i == 0:
            model.add(
                LSTM(units=hp.get('num_units'), input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
        elif i == hp.get('num_layers') - 1:
            model.add(LSTM(units=hp.get('num_units'), return_sequences=False))
        else:
            model.add(LSTM(units=hp.get('num_units'), return_sequences=True))
        model.add(Dropout(hp.get('dropout_rate')))
    model.add(Dense(1))
    model.compile(optimizer=hp.get('optimizer')(learning_rate=hp.get('learning_rate')), loss='mse', metrics=['mae'])
    return model


def find_best_grid_search(cluster):
    train, test = prepare_data_for_lstm(cluster=cluster)
    time_steps = [30, 60]
    for ts in time_steps:
        trainX, trainY, testX, testY, scaler, time_steps = prepare_train_and_test_dataset(train, test, time_steps=ts)
        hp = {
            'num_layers': [1, 2, 3],
            'num_units': [32, 64, 128],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.2, 0.3, 0.4],
            'activation': [relu, sigmoid],
            'optimizer': [Adam]
        }
        # Define the early stopping criteria
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        # Define the Random Search tuner
        tuner = RandomSearch(
            build_model(Xtrain=trainX, hp=hp),
            objective='val_loss',
            max_trials=10,
            executions_per_trial=3,
            directory='my_dir',
            project_name='my_project'
        )

        # Fit the tuner to the training data
        tuner.search(trainX, trainY, epochs=50, validation_data=(testX, testY), callbacks=[early_stop])

        # Print the summary of the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()

        # Evaluate the best model on the test data
        test_loss, test_mae = best_model.evaluate(testX, testY)
        print("Test Loss: ", test_loss)
        print("Test MAE: ", test_mae)


def create_model3(timesteps=30, neurons=50, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(1, timesteps), return_sequences=True, activation=activation))
    model.add(Dense(1))
    model.add(Activation('selu'))

    model.compile(loss='mse', optimizer=optimizer)
    return model


def create_model2(timesteps=30, neurons=50, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(1, timesteps), return_sequences=True, activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer)
    return model


def create_model1(timesteps=30, neurons=50, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(1, timesteps), return_sequences=True, activation=activation))
    # model.add(LSTM(round(neurons / 2), activation=activation, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer)
    return model


def create_model4(timesteps=30, neurons=50, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(1, timesteps), return_sequences=True, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer)
    return model


def find_best_model_v3(type_model: int, cluster):
    train, test = prepare_data_for_lstm(cluster=cluster)
    time_steps = [60, 30]

    for ts in time_steps:
        trainX, trainY, testX, testY, scaler, time_steps = prepare_train_and_test_dataset(train, test, time_steps=ts)
        param_grid = {'neurons': [50, 70, 100],
                      'optimizer': ['adam'],
                      'batch_size': [32, 64],
                      'activation': ['relu', 'linear'],
                      'timesteps': [ts]
                      }
        if type_model == 1:
            model = KerasRegressor(build_fn=create_model1, epochs=10, batch_size=1)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=ts)
            grid_result = grid.fit(trainX, trainY)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, std, param))

            model = create_model1(neurons=grid_result.best_params_['neurons'],
                                  activation=grid_result.best_params_['activation'], timesteps=ts)
        if type_model == 2:
            model = KerasRegressor(build_fn=create_model2, epochs=10, batch_size=1)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=ts)
            grid_result = grid.fit(trainX, trainY)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, std, param))

            model = create_model2(neurons=grid_result.best_params_['neurons'],
                                  activation=grid_result.best_params_['activation'], timesteps=ts)
        if type_model == 3:
            model = KerasRegressor(build_fn=create_model3, epochs=10, batch_size=1)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=ts)
            grid_result = grid.fit(trainX, trainY)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, std, param))

            model = create_model3(neurons=grid_result.best_params_['neurons'],
                                  activation=grid_result.best_params_['activation'], timesteps=ts)
        if type_model == 4:
            model = KerasRegressor(build_fn=create_model4, epochs=10, batch_size=1)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=ts)
            grid_result = grid.fit(trainX, trainY)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, std, param))

            model = create_model4(neurons=grid_result.best_params_['neurons'],
                                  activation=grid_result.best_params_['activation'], timesteps=ts)

        # save best parameters of every model
        joblib.dump(grid_result.best_params_,
                    f"lstm_new_results/cluster_{cluster}_model{type_model}_best_params.joblib")
        joblib.dump(grid_result.best_score_, f"lstm_new_results/cluster_{cluster}_model{type_model}_best_scores.joblib")


def model_type_1(ts, dr, in_act, activation):
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, ts), return_sequences=True, activation=in_act))
    model.add(LSTM(round(100 / 2), return_sequences=False, activation=activation))
    model.add(Dropout(dr))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def model_type_2(ts, dr, in_act, activation):
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, ts), return_sequences=True, activation=in_act))
    model.add(Dense(round(100 / 2), activation='relu'))
    model.add(LSTM(round(100 / 2), return_sequences=False, activation=activation))
    model.add(Dense(1, activation='relu'))
    model.add(Dropout(dr))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def model_type_3(ts, dr, in_act, activation):
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, ts), return_sequences=True, activation=in_act))
    model.add(Dense(round(100 / 2), activation='relu'))
    model.add(LSTM(round(100 / 2), return_sequences=True, activation=activation))
    model.add(Dense(round(100 / 4), activation='relu'))
    model.add(LSTM(round(100 / 4), return_sequences=False, activation=activation))
    model.add(Dense(1, activation='relu'))
    model.add(Dropout(dr))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def get_model_and_plot(cluster: int, station: str, model_type: int):
    # load data for station
    results_df = joblib.load('joblib_objects/results_df.joblib')
    data = results_df[station]

    # create model
    params = joblib.load(f'lstm_new_results_final/cl{cluster}_type1_params.joblib')
    if model_type == 1:
        model = model_type_1(ts=params['time_steps'], dr=params['dropout'], activation=params['lstm_activations'],
                             in_act=params['input_lst_layer'])
    if model_type == 2:
        model = model_type_2(ts=params['time_steps'], dr=params['dropout'], activation=params['lstm_activations'],
                             in_act=params['input_lst_layer'])

    # create train and test dataset
    train_size = int(len(data) * 0.95)
    data.index = pd.DatetimeIndex(data.index.values,
                                  freq=data.index.inferred_freq)
    data = data.drop(columns=['Date'])
    train = data.head(train_size)
    test_size = len(data) - train_size
    test = data.tail(test_size)

    train = train.values.reshape(-1, 1)
    test = test.values.reshape(-1, 1)

    # create trainx trainy testx testy
    trainX, trainY, testX, testY = prepare_train_and_test_dataset(train, test, params['time_steps'])

    # fit model
    model.fit(trainX, trainY, epochs=100, batch_size=32
              , verbose=0, validation_data=(testX, testY))
    testPredict = model.predict(testX)
    testPredict = testPredict.flatten().tolist()
    testY = testY.tolist()
    # plot results
    df = pd.DataFrame({"testPredict": testPredict, "testY": testY})

    # create a line chart with Plotly Express
    fig = px.line(df, y=["testPredict", "testY"])

    # show the chart
    pio.write_json(fig, f'json_predictions/cl{cluster}_st{station}_pred.json')


def find_best_model_final(type_model: int, cluster, df):
    train, test = prepare_data_for_lstm(cluster=cluster)
    time_steps = [15, 30]
    dropout = [0.2, 0.5]
    activations = ['relu', 'linear', 'tanh']
    input_activations = ['relu', 'linear', 'tanh']

    best_mae = float('inf')
    counter = 0
    for ts in time_steps:
        for dr in dropout:
            for activation in activations:
                for in_act in input_activations:
                    counter += 1
                    print(f"########################### {counter}/36 ########################### ")
                    trainX, trainY, testX, testY = prepare_train_and_test_dataset(train, test, time_steps=ts)
                    if type_model == 1:
                        model = model_type_1(ts, dr, in_act, activation)
                    elif type_model == 2:
                        model = model_type_2(ts, dr, in_act, activation)
                    elif type_model == 3:
                        model = model_type_3(ts, dr, in_act, activation)

                    # evaluate the model on the training dataset
                    model.fit(trainX, trainY, epochs=100, batch_size=32
                              , verbose=0, validation_data=(testX, testY))
                    testPredict = model.predict(testX)
                    train_mse_and_mae = model.evaluate(trainX, trainY)
                    train_mse, train_mae = train_mse_and_mae[0], train_mse_and_mae[1]
                    # print('MSE on Train Dataset: ', train_mse)
                    # print('MAE on Train Dataset: ', train_mae)

                    # Evaluate model on test data
                    test_mse_and_mae = model.evaluate(testX, testY)
                    test_mse, test_mae = test_mse_and_mae[0], test_mse_and_mae[1]
                    # print('MSE on Test Dataset: ', test_mse)
                    # print('MAE on Test Dataset: ', test_mae)

                    new_row = {
                        'input_lst_layer': in_act,
                        'lstm_activations': activation,
                        'time_steps': ts,
                        'dropout': dr,
                        'train_mse': train_mse,
                        'train_mae': train_mae,
                        'test_mse': test_mse,
                        'test_mae': test_mae,
                        'type_model': type_model
                    }
                    df = df.append(new_row, ignore_index=True)

                    if test_mae < best_mae:
                        best_mae = test_mae
                        if type_model == 1:
                            best_model = model_type_1(ts=ts, dr=dr, in_act=in_act, activation=activation)
                        if type_model == 2:
                            best_model = model_type_2(ts=ts, dr=dr, in_act=in_act, activation=activation)
                        if type_model == 3:
                            best_model = model_type_3(ts=ts, dr=dr, in_act=in_act, activation=activation)

    return df, best_model, best_mae


# def find_best_model(type_model, cluster):
#     train, test = prepare_data_for_lstm(cluster=cluster)
#     best_mse = 1
#     params = {}
#     best_model = Sequential()
#     time_steps = [30, 60]
#     neurons = [64, 32, 16]
#     dropout = [0.1, 0.2, 0.3]
#     epochs = [50, 100, 150]
#     batch_size = [16, 32, 64, 128, 256]
#     mse_mae_results = pd.DataFrame(
#         columns=['epochs', 'batch', 'time_steps', 'dropout', 'neurons', 'test_mse', 'test_mae', 'train_mse',
#                  'train_mae'])
#     for ts in time_steps:
#         trainX, trainY, testX, testY, scaler, time_steps = prepare_train_and_test_dataset(train, test, time_steps=ts)
#         for n in neurons:
#             for dr in dropout:
#                 for bs in batch_size:
#                     for e in epochs:
#                         print(f"Trying {ts} timesteps, {n} neurons, {dr} dropout, {bs} batch size, {e} epochs")
#                         if type_model == 1:
#                             model = Sequential()
#                             model.add(LSTM(n, input_shape=(1, time_steps), return_sequences=True, activation='linear'))
#                             model.add(LSTM(round(n / 2), return_sequences=False, activation='linear'))
#                             model.add(Dropout(dr))
#                             model.add(Dense(1, activation='relu'))
#                             model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#                         elif type_model == 2:
#                             model = Sequential()
#                             model.add(LSTM(n, input_shape=(1, time_steps), return_sequences=True, activation='linear'))
#                             model.add(Dense(round(n / 2), activation='linear'))
#                             model.add(LSTM(round(n / 2), return_sequences=True, activation='linear'))
#                             model.add(Dense(round(n / 4), activation='linear'))
#                             model.add(LSTM(round(n / 4), return_sequences=False, activation='relu'))
#                             model.add(Dense(1, activation='linear'))
#                             model.add(Dropout(dr))
#                             model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#                         elif type_model == 3:
#                             model = Sequential()
#                             model.add(LSTM(n, input_shape=(1, time_steps), return_sequences=True, activation='linear'))
#                             model.add(Dense(round(n / 2), activation='linear'))
#                             model.add(LSTM(round(n / 2), return_sequences=True, activation='linear'))
#                             model.add(Dense(round(n / 4), activation='linear'))
#                             model.add(LSTM(round(n / 4), return_sequences=False, activation='relu'))
#                             model.add(Dense(1, activation='relu'))
#                             model.add(Dropout(dr))
#                             model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#                         # evaluate the model on the training dataset
#                         model.fit(trainX, trainY, epochs=e, batch_size=bs
#                                   , verbose=0, validation_data=(testX, testY))
#                         testPredict = model.predict(testX)
#                         testPredict = scaler.inverse_transform(X=testPredict)
#                         test_rmse = np.sqrt(np.mean((testPredict - testY) ** 2))
#                         train_mse_and_mae = model.evaluate(trainX, trainY)
#                         train_mse, train_mae = train_mse_and_mae[0], train_mse_and_mae[1]
#
#                         # Evaluate model on test data
#                         test_mse_and_mae = model.evaluate(testX, testY)
#                         test_mse, test_mae = test_mse_and_mae[0], test_mse_and_mae[1]
#                         print('MSE on Test Dataset: ', test_mse)
#                         print('MAE on Test Dataset: ', test_mae)
#                         new_row = {
#                             'epochs': e,
#                             'batch': bs,
#                             'time_steps': ts,
#                             'dropout': dr,
#                             'neurons': n,
#                             'test_mse': test_mse,
#                             'test_mae': test_mae,
#                             'train_mse': train_mse,
#                             'train_mae': train_mae
#                         }
#                         mse_mae_results.loc[len(mse_mae_results)] = new_row
#                         if test_mse < best_mse:
#                             best_mse = test_mse
#                             best_model = model
#                             params = {
#                                 'epochs': e,
#                                 'batch': bs,
#                                 'time_steps': ts,
#                                 'dropout': dr,
#                                 'neurons': n,
#                                 'test_mse': test_mse,
#                                 'test_mae': test_mae,
#                                 'train_mse': train_mse,
#                                 'train_mae': train_mae
#                             }
#     mse_mae_results.to_csv(f'lstm_results/metrics_results_{cluster}_type_{type_model}.csv')
#     print('saved results to csv')
#     print(f'FOUND BEST MODEL FOR CLUSTER {cluster} and type {type_model}:')
#     print(f'{best_model.summary()} ')
#     dict_string = str(params)
#     print(params)
#     # Open a file for writing
#     filename_params = f'lstm_results/params_cl{cluster}_mod{type_model}.txt'
#     with open(filename_params, 'w') as file:
#         # Write the dictionary string to the file
#         file.write(dict_string)
#     joblib.dump(best_model, f'lstm_results/model_for_cl{cluster}_type_{type_model}.joblib')


def create_table(cluster: int):
    models = [1, 3, 4]
    table_df = pd.DataFrame(
        columns=["model_type", "neurons", "optimizer", "batch_size", "activation", "timesteps", "mse"])
    for model in models:
        params = joblib.load(f'lstm_new_results/cluster_{cluster}_model{model}_best_params.joblib')
        scores = joblib.load(f'lstm_new_results/cluster_{cluster}_model{model}_best_scores.joblib')
        if model == 3:
            model = 2
        elif model == 4:
            model = 3
        new_row = {"model_type": model, "neurons": params["neurons"], "optimizer": params["optimizer"],
                   "batch_size": params["batch_size"],
                   "activation": params["activation"], "timesteps": params["timesteps"], "mse": scores}
        table_df = table_df.append(new_row, ignore_index=True)
    table_df.to_csv(f"lstm_new_results/cluster_{cluster}_params_scores.csv")


def read_best_score_for_cluster(cluster: int):
    models = [1, 3, 4]
    best_score = float('inf')
    for model in models:
        models_score = joblib.load(f'lstm_new_results/cluster_{cluster}_model{model}_best_scores.joblib')
        if models_score < best_score:
            best_score = models_score
            best_model = joblib.load(f'lstm_new_results/cluster_{cluster}_model{model}_best_params.joblib')
    return best_model, best_score


def load_wining_models(cluster: int, type_model: int):
    dataframe = pd.read_csv(f'lstm_new_results_final/cluster{cluster}_results.csv')
    dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe[dataframe['type_model'] == int(type_model)]
    min_row = dataframe.loc[dataframe['test_mae'].idxmin()]

    best_rows = dataframe.nsmallest(5, ['test_mse', 'test_mae'])
    best_row = dataframe.nsmallest(1, ['test_mse', 'test_mae'])
    print(f'Best row for cluster {cluster} and type {type} is: test mse of {best_row["test_mse"]} and test mae of {best_row["test_mae"]}')
    # joblib.dump(best_rows, f'lstm_new_results_final/cl{cluster}_type{type_model}_params.joblib')
    # best_rows.to_csv(f'lstm_new_results_final/cl{cluster}_type{type_model}_params.csv')
    # print(f'Best param combination for cl {cluster} and type {type_model}: {min_row}')
    # joblib.dump(min_row, f'lstm_new_results_final/cl{cluster}_type{type_model}_params.joblib')
    # if type_model == 1:
    #     print(min_row['time_steps'])
    #     model = model_type_1(ts=min_row['time_steps'], dr=min_row['dropout'], activation=min_row['lstm_activations'],
    #                          in_act=min_row['input_lst_layer'])
    # if type_model == 2:
    #     model = model_type_2(ts=min_row['time_steps'], dr=min_row['dropout'], activation=min_row['lstm_activations'],
    #                          in_act=min_row['input_lst_layer'])
    # joblib.dump(model, f'lstm_new_results_final/cluster{cluster}_type{type_model}_best_model.joblib')
    # plot_model(model, to_file=f'lstm_network_cl{cluster}_type{type_model}.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':

    # get_best_model_for_cluster_1('varibobi')

    clusters = [1, 2, 3, 4, 5]
    type_models = [1, 2]

    for cluster in clusters:
        for type_model in type_models:
            load_wining_models(cluster, type_model)

    # for cluster in clusters:
    #     key = f'cluster{cluster}'
    #     stations = joblib.load("joblib_objects/cluster_stations.joblib")[key]
    #     for station in stations:
    #         get_model_and_plot(cluster=cluster, station=station, model_type=1)

    #     best_model, best_score = read_best_score_for_cluster(cluster)

    # create_table(cluster)

    # for cluster in clusters:
    #     df = pd.DataFrame(
    #         columns=['input_lst_layer', 'lstm_activations', 'time_steps', 'dropout', 'train_mse', 'train_mae',
    #                  'test_mse',
    #                  'test_mae', 'type_model'])
    #     best_mae = float('inf')
    #     for type_model in type_models:
    #         print(f'################## FIND BEST MODEL OF TYPE {type_model} AND CLUSTER {cluster} ##################')
    #         df, best_model_type, best_type_mae = find_best_model_final(type_model=type_model, cluster=cluster, df=df)
    #         if best_type_mae < best_mae:
    #             best_mae = best_type_mae
    #             best_model = best_model_type
    #     print(f'cluster {cluster} printed results')
    #     df.to_csv(f'lstm_new_results_final/cluster{cluster}_results.csv')
    #     joblib.dump(best_model, f'lstm_new_results_final/cluster{cluster}_best_model.joblib')
