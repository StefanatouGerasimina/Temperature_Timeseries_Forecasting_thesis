import ast

import joblib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def get_results(cluster: int, type: int):
    filename = f'params_cl{cluster}_mod{type}_fixed.txt'
    with open(filename, "r") as f:
        file_contents = f.read()
        parameters = ast.literal_eval(file_contents)
    return parameters

def save_plots(cluster:int, type:int):
    dict_parameters = get_results(cluster=cluster, type=type)
    dict_parameters['testPredict'] = [item[0] for item in dict_parameters['testPredict']]
    # TODO: Before plotting i have to denormalize my test true data
    dict_parameters['testY'] = [[num] for num in dict_parameters['testY']]
    scaler_filename = f'scaler_cl{cluster}_type_{type}.joblib'
    scaler = joblib.load(scaler_filename)
    testY = scaler.inverse_transform(dict_parameters['testY']).ravel().tolist()
    test_predicted = dict_parameters['testPredict']

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the true values in blue
    ax.plot(testY, color='blue', label='True Values')

    # Plot the predicted values in red
    ax.plot(test_predicted, color='red', label='Predicted Values')

    # Set the axis labels and legend
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.legend()

    # Show the plot
    plt.title(f'Cluster {cluster} with model type {type}')
    plt.show()
    plt.savefig(f'cl_{cluster}_modeltype{type}')


if __name__ == '__main__':
    clusters = [1, 2, 3]
    model_types = [1, 2, 3, 4]
    for cluster in clusters:
        for type in model_types:
            save_plots(cluster, type)