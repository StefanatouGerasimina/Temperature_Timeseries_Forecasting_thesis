import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px


class Preprocessing:
    def __init__(self, path):
        self.path = path

    def get_data(self):
        print("Inside of the data preperation")
        results_df = {}
        mySeries, namesofMySeries = [], []
        for filename in os.listdir(self.path):
            if filename.endswith(".csv"):
                file_path = self.path + filename
                station = filename.split('.')[0]
                df = pd.read_csv(file_path)
                df = df.rename(columns={'time': 'Date'})
                df = df.rename(columns={'temperature_2m_mean (°C)': 'Temperature'})
                # convert temperatures
                df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%d/%m/%Y'))
                df.index = pd.to_datetime(df['Date'], format='%d/%m/%Y')
                df.asfreq('D')
                mySeries.append(df)
                namesofMySeries.append(station)
                results_df[station] = df
        # save files into joblib objects
        joblib.dump(results_df, 'joblib_objects/results_df.joblib')
        print("Just created object dataframe")
        joblib.dump(mySeries, 'joblib_objects/mySeries.joblib')
        print("Just created object series")
        joblib.dump(namesofMySeries, 'joblib_objects/namesofMySeries.joblib')
        print("Just created object list for the names of my series")

        return mySeries, namesofMySeries, results_df

    def get_weakly_data(self, results_df):
        weakly_df = {}
        weaklySeries = []
        print("Starting creating  weakly data...")
        for key, value in results_df.items():
            new_value = value.groupby(pd.Grouper(freq='W')).describe()
            new_value['Date'] = new_value.index
            new_value.columns = new_value.columns.droplevel()
            new_value.drop(['count', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1, inplace=True)
            # new_value.rename(columns={'mean':'Temperature'},inplace=True, errors='raise')
            weakly_df[key] = new_value
            weakly_df[key].rename(columns={'mean': 'Temperature', '': 'Date'}, inplace=True)
            weaklySeries.append(new_value)
            print(weakly_df[key].columns)
        # save files into joblib objects
        joblib.dump(weakly_df, 'joblib_objects/results_df_weekly.joblib')
        print("Just created joblib object for weakly data.")
        joblib.dump(weaklySeries, 'joblib_objects/weaklySeries.joblib')
        print("Just created series for weakly data.")

        return weakly_df, weaklySeries

    def get_monthly_data(self, results_df):
        monthly_df, yearly_df = {}, {}
        monthlySeries, yearlySeries = [], []
        print("Starting creating  monthly data...")
        for key, value in results_df.items():
            new_value = value.groupby(pd.Grouper(freq='m')).describe()
            new_value['Date'] = new_value.index
            new_value.columns = new_value.columns.droplevel()
            new_value.drop(['count', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1, inplace=True)
            # new_value.rename(columns={'mean':'Temperature'},inplace=True, errors='raise')
            monthly_df[key] = new_value
            monthly_df[key].rename(columns={'mean': 'Temperature', "": "Date"}, inplace=True)
            monthlySeries.append(new_value)
            print(monthly_df[key].columns)

        # save files into joblib objects
        joblib.dump(monthly_df, 'joblib_objects/results_df_monthly.joblib')
        print("Just created joblib object for monthly data.")
        joblib.dump(monthlySeries, 'joblib_objects/monthlySeries.joblib')
        print("Just created series for monthly data.")

        return monthly_df, monthlySeries

    def get_yearly_data(self, results_df):
        # yearly
        yearly_df = {}
        yearlySeries = []
        print("Starting to create objects for yearly data.")
        for key, value in results_df.items():
            new_value = value.groupby(pd.Grouper(freq='y')).describe()
            new_value['Date'] = new_value.index
            new_value.columns = new_value.columns.droplevel()
            new_value.drop(['count', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1, inplace=True)
            yearly_df[key] = new_value
            yearly_df[key].rename(columns={'mean': 'Temperature', '': 'Date'}, inplace=True)
            yearlySeries.append(new_value)

        # save files into joblib objects
        joblib.dump(yearly_df, 'joblib_objects/results_df_yearly.joblib')
        print('Just created yearly df.')
        joblib.dump(yearlySeries, 'joblib_objects/yearlySeries.joblib')
        print("Just created series for yearly data.")

        return yearly_df, yearlySeries


def plot_time_series(type):
    """
    Plots a time series using matplotlib.

    Args:
        time_series (list or numpy array): The time series data to be plotted.
    """
    if type == 'daily':
        data = joblib.load("joblib_objects/results_df.joblib")
    elif type == 'monthly':
        data = joblib.load("joblib_objects/results_df_monthly.joblib")
    elif type == 'weakly':
        data = joblib.load("joblib_objects/results_df_weekly.joblib")
    elif type == 'yearly':
        data = joblib.load("joblib_objects/results_df_yearly.joblib")
    else:
        data = joblib.load("joblib_objects/results_df.joblib")
    # merge the dataframes into one dataframe
    data['Date'] = data['arkadia']['Date']
    df = pd.concat(data, names=['Station']).reset_index(level=1, drop=True)

    # plot the lines using px.line with dictionary keys as legend labels
    fig = px.line(df, x='Date', y='Temperature', color=df.index, line_group=df.index.get_level_values(0))
    fig.show()
    save_path = os.path.join(os.getcwd(), f'plots/{type}_data.json')
    fig.write_json(save_path)



#
if __name__ == '__main__':
    prep = Preprocessing(path='daily_temperature/')
    mySeries, namesofMySeries, results_df = prep.get_data()
#     weakly_df, weaklySeries = prep.get_weakly_data(results_df=results_df)
    monthly_df, monthlySeries = prep.get_monthly_data(results_df=results_df)
#     yearly_df, yearlySeries = prep.get_yearly_data(results_df=results_df)
#     matplotlib.use('TkAgg')  # or 'Qt5Agg'
    types = ['daily', 'weakly', 'monthly', 'yearly']
    for type in types:
        plot_time_series(type)
