import glob
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import pacf, acf
import plotly.io as pio
from statsmodels.graphics.tsaplots import plot_acf



class Analysis:
    def __init__(self, type):
        self.type = type

    def get_lag_plots(self):
        if self.type == 'daily':
            data = joblib.load("joblib_objects/results_df.joblib")
        elif self.type == 'monthly':
            data = joblib.load("joblib_objects/results_df_monthly.joblib")
        elif self.type == 'weakly':
            data = joblib.load("joblib_objects/results_df_weekly.joblib")
        elif self.type == 'yearly':
            data = joblib.load("joblib_objects/results_df_yearly.joblib")
        else:
            data = joblib.load("joblib_objects/results_df.joblib")

        # create lag plot
        for key, value in data.items():
            value['shifted'] = value['Temperature'].shift(1)
            fig = px.scatter(value, x='shifted', range_x=[-5, 40], range_y=[-5, 40], y='Temperature',
                             color='Temperature', range_color=[-5, 40])
            fig.update_layout(title='Lag Plot', xaxis_title='Original Values', yaxis_title='Lagged Values')
            # fig.show()
            # save_path = os.path.join(os.getcwd(), f'plots/{self.type}_{key}_lag.json')
            # fig.write_json(save_path)

            # self.plot_autocorrelation(data = value['Temperature'])

            # plot autocorellation based on time
            plt.figure(figsize=(8, 6))
            plot_acf(value['Temperature'], lags=30)  # Adjust the number of lags as needed
            plt.title(f'Autocorrelation Plot for station {key}')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.show()

    def plot_autocorrelation(self, data):
        fig = px.line(data, x=data.index, y=data.values)
        fig.update_layout(
            title='Autocorrelation Plot',
            xaxis_title='Lag',
            yaxis_title='Autocorrelation'
        )
        fig.show()

    def histograms(self):
        """
        This function is used in order to plot histograms with the Temperatures rages for every station. This way
        we can see which stations/cities mark the highest temperatures more frequently.
        """
        daily_data = joblib.load("joblib_objects/results_df.joblib")
        new_daily_data = {}
        unique_years = list(daily_data['arkadia'].index.year.unique()[:-1])
        for year in unique_years:
            for key, value in daily_data.items():
                mask = value.index.year == year
                rows = value.loc[mask]
                histogram_masks = {
                    "-5-0": len(
                        rows.loc[(rows['Temperature'] >= -5) & (rows['Temperature'] < 0)]['Temperature'].tolist()),
                    "0-5": len(
                        rows.loc[(rows['Temperature'] >= 0) & (rows['Temperature'] < 5)]['Temperature'].tolist()),
                    "5-10": len(
                        rows.loc[(rows['Temperature'] >= 5) & (rows['Temperature'] < 10)]['Temperature'].tolist()),
                    "10-15": len(
                        rows.loc[(rows['Temperature'] >= 10) & (rows['Temperature'] < 15)]['Temperature'].tolist()),
                    "15-20": len(
                        rows.loc[(rows['Temperature'] >= 15) & (rows['Temperature'] < 20)]['Temperature'].tolist()),
                    "20-25": len(
                        rows.loc[(rows['Temperature'] >= 20) & (rows['Temperature'] < 25)]['Temperature'].tolist()),
                    "25-30": len(
                        rows.loc[(rows['Temperature'] >= 25) & (rows['Temperature'] < 30)]['Temperature'].tolist()),
                    "30-35": len(
                        rows.loc[(rows['Temperature'] >= 30) & (rows['Temperature'] < 35)]['Temperature'].tolist()),
                    "35-40": len(
                        rows.loc[(rows['Temperature'] >= 35) & (rows['Temperature'] < 40)]['Temperature'].tolist()),
                    "40-45": len(
                        rows.loc[(rows['Temperature'] >= 40) & (rows['Temperature'] < 45)]['Temperature'].tolist())
                }
                new_daily_data[key] = histogram_masks
            categories = list(new_daily_data.keys())
            labels = list(new_daily_data[categories[0]].keys())

            # Create a list of traces
            traces = []
            for category in categories:
                values = [new_daily_data[category][label] for label in labels]
                trace = go.Bar(x=labels, y=values, name=category)
                traces.append(trace)

            # Create the layout
            layout = go.Layout(title=f'Year {year}: Distribution of Temperatures',
                               xaxis=dict(title='Temperature Range'), yaxis=dict(title='Sum of Temperature values'))

            # Create the figure
            fig = go.Figure(data=traces, layout=layout)

            # Show the figure
            fig.show()
            save_path = os.path.join(os.getcwd(), f'plots/{self.type}_{year}_temperatures_distribution.json')
            fig.write_json(save_path)

    def year_for_station_bar_plot(self, station):
        daily_data = joblib.load("joblib_objects/results_df.joblib")
        unique_years = list(daily_data[station].index.year.unique()[:-1])
        new_daily_data = {}
        for year in unique_years:
            mask = daily_data[station].index.year == year
            rows = daily_data[station].loc[mask]
            histogram_masks = {
                "-5-0": len(rows.loc[(rows['Temperature'] >= -5) & (rows['Temperature'] < 0)]['Temperature'].tolist()),
                "0-5": len(rows.loc[(rows['Temperature'] >= 0) & (rows['Temperature'] < 5)]['Temperature'].tolist()),
                "5-10": len(rows.loc[(rows['Temperature'] >= 5) & (rows['Temperature'] < 10)]['Temperature'].tolist()),
                "10-15": len(
                    rows.loc[(rows['Temperature'] >= 10) & (rows['Temperature'] < 15)]['Temperature'].tolist()),
                "15-20": len(
                    rows.loc[(rows['Temperature'] >= 15) & (rows['Temperature'] < 20)]['Temperature'].tolist()),
                "20-25": len(
                    rows.loc[(rows['Temperature'] >= 20) & (rows['Temperature'] < 25)]['Temperature'].tolist()),
                "25-30": len(
                    rows.loc[(rows['Temperature'] >= 25) & (rows['Temperature'] < 30)]['Temperature'].tolist()),
                "30-35": len(
                    rows.loc[(rows['Temperature'] >= 30) & (rows['Temperature'] < 35)]['Temperature'].tolist()),
                "35-40": len(
                    rows.loc[(rows['Temperature'] >= 35) & (rows['Temperature'] < 40)]['Temperature'].tolist()),
                "40-45": len(rows.loc[(rows['Temperature'] >= 40) & (rows['Temperature'] < 45)]['Temperature'].tolist())
            }
            new_daily_data[year] = histogram_masks
        print(new_daily_data)
        categories = list(new_daily_data.keys())
        labels = list(new_daily_data[categories[0]].keys())

        # Create a list of traces
        traces = []
        for category in categories:
            values = [new_daily_data[category][label] for label in labels]
            trace = go.Bar(x=labels, y=values, name=category)
            traces.append(trace)

        # Create the layout
        layout = go.Layout(title=f'Station {station}: Distribution of Temperatures',
                           xaxis=dict(title='Temperature Range'),
                           yaxis=dict(title='Sum of Temperature values'))

        # Create the figure
        fig = go.Figure(data=traces, layout=layout)

        # Show the figure
        fig.show()
        save_path = os.path.join(os.getcwd(), f'plots/{self.type}_{station}_temperatures_distribution.json')
        fig.write_json(save_path)

    def box_plot(self):
        daily_data = joblib.load("joblib_objects/results_df.joblib")
        namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
        for station in namesofMySeries:
            a = len(daily_data[station])
            # sns.boxplot(x=daily_data[station].index.year, y=daily_data[station]['Temperature'])
            fig = px.box(x=daily_data[station].index.year, y=daily_data[station]['Temperature'],
                         title=f'Boxplot for station {station}', points='outliers')
            fig.update_layout(xaxis_title='Years', yaxis_title='Daily Temperatures')
            fig.update_traces(boxpoints='all')

            fig.show()
            save_path = os.path.join(os.getcwd(), f'plots/{self.type}_{station}_boxplot.json')
            fig.write_json(save_path)

            # find out if data have normal distribution
            # Perform Shapiro-Wilk test
            stat, p = shapiro(daily_data[station]['Temperature'])
            # Interpret the results
            alpha = 0.05
            print(station)
            if p > alpha:
                print('Data is normally distributed ')
            else:
                print('Data is not normally distributed')

    def seasonal_decomposition(self):
        daily_data = joblib.load("joblib_objects/results_df.joblib")
        namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
        stationary = {}
        for station in namesofMySeries:
            result = adfuller(daily_data[station]['Temperature'])
            test_statistic = result[0]
            p_value = result[1]
            if p_value < 0.05:
                stationary[station] = 'Stationary'
            else:
                stationary[station] = 'Non Stationary'
        for station in namesofMySeries:
            # perform seasonal decomposition
            decomposition = sm.tsa.seasonal_decompose(daily_data[station]['Temperature'], model='additive', period=365)

            # create seasonal decomposition plot
            fig = make_subplots(rows=4, cols=1)
            # add traces for original time series and each component
            fig.add_trace(
                go.Scatter(x=daily_data[station].index, y=daily_data[station]['Temperature'], name='Original'), row=1,
                col=1)
            fig.add_trace(go.Scatter(x=daily_data[station].index, y=decomposition.trend, name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=daily_data[station].index, y=decomposition.seasonal, name='Seasonal'), row=3,
                          col=1)
            fig.add_trace(go.Scatter(x=daily_data[station].index, y=decomposition.resid, name='Residual'), row=4, col=1)

            # update layout of figure
            fig.update_layout(title=f'Seasonal Decomposition of {station} Daily Temperatures', height=800,
                              showlegend=False)

            # show figure
            fig.show()
            save_path = os.path.join(os.getcwd(), f'plots/{self.type}_{station}_seasonal_decomposition.json')
            fig.write_json(save_path)
            # fig = decomposition.plot()
            # fig.show()

    def stationarity_checking(self, adf_df):
        if len(self.type) > 0:
            joblib_filename = f'joblib_objects/results_df_{self.type}.joblib'
        else:
            joblib_filename = 'joblib_objects/results_df.joblib'
        data = joblib.load(joblib_filename)
        for station, df in data.items():
            result = adfuller(df['Temperature'])
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t{}: {}'.format(key, value))
            acf_values = acf_values = acf(df['Temperature'], fft=False)
            acf_value = np.abs(acf_values)
            if np.all(np.abs(acf_values) < 1.96 / np.sqrt(len(df))):
                strong_stationarity = True
            else:
                strong_stationarity = False
            if result[1] < 0.05:
                print(f'Timeserie for station {station} is stationary')
            else:
                print(f'Timeserie for station {station} is NOT stationary.')

            adf_df = adf_df.append({'Station': station, 'P-value': result[1], 'ACF-value': acf_value, 'Strong_stationarity': strong_stationarity}, ignore_index=True)
        adf_df.to_csv(f'descr_stats/adf_p_value.csv')
    def descriptive_statistics(self):
        if len(self.type) > 0 and self.type != 'daily':
            joblib_filename = f'joblib_objects/results_df_{self.type}.joblib'
        else:
            joblib_filename = 'joblib_objects/results_df.joblib'
        data = joblib.load(joblib_filename)
        for station, df in data.items():
            descriptive_stats = {}
            mean_temp = df['Temperature'].mean()
            median_temp = df['Temperature'].median()
            range_temp = df['Temperature'].max() - df['Temperature'].min()
            std_dev_temp = df['Temperature'].std()
            var_temp = df['Temperature'].var()
            q1 = df['Temperature'].quantile(0.25)
            q3 = df['Temperature'].quantile(0.75)
            iqr_temp = q3 - q1
            skew_temp = df['Temperature'].skew()
            kurt_temp = df['Temperature'].kurtosis()
            acf_temp = df['Temperature'].autocorr()
            pacf_temp = pacf(df['Temperature'])
            descriptive_stats[station] = {
                'mean': mean_temp,
                'median': median_temp,
                'range': range_temp,
                'std_dev': std_dev_temp,
                'var': var_temp,
                'iqr': iqr_temp,
                'skew': skew_temp,
                'kurt': kurt_temp,
                'acf': acf_temp,
                'pacf': pacf_temp
            }

            # monthly statistics
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            df['month'] = df['Date'].dt.month
            df['year'] = df['Date'].dt.year

            # create line plot for monthly mean temperature
            monthly_mean_temp = df.groupby(['year', 'month'])['Temperature'].mean()
            dates = monthly_mean_temp.index
            dates_strings = [f"{year}-{month}" for year, month in dates]
            # df_plot = pd.DataFrame({'Date': dates_strings, 'Temperature': monthly_mean_temp.values})
            # fig = px.line(df_plot, x='Date', y='Temperature')
            # fig.add_scatter(x=df_plot['Date'], y=df_plot['Temperature'], mode='markers', name='Data Points')
            # pio.write_image(fig, f'descr_stats/descr_{station}_{self.type}_line_plot.png')

            # create box plot for monthly min & max temperature
            monthly_min_temp = df.groupby(['year', 'month'])['Temperature'].min()
            monthly_max_temp = df.groupby(['year', 'month'])['Temperature'].max()
            monthly_min_max_df = pd.DataFrame(
                {'Date': dates_strings, 'Min_Temperature': monthly_min_temp, 'Max_Temperature': monthly_max_temp})
            monthly_min_max_df.to_csv(f'descr_stats/monthly_min_max_{station}_{self.type}.csv', index=False)
            top_min_temps = monthly_min_max_df.nsmallest(5, 'Min_Temperature')['Min_Temperature']
            top_max_temps = monthly_min_max_df.nlargest(5, 'Max_Temperature')['Max_Temperature']
            minmaxvalues = top_min_temps.values.tolist() + top_max_temps.values.tolist()
            min_index, max_index = top_min_temps.index.tolist(), top_max_temps.index.tolist()
            date_index = min_index + max_index
            dates_strings = [f"{year}-{month}" for year, month in date_index]
            minmaxdf = pd.DataFrame({'Date': dates_strings, 'MinMaxTemps': minmaxvalues})
            minmaxdf.to_csv(f'descr_stats/top_monthly_min_max_{station}.csv', index=False)
            fig = px.box(df, x='month', y='Temperature', color='year')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig.update_layout(xaxis={'tickvals': list(range(1, 13)), 'ticktext': month_names})
            fig.update_layout(title=f'{station}')
            fig.show()
            pio.write_image(fig, f'descr_stats/descr_{station}_{self.type}_box_plot.png')

            # std csv
            # std_dev = df.groupby(['month','year'])['Temperature'].std()
            # sorted_std_df = std_dev.sort_values(ascending=False).to_frame()
            # sorted_std_df.to_csv(f'descr_stats/std_{station}_{self.type}.csv')



if __name__ == '__main__':
    types = ['daily']
    an = Analysis(type='daily')
    an.get_lag_plots()
    an.descriptive_statistics()
    # adf_results = pd.DataFrame(columns=['Station','P-value', 'ACF-value', 'Strong_stationarity'])
    # an.stationarity_checking(adf_results)
    # for type in types:
    #     an = Analysis(type=type)
    #     an.stationarity_checking()
    #     break
