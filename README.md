# Temperature_Timeseries_Forecasting_thesis
This project describes my Thesis at the end of my studies at the Department of Digital Systems from the University of Piraeus. 

**You can find my thesis [[here](https://drive.google.com/file/d/1Tnk10NN9U4IVu2aws_tsfsCEu3rigMHj/view?usp=sharing)] where all the procedures, the analysis, the model evaluation etc
are mentioned.**


#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is to analyze and forecast air Temperature, in the most "hot" places in Greece. Due to the climate change, but 
also Greece marking high warm days, uneceptional fires are a common phenomenon resulting into massive forest damages.

### Methods Used
* Timeseries Analytics with different plots and frequences.
* Self Organised Maps for timeseries classification.
* KMeans
* Timeseries forecasting with LSTM networks
* Error analysis with MSE and MAE.

### Technologies
* Python
* Pandas, numpy
* Tensorflow, sklearn
* Keras
* streamlit
* etc 

## Project Description
This project focuses on the application of time series analysisin the context of climate change data, analyzing daily temperature timeseries from 11 Weather Stations. The study uses
classification techniques such as Kmeans and Self organized Maps, for grouping the stations based on the standards they form. The effectiveness of the two Tests categorization, evaluated using silhouette score and quantification error,
providing information on the homogeneity and discretization of clusters. At the same time, this thesis explores the possibilities of predicting two LSTM models (Long Short Term Memory) with different architecture, to predict the
temperature of the next day. These results are evaluated using both MSE (Mean Squared Error) and MAE (mean Absolute Error) to assess the accuracy and reliability of their prediction. Combining
clustering analysis and forecasting techniques, this study contributes to full understanding of climate change patterns and helps to get documented decisions on climate-related mitigation and adaptation strategies; and
the phenomena arising from it concerning temperature.

## Needs of this project

- Data processing
- Data analysis
- Models training

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/tree/main/daily_temperature)] within this repo.
3. Data processing/transformation scripts are being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/blob/main/analysis.py)], [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/blob/main/preprocesing.py)],[[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/blob/main/normalization.py)].
4. Data clustering scripts are being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/blob/main/clustering.py)] and the results are being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/tree/main/clustering_evaluation)].
5. Data Forecasting scripts are being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/blob/main/forecasting.py)] and the results are being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/tree/main/lstm_new_results)], [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/tree/main/lstm_new_results_final)], [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/tree/main/plots)]
6. Models Evaluation scripts are being kept [[here](https://github.com/StefanatouGerasimina/Temperature_Timeseries_Forecasting_thesis/blob/main/models_evaluation.py)].


## Instructions
Clone the repo as your local project and after installing all the neccesairy packages, run the following command: streamlit run app.py



