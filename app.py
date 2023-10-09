import os

import joblib
import pandas as pd
import plotly.io as pio
import streamlit as st
from PIL import Image

st.set_page_config(page_title="My Streamlit App", page_icon="\U0001F393", layout="wide",
                   initial_sidebar_state="expanded")

# Create a sidebar navigation bar with links to different pages
nav = st.sidebar.radio("Navigate through the different processes of this Timeseries Analytics project",
                       ["Home", "Data Sources", "Data Preview & Analytics",
                        "Data Clustering", "Data Forecasting", "Contact"])

# Render content based on which link is selected
if nav == "Home":
    st.header(" \U0001F321 Welcome to my Timeseries Analytics project!")

    st.subheader("Description")
    st.markdown(" This app is designed to provide you with an interactive experience for exploring and "
                "visualizing my selected data in new interesting ways. "
                "Whether you're a data scientist, a meteorologist, a researcher, or just someone who's interested in analyzing data, "
                "this research has something for you. With a range of **powerful visualization tools, and modern methods of analytics**, "
                "you can easily explore our selected data, get an idea for **timeseries forecasting methods**, and help your country's stations predict their next day Temperture."
                "<u>Our goal is to find better ways for temperature forecasting, espesially on those areas that have marked the highest temperature in the summer days.</u>"
                "We hope you find my researsh interesting and we look forward seeing the amazing insights you'll uncover!",
                unsafe_allow_html=True)

    st.subheader('Instructions')
    st.write(
        "As you can see on the left of this UI there is a navigating sidebar, containing all the modules developed during the research, and"
        "by clicking on each section, you can see its content in more detail. ")
    st.write("Their contents are analyzed as follows:")
    st.markdown("""
    - **Home**: Contains welcome message and navigation instructions.
    - **Data**: Description of the dataset, my project sources and the goal of the project.
    - **Preprocessing & Normalization**: Contains the methods used in order to preprocess our data and the Normalization techniques.
    - **Analysis**: Different kind of plots to get a better understanding of data.
    - **Data Clustering**: Trying two different timeseries Clustering methods and evaluate them with Silhouette score and  quantization error.
    - **Data Forecasting**: Creating and evaluating different LSTM models for the cluster calculated in the previous step. 
    - **Contact**: Contact Information.
    """)

elif nav == "Data Sources":
    st.header("Dataset Information")
    st.subheader("Data Source")

    # Create a 2-column layout
    col1, col2 = st.columns([1, 4])

    # Add image to the first column
    with col1:
        image_path = os.path.join(os.getcwd(), 'images/open-meteo (1).jpg')
        image = Image.open(image_path)
        st.image(image, caption='Open-Meteo')

    # Add paragraph to the second column
    with col2:
        st.write("Open Meteo is a free and open-source weather data platform that provides users with access "
                 "to real-time weather data, historical weather data, and forecast models. The platform was "
                 "developed by a team of meteorologists and software developers to provide a reliable source"
                 " of weather data for developers, researchers, and anyone interested in weather data analysis. "
                 "OpenMeteo offers a variety of APIs and tools that make it easy for users to access and analyze "
                 "weather data, bus also offers the choice of an online tool. The platform is designed to be easy"
                 " to use and flexible, allowing users to customize their data queries and visualizations to meet"
                 " their specific needs.")
    st.write("In this case, I used the online option, where as input were used the cities with most frequenced fires, and by selecting  "
             "the time period between 01-01-2013 and 01-01-2023 we got historical data on the daily mean temperature"
             " in Celsius for 11 stations.")
    st.write(
        "Τhe selection of these 11 station was based on the following two criteria. **Which regions in Greece have the "
        "largest and most frequent firestorms due to high temperatures?**")
    st.write("In this way, I try to investigate these areas, find out if there is any pattern or any correlation"
             " between them, or even predict future fires based on their daily temperature, especially for the warmer"
             " days.")
    st.write("As soon as I set as input the criteria mentioned above, there is an option in order to download the "
             "historical data in a csv format. In addition to the historical data, this file contains information "
             "on the coordinates and metrics of the data.")

    st.subheader("Stations")
    st.write(
        "Due to the geographical position of our country, Greece experiences very high temperatures in some cities during"
        " the summer months, with the result that those cities suffer quite often from fires."
        "Top 10 cities based on the criteria mentioned above are: ")
    st.markdown(""" 
        1. **Varibobi** (latitude: 38.199997, longitude:23.699997)
        2. **Trikala** (latitude: 39.600006, longitude:21.800003)
        3. **Rethymno** (latitude: 35.4, longitude:24.600006)
        4. **Messinia** (latitude: 37.1, longitude:22)
        5. **Mati** (latitude: 37.800003, longitude:22.300003)
        6. **Larissa** (latitude: 39.600006, longitude:22.40001)
        7. **Kineta** (latitude: 37.4, longitude:23.2)
        8. **Keratea** (latitude: 37.800003, longitude:24)
        9. **Halkidiki** (latitude: 40.40001, longitude:23.300003)
        10. **Corinthos** (latitude: 38, longitude:23)
        11. **Arkadia** (latitude: 37.6, longtitude:22.90001)""")

    st.markdown("#### You can find more information about fires in the aforementioned areas here:")
    st.write(
        "[Varibobi-WildFires](https://www.independent.co.uk/climate-change/news/greek-fires-heatwave-athens-heatwave-b1896861.html), "
        "[Rethymno-WildFores](https://greekreporter.com/2022/07/16/wildfires-raging-rethymno-crete/), "
        "[Mati-WildFires](https://greekreporter.com/2021/07/23/deadly-wildfire-at-mati-still-haunts-greece-two-years-later/), "
        "[Corinthos-WildFires](https://greekreporter.com/2021/08/14/wildfires/), "
        "[HistoricalFires-paralaxi](https://parallaximag.gr/life/istorikes-foties-pou-pligosan-tin-ellada), "
        "[BiggestFires-thetoc](https://www.thetoc.gr/koinwnia/article/o-kuklos-ton-kamenon-dason-pos-na-apofugoume-tis-katastrofikes-purkagies/), "
        "[ForestFires-wwf](http://www.oikoskopio.gr/pyroskopio/pdfs/pyrkagies-ellada.pdf), "
        "[ForestFires-gnd](https://www.mononews.gr/afieromata/new-green-deal/ellada-se-pies-perioches-tha-afxithoun-i-dasikes-pirkagies), "
        "[BigForestFires-newsbeast](https://www.newsbeast.gr/greece/arthro/9517368/oi-10-pio-akraies-dasikes-pyrkagies-stin-ellada-ta-teleftaia-20-chronia-eginan-stachti-2-800-000-stremmata)"
    )

elif nav == "Data Preview & Analytics":
    st.header("Data Preview")
    st.subheader("Daily Data")
    st.write("This section presents the time series for all stations described in the **Data Source** section . "
             "The timeseries shown in the graph below, describe daily records for the average daily temperature.")
    st.write("As mentioned in the previous unit 'Data source', the selected dataset come with some extra information in their csv"
             " file, such as the coordinates of each station and the Temperature metric. In order to use the data further,"
             " I removed the irrelevant information and only kept a column for the Datetime of the row and its "
             "average daily Temperature. Right after that editing process, the final selected timeseries look like this:")

    file_list = joblib.load("joblib_objects/namesofMySeries.joblib")
    selected_file = st.selectbox("**Please select a station to display**", file_list, key='option1')

    df = pd.read_csv(os.path.join(os.getcwd(), f"daily_temperature/{selected_file}.csv"))
    st.dataframe(df)

    st.write("By default all the station are selected. If you want to isolate a line and study the timeseries for "
             "one station, you can double-click on its name at the legend bar on the right of the plot. Also, you have"
             "the ability to exclude a station from the diagram, by just clicking its name from the legends bar."
             "  If you want to see more detailed values of Temperature for a specific period of time, you can zoom in to"
             " the diagram, by marking the time interval you want.")
    # load JSON file and create figure
    with open("plots/daily_data.json", "r") as f:
        fig_json = f.read()
    fig = pio.from_json(fig_json)

    # display figure in Streamlit app
    fig.update_layout(
        width=1250,  # set the width of the figure
        height=600,  # set the height of the figure
    )
    st.plotly_chart(fig)
    st.write(
        "From the above diagram we notice that for all stations, there is a **seasonal pattern**, a recurring motif "
        "every once a year. We see the same in the following charts. In order to better conduct seasonality, I "
        "reproduce our data for all stations with weekly, monthly and per year frequency. Thus, "
        "it can be confirmed that the repeated motifs in the selected dataset are once year. At first sight, there seems to be no "
        "trend, as the average daily Temperature per year seems to remain stable.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Weekly Temperatures")
        with open("plots/weakly_data.json", "r") as f:
            fig_json = f.read()
        fig = pio.from_json(fig_json)

        # display figure in Streamlit app
        fig.update_layout(
            width=650,  # set the width of the figure
            height=500,  # set the height of the figure
        )
        st.plotly_chart(fig)
    with col2:
        st.subheader("Monthly Temperatures")
        with open("plots/monthly_data.json", "r") as f:
            fig_json = f.read()
        fig = pio.from_json(fig_json)
        fig.update_layout(
            width=650,  # set the width of the figure
            height=500,  # set the height of the figure
        )
        st.plotly_chart(fig)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Statistic metrics with box plots")
    st.write("In order to get a better glance on the **trend**,some **box plots** are visualised bellow for every station per year.")
    st.write("A box plot is a graphical representation of a dataset that displays the distribution of the data"
             " through its quartiles. The box represents the inter-quartile range (IQR), which is the range between "
             "the first quartile (Q1) and the third quartile (Q3). The box spans the middle 50% of the data. For "
             "example for the station in arkadia and the year of 2013, Q1 represents the value below which 913 of"
             " the values fall (since 913/3653 = 0.25, or 25%), and Q1s average temperature is 10.6. Similarly, "
             "we can see the mean, and the average temperatures for other quartiles, and so its maximum and minimum"
             " daily Temperatures.")
    file_list = joblib.load("joblib_objects/namesofMySeries.joblib")
    station = st.selectbox("**Please select a station to display**", file_list, key='option2')

    with open(f'plots/daily_{station}_boxplot.json', "r") as f:
        fig_json = f.read()
    fig = pio.from_json(fig_json)

    # display figure in Streamlit app
    fig.update_layout(
        width=1250,  # set the width of the figure
        height=600,  # set the height of the figure
    )
    st.plotly_chart(fig)
    st.markdown("<small>For the year of 2023, there seems to be no box plot beacuse we have data only for the first "
                "day of the year.</small>", unsafe_allow_html=True)
    st.write("From the charts above, we can not draw with certainty any conclusion about a trend, as it seems "
             "that the mean Temperature per year follows a 'Wavy' flow and does not follow neither a ascending nor a "
             "descending order. What we can observe, however, is that the years **2017** and **2021** mark the highest"
             " average temperatures of all the years and for all the stations. If we had data for more years, we might "
             "be able to draw some kind of conclusion about the high-temperature years, so that we can draw better "
             "conclusions about the fluctuating flow of the media that we see from our charts. But due to the lack of "
             " data from the previous years, we can not draw any conclusions for the trend, but only claim that there"
             " is none.")
    st.write("A very important conclusion coming from the observation of the box plots above, is that there are **no "
             "outliers** in all of our 11 timeseries. More specifically, it does not seem to have "
             "any data point either above the maximum either bellow the minimum value in neither box plots."
             "That information will help me with the Timeseries Clustering in the next unit, in order to choose "
             "a normalizer/scaler.")

    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     image_path = f'plots/{station}_lag_autocorelation.png'
    #     image = Image.open(image_path)
    #     st.image(image, caption='')
    # with col2:
    #     st.write(
    #         "For all time series in our dataset, it is observed  a slight decay at the ends of their diagrams, suggesting a decreasing "
    #         "autocorrelation as the delay increases. In other words, the effect of past temperature "
    #         "values, on current observation decreases as the time lag grows. It is therefore observed that the "
    #         "temperature values for all time series analyzed in this thesis do not show significant long-term "
    #         "dependence or memory. The above observation is also proved by the diagrams of autocorrelation versus "
    #         "the values of delay, as you can see in the diagram on the left.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Auto-Corellation")
    st.write("Next, we explore the **auto-correlation** of data, with the help of the above Lag plots. Auto-correlation"
             " analysis measures the relationship of the observations between the different points in time, and thus"
             " seeks a pattern or trend over the timeseries. In a lag plot, each data point in the time series is "
             "plotted against the data point that is a fixed number of time steps (lags) earlier.The lag plot can be"
             " used to detect whether a time series is random or has some level of auto-correlation. If the time series"
             " is random, then the points in the lag plot should be scattered randomly and evenly. If the time series"
             " has auto-correlation, then the points in the lag plot will form a pattern, such as a diagonal line or a"
             " curve.")
    station = st.selectbox("**Please select a station to display lag plots for**", file_list, key='option3')
    with open(f'plots/daily_{station}_lag.json', "r") as f:
        fig_json = f.read()
    fig = pio.from_json(fig_json)

    # display figure in Streamlit app
    fig.update_layout(
        width=1250,  # set the width of the figure
        height=600,  # set the height of the figure
    )
    st.plotly_chart(fig)
    col1, col2 = st.columns([1, 3])
    with col1:
        image_path = f'plots/{station}_lag_autocorelation.png'
        image = Image.open(image_path)
        st.image(image, caption='')
    with col2:
        st.write(
            "From the lag plots, for all of our station, we see a positive auto-correlation. Positive auto-correlation "
            "means that the increase observed in a time interval leads to a proportionate increase in the lagged time"
            " interval. In more detail, for my dataset the temperature the next day tends to rise when it’s been "
            "increasing and tends to drop when it’s been decreasing during the previous days. And so, that "
            "relationship is shown in the plots above.")
        st.write(
            "For all time series in our dataset, it is observed  a slight decay at the ends of their diagrams, suggesting a decreasing "
            "autocorrelation as the delay increases. In other words, the effect of past temperature "
            "values, on current observation decreases as the time lag grows. It is therefore observed that the "
            "temperature values for all time series analyzed in this thesis do not show significant long-term "
            "dependence or memory. The above observation is also proved by the diagrams of autocorrelation versus "
            "the values of delay, as you can see in the diagram on the left.")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Seasonal Decomposition")
    st.write("At this point, we studied some plots in order to get more information about the trend, the seasonality,"
             "the distribution of our data with various of charts, like the boxplot, the line chart lag plots etc. "
             "Although, there is one method which provides us with more info about all the important"
             "features of our timeseries, in just three plots, and it is called **Seasonal Decomposition**.")
    st.write("Seasonal decomposition is a statistical technique used to separate a time series into its individual"
             " components, including trend, seasonal, and residual components.But, in order to perform the seasonal"
             " decomposition method, it is necessary to implement adf tests for all our time series, and see if our"
             " data are stationary and decide what decomposition method to use."
             " Augmented Dickey-Fuller (ADF) test is a commonly used ")
    st.markdown("After implementing ADF tests for all of our timeseries, it was shown that all of our stations, contain"
                "stationary data. That been said, we can now use the model <u>additive</u> in our seasonal "
                "decomposition, as the seasonal fluctuations appear to be roughly constant over time.",
                unsafe_allow_html=True)

    namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
    station = st.selectbox("**Please select the station you wish to see its seasonal decomposition**", namesofMySeries,
                           key='option5')
    with open(f'plots/daily_{station}_seasonal_decomposition.json', "r") as f:
        fig_json = f.read()
    fig = pio.from_json(fig_json)

    # display figure in Streamlit app
    fig.update_layout(
        width=1250,  # set the width of the figure
        height=600,  # set the height of the figure
    )
    st.plotly_chart(fig)
    st.write(
        "It might be incorrect to claim that there is a trend in any of our timeseries, because for none of our "
        "stations the overall direction of the data is increasing or decreasing, or in general follows a specific"
        " pattern. Although, we could claim that there is a long term pattern, containing a convex and a concave "
        "curve in between in every two years, but for some stations such as in Keratea this trend is not as "
        "obvious as other stations, such as in Trikala."
        " In addition, it is important to mention that some stations seem to have the same trend flow, such as"
        " the stations in larissa and Trikala, or the stations in Halkidiki, Corinthos and Arkadia. This "
        "observation, might be important in the future steps and more specifically in the clustering unit."
        "Another important observation coming from the third plot, is that all of our stations "
        "have seasonality of 1 year. That strong seasonality pattern, might be also reflected in the trend component.")


elif nav == "Data Clustering":
    st.header("Data Clustering")
    st.write("Welcome to the data clustering page of our app! Here, you can explore the clustering results of two "
             "popular timeseries clustering methods, and analyze their evaluation. Time series clustering is a"
             " powerful tool for grouping similar time series together based on their patterns, trends, and behaviors. "
             "And for that occasion, we are using the Self Organising Maps and the Timeseries Kmeans clustering "
             "algorithms. "
             "Before experimenting with these clustering techniques, it was necessary to use a normalization technique, "
             "just to ensure that each variable in the dataset is given equal weight during the clustering process. "
             "There are many options for this exact task, but each method has of course its own pros and cons, and after"
             "analyzing our data and studying 3 of the most important scaler, the MinMax scaler seemed the best fit."
             "With Min Max scaler, the minimum and maximum values of a feature in the dataset are identified, "
             "and the values of the feature are rescaled to fit within a specific range, typically between 0 and 1. T"
             "his is achieved by subtracting the minimum value from each data point, and then dividing the result by "
             "the difference between the maximum and minimum values of the feature. It is useful for data that are"
             "not normally distributed and do not contain outliers, both of these criteria feature tour dataset. "
             "At last, the evaluation of these clustering techniques is implemented with the Dynamic"
             "Time Wraping matrix for each timebased data frequency.")
    st.subheader("Som Clustering")
    st.write("**How does it work?**")
    st.write("SOM clustering on timeseries involves using a Self-Organizing Map (SOM) algorithm to cluster "
             "similar patterns within a time series dataset, creating a map of neurons that group together based on "
             "similarities in their input time series. It uses a "
             "low-dimensional grid to represent high-dimensional input data.")
    st.write(
        "With that being said, SOM is one of the best fitted clustering algorithms for our dataset, and with some"
        "tuning we can take a further look at the different results it produces and evaluate them with the DTW"
        "distance method -as mentioned above- and select that best fitted parameters.")

    st.write("Getting started with the tuning of the som models, the first parameters we need to tune are the output "
             "space of the model (som x and som y), which are used in order to identify the location of "
             "each node in the SOM grid. In order to identify an appropriate SOM grid size, we used PCA."
             "After finding the best values for somx and somy for every time-based data, we tune the activation"
             "distance that activates the map, by experimenting with cosine, manhattan and euclidean distance. "
             "Last but not least, we tune the size of epochs in order to get better convergence of the SOM "
             "and avoid over-fitting.")
    st.write("As far as the evaluation of the hyperparameter tuning goes, we use the quantization error.In SOM, the "
             "quantization error is calculated as the average distance between each data point and its closest node in"
             " the SOM grid. SOM assigns each data point to the closest node in the SOM grid and calculates the "
             "distance between the data point and the weight vector of the closest node.")
    st.write("Here you can select the time frequency of the data, which as mentioned before is daily monthly or yearly,"
             "and take a look at the results of the som clustering method, and also the distance metric for the "
             "clustering's activation function.")
    timebased = st.selectbox("**Please select the time-based frequency of the data**", ["daily", "monthly", "yearly"],
                             key='option5')
    shape_method = st.selectbox("**Please select the method for the activation function**",
                                ["euclidean", "manhattan", "cosine"],
                                key='option6')
    # If file is uploaded
    df = pd.read_csv(f"som/som_tuning_results_{timebased}_som_xy_combos.csv")
    filtered_df = df[df['shape_method'] == shape_method]
    filtered_df = filtered_df.drop('shape_method', axis=1)
    # Display the DataFrame using the `dataframe` function
    st.dataframe(filtered_df)
    st.write("Based on the quantization error, for the daily frequence data, the best parameter combination is the "
             "euclidean distance for 200 epochs, while for monthly data, the best combination based on the quantization"
             " error is the manhattan distance again for 200 epochs, and the yearly data, result into best combination"
             " for cosine distance of 10 epochs, as well. As fas as the silhouette score goes, for daily data the "
             "best parameter combination was the Manhattan with 10 epochs, for monthly the manhattan with 50 epochs and"
             "for yearly euclidean with 100 epochs. As we can see there is not any specific metric that gives the best scores"
             "for my metrics, but it seems that the euclidean distance, scores the best optimized scores and error value"
             "for both daily and motnhly data with 10 and 100 epochs each. While for yearly frequenced data, best fit seems to be the"
             "cosine metric with 50 epochs. ")
    st.subheader("KMeans Clustering")
    st.write("**How does it work?**")
    st.write("K-means clustering on timeseries involves using the K-means algorithm to partition a time series dataset"
             " into k clusters, based on similarities in their values across time. The algorithm iteratively assigns"
             " data points to the closest cluster centroid and updates the centroids until convergence.")
    st.write("The first thing we need to know about kmeans is the value of k clusters. In order to find"
             "the best fitted number of clusters, we used the silhouette score. The silhouette score, measures the "
             "quality of the clustering output based on how similar a point is to its own cluster "
             "compared to other clusters. In fact for daily data the best K value is 5, for monthly data is 6 and "
             "for yearly, 3. As we have calculated the number of clusters, we try different distance metrics, that"
             " reused for both cluster ssignment and barycenter computation. Thus we expreriment with euclidean,"
             " dtw and soft dtw distance.")
    st.write("As mentioned above, quantization error is an important metric in evaluating the quality of the SOM "
             "clustering model and can help to identify potential issues or limitations of the method. In K-means"
             " clustering, the quantization error is calculated as the sum of the squared distances between each data"
             " point and its corresponding centroid. An obvious question arising from the abandonment of error "
             "to kmeans, is what accounts for the great difference in its prices from those of som clusteringSince "
             "K-means clustering does not reduce the dimensionality of "
             "the input data, the quantization error values can be larger and may range up to thousands or even "
             "millions, depending on the scale and range of the input data.")
    timebased = st.selectbox("**Please select the time-based frequency of the data**", ["daily", "monthly", "yearly"],
                             key='option7')
    # If file is uploaded
    total_df = pd.read_csv(f"kmeans/kmeans_timebased_total_df.csv")
    mask = total_df['timebased'] == timebased
    sub_df = total_df[mask]
    sub_df = sub_df.drop('timebased', axis=1)
    st.dataframe(sub_df)
    st.write("A smaller score but with small differences, for daily, monthly and yearly data, seems to have"
             " Euclidean distance. Of course, below we will see another method of evaluating the results in order to "
             "confirm or conceal the above-mentioned results.")
    st.subheader("Evaluating Clustering Results")
    st.write("**Evaluating the quantization errors**")

    st.write("For Kmeans clustering the best quantization error comes from the usage of the soft dtw distance on "
             "daily, monthly and yearly data with quantization error of 11,647,644.56, 11,649,477.22 and 144.1405899"
             "for each timeseries frequency. The clusters for each best fitted combination are listed bellow:")
    kdf = pd.read_csv(f"kmeans/kmeans_timebased_total_df.csv")
    mask_metric = kdf['metric'] == 'softdtw'
    sub_df = kdf[mask_metric]
    timebased = st.selectbox("Please select the time-based frequency of the data", ["daily", "monthly", "yearly"],
                             key='option11')
    metric_mask = sub_df['timebased'] == timebased
    sub_df = sub_df[metric_mask]
    sub_df = sub_df.drop(columns=['metric', 'silhouette_score', 'quantization_error', 'timebased'], axis=1)
    missing_cols = sub_df.columns[sub_df.isna().any()]
    sub_df = sub_df.drop(columns=missing_cols)
    st.dataframe(sub_df)
    st.write("As regards to som clustering, due to the multiple combinations of parameters, we have displayed the top"
             "5 combination with the minimum quantization errors."
             ""
             " You can see top 5 combinations of parameters for the Som clustering, on the array bellow:")
    timebased = st.selectbox("Please select the time-based frequency of the data", ["daily", "monthly", "yearly"],
                             key='option8')
    df = pd.read_csv(f"som/som_top5.csv")
    mask = df['timebased'] == timebased
    sub_df = df[mask]
    sub_df = sub_df.drop('timebased', axis=1)
    sub_df = sub_df.drop('top', axis=1)

    st.dataframe(sub_df)
    st.write("**Evaluating the Silhouette score**")
    st.write("Silhouette score is a measure of how well each data point fits into its assigned cluster in a clustering"
             " algorithm. It is a popular evaluation metric used to compare different clustering algorithms "
             "and parameter settings, and can be used to guide the selection of the best clustering model"
             " for a given dataset. In our case, we use the silhouette score to find the best model for daily, monthly"
             "and yearly time-based data from som and kmeans clustering results.")
    st.write("When evaluating a model, it is important to consider both the quantization"
             " error and the silhouette score, as they provide different information about the clustering results. "
             "A good clustering model should have a low quantization error and a high silhouette score, indicating "
             "that the clusters are tightly packed together and well-separated.")
    st.write(
        "In the dataframe displayed bellow, we can see the top 5 combinations of parameters on the som clustering, "
        "for maximum silhouette scores:")
    timebased = st.selectbox("Please select the time-based frequency of the data", ["daily", "monthly", "yearly"],
                             key='option10')
    col1, col2 = st.columns([1, 1])
    with col1:
        df = pd.read_csv("som/som_sil_top.csv")
        mask = df['timebased'] == timebased
        sub_df = df[mask]
        sub_df = sub_df.drop('timebased', axis=1)
        st.dataframe(sub_df)
    with col2:
        image = Image.open(f"clustering_evaluation/heatmap_{timebased}.png")
        st.image(image, width=350)
    if timebased == 'daily':
        st.write(
            "The best model based only on the silhouette score is the manhattan distance with 10 epochs, but regard on "
            "both silhouette score and quantization error, that should be the manhattan distance with 100 epochs."
            "For the second case the clusters are shown bellow:")
        df = pd.read_csv(f"som/som_tuning_results_daily_som_xy_combos.csv")
        shaped_df = df[df['shape_method'] == 'manhattan']
        epoched_df = shaped_df[shaped_df['epochs'] == 100]
        daily_df = epoched_df.drop(columns=['epochs', 'shape_method', 'silhouete_score', 'quantization_error'])
        missing_cols = daily_df.columns[daily_df.isna().any()]
        daily_df = daily_df.drop(columns=missing_cols)
        st.dataframe(daily_df)
    if timebased == 'monthly':
        st.write("The best model based only on the silhouette score is the manhattan distance with 50 epochs, but "
                 "considering also the quantization error, the best performance was by the combination of the manhattan"
                 "distance and the 200 number of epochs.For the second case the clusters are shown bellow:")
        df = pd.read_csv(f"som/som_tuning_results_monthly_som_xy_combos.csv")
        shaped_df = df[df['shape_method'] == 'manhattan']
        epoched_df = shaped_df[shaped_df['epochs'] == 200]
        monthly_df = epoched_df.drop(columns=['epochs', 'shape_method', 'silhouete_score', 'quantization_error'])
        missing_cols = monthly_df.columns[monthly_df.isna().any()]
        monthly_df = monthly_df.drop(columns=missing_cols)
        st.dataframe(monthly_df)
    if timebased == 'yearly':
        st.write(
            "The best model based only on the silhouette score, is the one with cosine distance for 10 epochs, while"
            "considering also the quantization error, the best model is the one with the cosine distance again but with"
            "50 epochs.For the second case the clusters are shown bellow:")
        df = pd.read_csv(f"som/som_tuning_results_yearly_som_xy_combos.csv")
        shaped_df = df[df['shape_method'] == 'cosine']
        epoched_df = shaped_df[shaped_df['epochs'] == 50]
        yearly_df = epoched_df.drop(columns=['epochs', 'shape_method', 'silhouete_score', 'quantization_error'])
        missing_cols = yearly_df.columns[yearly_df.isna().any()]
        yearly_df = yearly_df.drop(columns=missing_cols)
        st.dataframe(yearly_df)
    st.subheader("Results")
    st.write("Our next step is to create forecasting models for the daily data, that will be able to predict"
             "the temperature of the next day. Thus, we need to result into one clustering model and tune each cluster."
             " Based on the result of the KMeans and the SOM clustering tuning, the highest silhouette score is the "
             "model of Som algorithm with manhattan distance of 100 epochs!")

elif nav == "Data Forecasting":
    st.header("Data Forecasting")
    st.write("Welcome to my timeseries forecasting page!")
    st.write("After computing the clusters of the timeseries, we implement hyperparameter tuning on 3 types of LSTM "
             " models for every cluster. By doing so, we result into the 5 models (as many as our clusters for daily"
             "data) and we retrain each and every station timeseries with the resulted parameters. In order to decide"
             "the best model for each cluster, we use 2 evaluation metrics, the MSE (Mean Squared Error) and MAE"
             " (Mean Absolute Error). By using both metrics, we can gain a better understanding of the strengths"
             " and weaknesses of the model."
             " For example, if the MSE is low but the MAE is high, it may indicate that the model is good at predicting"
             " most values accurately but struggles with outliers. Similarly, if the MAE is low but the MSE is high,"
             " it may indicate that the model is good at predicting outliers accurately but struggles with smaller"
             " errors."
             "As far as the tuning goes, we try tuning the number of timesteps (in days) the number of neurons,"
             "the batch_size and the dropout level.")
    st.subheader("**Understanding the parameters of the tuning**")
    st.write("- **Activation Function**: Different activation functions can have different strengths in allowing the "
             "model to learn and remember information over time, so choosing the appropriate one for the task at hand "
             "can improve the performance of the LSTM model. In our case, for our activation functions, we tune "
             "the layer with relu linear or tanh, that are more appropriate for regression problems.")
    st.write("- **Timesteps**:  If the look_back parameter is too small, the model may not capture important "
             "dependencies between past and future data points. In addition, The look_back parameter can have a"
             " significant impact on the accuracy of the predictions. By tuning the look_back parameter, we can find"
             " the optimal number of time steps to use as input for each prediction, resulting in more accurate "
             "predictions.")
    st.write("- **Dropout rate**: Using a dropout rate in your neural network helps prevent overfitting by randomly "
             "deactivating a fraction of the neurons during training, forcing the network to learn more robust and"
             " generalized features, ultimately improving its performance on unseen data.")
    st.write("- **Layers**: The number of layers in an LSTM determines the model's complexity. A deeper network with"
             " more layers can learn more complex temporal relationships in the input data, which can improve the"
             " model's performance.Also,  Increasing the number of layers in the model can increase the memory capacity"
             " of the network, which can be important for tasks that require long-term dependencies, such as language "
             "modeling or speech recognition. But, Adding too many layers to an NN can result in overfitting, where "
             "the model becomes too complex and starts to memorize the training data instead of generalizing to new "
             "data. You can see the type of models and the layer combinations that is used right bellow:")
    st.subheader("**Type of models**")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            """
            1. **Type model 1:** LSTM model with input layer of input shape (1, timesteps) and selected activation 
            function between relu, linear or tanh (tuning parameter), one hidden Lstm layer with activation function
            based on the hyperparameter tuning one Dropout Layer and one Dense Layer as the output layer of adam optimizer which 
            focuses in the optimization of the mse loss function. 
            2. **Type model 2:** Lstm model with input layer of input shape (1, timesteps) and selected activation 
            function relu, tanh or linear, one Dense hidden layer of 50 neurons , one Lstm hidden layer, again with 50 
            neurons and activation function selected by the hyperparameter tuning, and one last Dense layer as the output
             layer with relu as activation function. 
            """
        )
    with col2:
        cl = st.selectbox("You can look into network examples for every type of our Lstm models, by selecting the"
                          "the model type that you want to visualize:",
                          ["Type model 1", "Type model 2"])
        st.write("In order to visualize an example for every model type, we use random values for the parameters")
        m_type = cl[-1]
        image_path = f"lstm_new_results/lstm_network{m_type}.png"
        image = Image.open(image_path)
        st.image(image, caption=f'Type Model {m_type}', width=150)

    st.subheader("**Results of Hyperparameter tuning**")
    st.write("In this section, you can see the results of the hyperparameter tuning for every cluster and for every "
             "model type, by selecting each cluster name.The clusters and their stations are mentioned bellow:")
    st.markdown("""
    - Cluster 1: Varibobi
    - Cluster 2: Halkidiki, Trikala
    - Cluster 3: Rethymno
    - Cluster 4: Corinthos, Keratea, Kineta
    - Cluster 5: Arkadia, Larissa, Mati, Messinia
    """)
    cluster = st.selectbox(
        "**Please select the Cluster, that you want to see the hyperparameter tuning results for:**",
        ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"],
        key='option12')
    cl_option = cluster[-1]
    type = st.selectbox(
        "**And the type model of the cluster::**",
        ["Type model 1", "Type model 2"],
        key='option13')
    type_model = type[-1]
    col1, col2 = st.columns([2, 1])
    with col1:
        dataframe = pd.read_csv(f"lstm_new_results_final/cluster{cl_option}_results.csv")
        dataframe = dataframe[dataframe['type_model'] == int(type_model)]
        dataframe = dataframe.loc[:,
                    ['input_lst_layer', 'lstm_activations', 'time_steps', 'dropout', 'train_mse', 'train_mae',
                     'test_mse',
                     'test_mae']]
        st.write(
            "During the hyperparameter tuning of every model, mentioned above, the best combination of the hyperparameters"
            "are shown bellow in the table")
        st.dataframe(dataframe)

    with col2:
        st.write(
            "The winning tuning model for every cluster and for each type of model is shown bellow, and by selecting different cluster in"
            " the box above, you can see the resulted model for each cluster.")

        image_path = f"lstm_new_results/lstm_network_cl{cl_option}_type{type_model}.png"
        image = Image.open(image_path)
        st.image(image, caption=f'Model for cluster {cl_option} and type {type_model}', width=200)
    st.subheader("Results")
    st.write("The results of the Hyperparameter tuning, for all clusters and models, are shown bellow in the table:")
    results = pd.read_csv('plots/best_results.csv')
    st.dataframe(results)
    st.write(
        "A first observation based on the results above, is that the dropout rate of 0.2 seems to perform much"
        "better than the one of 0.5 for most of our parameter combinations. On the other hand, the activation "
        "functions on both hidden and input layer do not seem to follow any pattern, and either the time_steps. "
        "Finally, another big note from the above mentioned results, is that the type model 1, performs extremely "
        "better for each cluster than the second type.")
    st.write("From the above results it follows that for all classes, the first type of model with their corresponding"
             " combination of parameters presents better results. This may be due to the complexity of the second"
             " model, a feature which prevents the learning of patterns and the imprinting of time series patterns. "
             "At the same time, the placement of LSTM levels seems to contribute initially, as in the first species, "
             "an lstm level is placed after the input level, a feature which allows the model to process data"
             " appropriately and record their time patterns. Less complexity also characterizes the first model "
             "because of its architecture. In particular, the architecture of the first model, with an LSTM plane"
             " followed by a dropout plane and a dense plane/dense layer , provides a good balance of complexity and"
             " model capacity. The LSTM level can capture long-term dependencies and time patterns, while the dropout "
             "layer helps prevent overfitting., but also to better capture the components of the time series of the "
             "dataset of this thesis.")
    df = joblib.load("joblib_objects/cluster_stations.joblib")
    station = st.selectbox(
        "Now, on the following plot, you can see the results of the predictions and the test temperature data, "
        "for the station of your choice. **Please Select** the station of the cluster, that you want to see the results for, based on "
        " the hyperparameter tuning as mentioned above.", df[f"cluster{cl_option}"])
    with open(f'json_predictions/cl{cl_option}_st{station}_pred.json', "r") as f:
        fig_json = f.read()
    fig = pio.from_json(fig_json)

    # display figure in Streamlit app
    fig.update_layout(
        width=1200,  # set the width of the figure
        height=500,  # set the height of the figure
    )
    st.plotly_chart(fig)


else:
    st.header("Contact Us")
    st.markdown('Thank you for reading my thesis and I hope you found it interesting! '
                'In case you have any questions or ideas you can find me on my email '
                'to stefanatougerasimina@gmail.com.')
    st.subheader("Graditude")
    st.write("I would like to thank my professor Maria Halkidi for supervising my thesis!")
