import pandas as pd
import numpy as np
import re
import datetime as dt
import math
import geopandas as gpd
import h3 # h3 bins from uber
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, minmax_scale, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.cluster.hierarchy as sch
import holidays
from fastai.vision.all import * # Needs latest version, and sometimes a restart of the runtime after the pip installs
from sklearn_extra.cluster import KMedoids



def clean_weather_data(df_weather):
    '''
    Fills the missing information by looking at the previous and the following existing values
    and then incrementaly distributing the difference over the missing days.
    This guarantees a smooth development of the weather data over time.
    '''
    
    missing = df_weather[pd.isnull(df_weather).any(1)].index
    
    if len(missing) > 0:
        for col in df_weather.columns[1:]:
            before = df_weather.loc[missing[0]-1, col]
            after = df_weather.loc[missing[-1]+1, col]
            diff = after - before
            for i in range(len(missing)):
                df_weather.loc[missing[i], col] = before+diff/(len(missing)+1)*(i+1)
                
    return df_weather



def add_weather_change(df_weather):
    df_weather["change_water_atmosphere"] = 0
    df_weather["change_temperature"] = 0
    for row in range(df_weather.shape[0]):
        if row == 0:
            df_weather.loc[row, "change_water_atmosphere"] = 0
            df_weather.loc[row, "change_temperature"] = 0
        else:
            df_weather.loc[row, "change_water_atmosphere"] = df_weather.loc[row, "precipitable_water_entire_atmosphere"] - df_weather.loc[row-1, "precipitable_water_entire_atmosphere"]
            df_weather.loc[row, "change_temperature"] = df_weather.loc[row, "temperature_2m_above_ground"] - df_weather.loc[row-1, "temperature_2m_above_ground"]
    return df_weather



def join_accident_to_weather(df_accident, df_weather):
    '''
    Left-joins the accident data to the weather data, resulting in a dataframe containing the weather information
    for every day as well as the aggregated accidents.
    '''
    
    # Count accidents per day and leftjoin to weather dataframe
    df_accident["date"] = df_accident["datetime"].apply(lambda x: x.date())
    if type(df_weather.loc[0, "Date"]) is not dt.date:
        df_weather["Date"] = df_weather["Date"].apply(lambda x: x.date())
    accident_count = df_accident.groupby("date").count()["uid"].reset_index()
    df_combined = df_weather.merge(accident_count[["date", "uid"]], left_on="Date", right_on="date", how='left')
    
    # Fill NaNs with zeros
    df_combined.fillna(value=0, inplace=True)
        
    # Drop duplicate Date column
    df_combined.drop("date", axis=1, inplace=True)
    
    # Rename column
    df_combined.rename(columns={"Date":"date", "uid":"accidents"}, inplace=True)
    
    # Adding column with 1 for sundays and holidays, 0 for working days
    df_combined["sun_holiday"] = df_combined["date"].apply(lambda x: 1 if (x.weekday() == 6) or (x in holidays.Kenya()) else 0)
    
    # Change type to integer
    df_combined["accidents"] = df_combined["accidents"].astype("int")
    
    return df_combined



def scale_pca_weather(df_combined):
    '''
    Scaling and analysing the principal components of the weather data.
    '''
    
    # Scaling
    mm_scaler = MinMaxScaler()
    X_mm = df_combined[["precipitable_water_entire_atmosphere", "relative_humidity_2m_above_ground",
                        "specific_humidity_2m_above_ground", "temperature_2m_above_ground"]]
    X_mm_scaled = mm_scaler.fit_transform(X_mm)

    std_scaler = StandardScaler()
    X_std = df_combined[["u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground",
                         "change_water_atmosphere", "change_temperature"]]
    X_std_scaled = std_scaler.fit_transform(X_std)

    X_scaled = pd.DataFrame(np.concatenate((X_mm_scaled, X_std_scaled), axis=1), columns=["precipitable_water_entire_atmosphere", "relative_humidity_2m_above_ground",
                              "specific_humidity_2m_above_ground", "temperature_2m_above_ground", "u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground",
                               "change_water_atmosphere", "change_temperature"])
    
    # Principal componant analysis (PCA)
    pca = PCA(n_components=0.99)
    pca_decomposition = pca.fit(X_scaled)
    X_pca = pca_decomposition.transform(X_scaled)
    df_combined_pca = pd.DataFrame(X_pca)
    df_combined_pca = df_combined_pca.join(df_combined[["date", "accidents", "sun_holiday"]])
    
    return df_combined_pca

def split_combined(df_combined_pca, predict_period='2019_h2'):
    if predict_period == '2019_h1':
        X_train = df_combined_pca[df_combined_pca["date"] < dt.date(2019, 1, 1)][[0, 1, 2, 3, 4, "sun_holiday"]]
        y_train = df_combined_pca[df_combined_pca["date"] < dt.date(2019, 1, 1)]["accidents"]
        X_test = df_combined_pca[(df_combined_pca["date"] >= dt.date(2019, 1, 1)) & (df_combined_pca["date"] < dt.date(2019, 7, 1))][[0, 1, 2, 3, 4, "sun_holiday"]]
    elif predict_period == '2019_h2':
        X_train = df_combined_pca[df_combined_pca["date"] < dt.date(2019, 7, 1)][[0, 1, 2, 3, 4, "sun_holiday"]]
        y_train = df_combined_pca[df_combined_pca["date"] < dt.date(2019, 7, 1)]["accidents"]
        X_test = df_combined_pca[(df_combined_pca["date"] >= dt.date(2019, 7, 1)) & (df_combined_pca["date"] < dt.date(2020, 1, 1))][[0, 1, 2, 3, 4, "sun_holiday"]]
    return X_train, X_test, y_train


def predict_poly(X_train, X_test, y_train):
    poly = PolynomialFeatures(degree=4)
    X_train_poly = poly.fit_transform(X_train.drop("sun_holiday", axis=1))
    lin_poly = LinearRegression()
    lin_poly.fit(X_train_poly, y_train)
    X_test_poly = poly.transform(X_test.drop("sun_holiday", axis=1))

    return lin_poly.predict(X_test_poly)
    

def predict_accidents_on_weather(df_accident, df_weather, predict_period='2019_h1'):
    '''
    Takes the raw data and returns the number of predicted road traffic accidents for every day in the predict period:
    First half 2019 : (predict_period='2019_h1') 
    Second half of 2019 : (predict_period='2019_h2') 
    '''
    
    df_weather = clean_weather_data(df_weather)
    df_weather = add_weather_change(df_weather)
    df_combined = join_accident_to_weather(df_accident, df_weather)
    df_combined_pca = scale_pca_weather(df_combined)
    X_train, X_test, y_train = split_combined(df_combined_pca, predict_period=predict_period)
    y_pred = predict_poly(X_train, X_test, y_train)
    y_pred = [0 if i < 0 else i for i in y_pred]
    return y_pred



def create_crash_df(train_file = '../Inputs/Train.csv'):  
    '''
    loads crash data from input folder into dataframe
    '''
    crash_df = pd.read_csv(train_file, parse_dates=['datetime'])
    return crash_df
    
def create_temporal_features(df, date_column='datetime'):
    '''
    Add the set of temporal features the the df based on the datetime column. Returns the dataframe.
    '''
    dict_windows = {1: "00-03", 2: "03-06", 3: "06-09", 4: "09-12", 5: "12-15", 
                    6: "15-18", 7: "18-21", 8: "21-24"}
    dict_months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    rainy_season = ["Mar", "Apr", "May", "Oct", "Nov", "Dec"]
    df["time"] = df[date_column].apply(lambda x: x.time())
    df["time_window"] = df[date_column].apply(lambda x: math.floor(x.hour / 3) + 1)
    df["time_window_str"] = df["time_window"].apply(lambda x: dict_windows.get(x))
    df["day"] = df[date_column].apply(lambda x: x.day)
    df["weekday"] = df[date_column].apply(lambda x: x.weekday())
    df["month"] = df[date_column].apply(lambda x: dict_months.get(x.month))
    df["half_year"] = df[date_column].apply(lambda x: 1 if x.month<7 else 2)
    df["rainy_season"] = df["month"].apply(lambda x: 1 if (x in rainy_season) else 0)
    df["year"] = df[date_column].apply(lambda x: x.year)
    df["date_trunc"] = df[date_column].apply(lambda x: x.date()) #this does something strange that breaks the code if higher
    df["holiday"] = df["date_trunc"].apply(lambda x: 1 if (x in holidays.Kenya()) else 0)
    df["weekday"] = df["date_trunc"].apply(lambda x: 7 if (x in holidays.Kenya()) else x.weekday())
    
    return df

def drop_temporal(df):
    '''
    helper function to remove all the granular temporal columns once they have been used for generating other columns for joining.
    '''
    df = df.drop(["day", "time_window", "time_window_str", "time_window_str", "month", "year", "weekday", "rainy_season", "date_trunc", "time", "half_year", "holiday"], axis=1)
    return df

def split_accident_df(data, strategy, test_size=0.3, random_state=42):
    '''
    Splits the data set into a train and a test set.
    strategy:
        random = splits off random indices, using test_size and random_state parameters
        year_2019 = splits the days of 2019 off into a test set
        percentage_month = splits off the last days of every month to the test set according to the test_size
        2nd_half_2018 = oversamples the months from July to December 2018 by about 33%
    '''
    if strategy == "random":
        data = data.sample(frac=1, random_state=random_state).reset_index().drop("index", axis=1)
        split_at = round(data.shape[0] * test_size)
        data_train = data.iloc[split_at:, :]
        data_test = data.iloc[:split_at, :]
    elif strategy == "year_2019":
        data_train = data[data["datetime"] < "2019-01-01"]
        data_test = data[data["datetime"] >= "2019-01-01"]
    elif strategy == "percentage_month":
        split_at = round(30 * (1-test_size))
        data_train = data.loc[data["day"] <= split_at]
        data_test = data.loc[data["day"] > split_at]
    elif strategy == "2nd_half_2018":
        train_samples = round(data.shape[0] * (1-test_size))
        test_samples = round(data.shape[0] * test_size)
        data_train = data.sample(n=train_samples, weights="half_year", random_state=random_state)
        data_test = data.sample(n=test_samples, weights="half_year", random_state=random_state)
        
    return data_train, data_test

def outlier_removal(crash_df, filter=0.00):

    if filter == 'hex_bin':
        crash_df = assign_hex_bin(crash_df)
        hex_bin_filter =  ['867a45067ffffff', '867a45077ffffff', '867a4511fffffff',
                           '867a4512fffffff', '867a45147ffffff', '867a4515fffffff',
                           '867a45177ffffff', '867a45817ffffff', '867a4584fffffff',
                           '867a4585fffffff', '867a458dfffffff', '867a458f7ffffff',
                           '867a45a8fffffff', '867a45b0fffffff', '867a45b17ffffff',
                           '867a45b67ffffff', '867a45b77ffffff', '867a6141fffffff',
                           '867a614d7ffffff', '867a616b7ffffff', '867a6304fffffff',
                           '867a632a7ffffff', '867a63307ffffff', '867a6331fffffff',
                           '867a6360fffffff', '867a63667ffffff', '867a6396fffffff',
                           '867a656c7ffffff', '867a65797ffffff', '867a6e18fffffff',
                           '867a6e1b7ffffff', '867a6e4c7ffffff', '867a6e517ffffff',
                           '867a6e59fffffff', '867a6e5a7ffffff', '867a6e5b7ffffff',
                           '867a6e657ffffff', '867a6e737ffffff', '867a6e797ffffff',
                           '867a6e79fffffff', '867a6e7b7ffffff', '867a6ecf7ffffff',
                           '867a6ed47ffffff', '867a6ed97ffffff', '867a6eda7ffffff' ]
        crash_df = crash_df.loc[~crash_df['h3_zone_6'].isin(hex_bin_filter)]
    else: 
        '''filters top and bottom quantiles of data based on filter input'''
        crash_df = crash_df.loc[crash_df['latitude'] < crash_df['latitude'].quantile(1-filter)]
        crash_df = crash_df.loc[crash_df['latitude'] > crash_df['latitude'].quantile(filter)]
        crash_df = crash_df.loc[crash_df['longitude'] < crash_df['longitude'].quantile(1-filter)]
        crash_df = crash_df.loc[crash_df['longitude'] > crash_df['longitude'].quantile(filter)]
    return crash_df

def assign_hex_bin(df,lat_column="latitude",lon_column="longitude", hexbin_resolution=6):
    '''
    Takes lat,lon and creates column with h3 bin name for three levels of granualirity.
    '''
    df["h3_zone_{}".format(hexbin_resolution)] = df.apply(lambda x: h3.geo_to_h3(x[lat_column], x[lon_column], hexbin_resolution),axis=1)
    return df

def plot_centroids(crash_data_df, centroids, cluster='cluster', labels = 'b'):
    '''
    plots the crash data points from crash_data_df and overlays the ambulance location from centroids. 
    Can be used in a loop by giving 'cluster' value as a parameter to label the chart with the cluster name.
    '''
    
    fig, axs = plt.subplots(figsize=(8, 5))
    plt.scatter(x = crash_data_df['longitude'], y=crash_data_df['latitude'], s=1, label='Crash locations', c=labels )
    plt.scatter(x = centroids[:,1], y=centroids[:,0], marker="x",
                color='r',label='Ambulances locations',s=100)
    axs.set_title('Scatter plot : Ambulaces locations vs Crash locations :'+cluster)
    plt.xlabel("latitude")
    plt.ylabel("longitude")
    plt.legend()
    plt.show()
        
def plot_dendrogram(df):
    '''Use Dendrogram to determine an optimal number of clusters'''
    plt.figure(figsize=(45,18))
    plt.title('Androgram')
    plt.xlabel('time_buckets_days')
    plt.ylabel('Euclidean distances')
    dendrogram = sch.dendrogram(sch.linkage(df, method = 'ward'))
    plt.show()

def calculate_TW_cluster(crash_df, method='MeanShift', verbose=0):
    '''
    Takes crash dataframe with temporal features added as input
    Function to perform clustering of time windows and assign labels back to crash dataframe. 
    Output is dataframe with additional column for labels
    If verbosity is increased, information about the clusters to printed.
    '''
    group_stats = crash_df.groupby(['time_window_str', 'weekday'])
    group_stats = group_stats.agg({'latitude': [np.mean, np.std],'longitude': [np.mean, np.std, 'count']})
    # flatten out groupby object and name columns again
    group_stats = group_stats.reset_index()
    group_stats.columns = group_stats.columns.get_level_values(0)
    group_stats.columns.values[[2,3,4,5,6]] = ['latitude_mean', 'latitude_std',
                                               'longitude_mean', 'longitude_std', 'RTA_count']
    X = group_stats.loc[:,['RTA_count']]#, 'latitude_mean', 'latitude_std','longitude_mean', 'longitude_std']]
    scaler = StandardScaler()
    scale_columns = ['latitude_mean', 'latitude_std','longitude_mean', 'longitude_std']
    #X[scale_columns] = scaler.fit_transform(X[scale_columns])
    if verbose > 5:
        X1 = X.copy()
        X1['RTA_count'] = minmax_scale(X1['RTA_count'])
        plot_dendrogram(X1)
        
    if method == 'MeanShift':
        #X['RTA_count'] = minmax_scale(X['RTA_count'])
        ms_model = MeanShift().fit(X)
        labels = ms_model.labels_

    elif method == 'GMM':
        X['RTA_count'] = minmax_scale(X['RTA_count'])
        gmm = GaussianMixture(n_components=4, verbose=verbose, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
    else:
        display('Select method "MeanShift" or "GMM"')
        #return 'error'

    labels = pd.DataFrame(labels,columns=['cluster'])
    clustered_time_buckets = pd.concat([group_stats,labels], axis=1)

    if verbose > 0:
        display(clustered_time_buckets.groupby('cluster').agg({'RTA_count': ['count', np.sum]}))
    if verbose > 1:
        plot_TW_cluster(clustered_time_buckets)
    
    crash_df = crash_df.merge(clustered_time_buckets[['time_window_str', 'weekday','cluster']],
                              how='left', on=['time_window_str', 'weekday'])

    return crash_df

def plot_TW_cluster(clustered_time_buckets):
    '''
    Displays stripplot to show how different times of the week are assigned to TW clusters.
    '''
    tb_clusters = sns.FacetGrid(clustered_time_buckets,hue='cluster', height=5)
    tb_clusters.map(sns.stripplot,'weekday', 'time_window_str', s=25, order = ['00-03', '03-06', '06-09', '09-12', 
                                                                              '12-15', '15-18', '18-21', '21-24'])
    
def assign_TW_cluster(weekday, time_window, holiday=0, strategy='baseline'):
    '''
    Can be used in a lambda function to return the time window cluster for a given day and time window.
    e.g. crash_df["cluster"] = crash_df.apply(lambda x: return_TW_cluster(x.weekday, x.time_window_str) ,axis=1)
    This is called by the function: create_cluster_feature. 
    '''
    # baseline returns a single value for all time windows so there will only be a single placement set
    if strategy == 'baseline':
        return 'baseline'
    
    # mean_shift_modified uses the results of the mean shift clustering
    # and applies human approach to create 3 simple clusters
    if strategy == 'mean_shift_modified':
        if weekday == 7:
            return 'off_peak'        
        elif weekday == 6:
            return 'off_peak'
        elif weekday in [0,1,2,3,4]:
            if time_window in ["06-09"]:
                return 'peak'
            elif time_window in ["09-12", "12-15", "15-18", "18-21"]:
                return 'middle'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'    
        elif weekday == 5:
            if time_window in ["06-09", "12-15", "15-18", "18-21"]:
                return 'middle'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
            elif time_window in ["09-12"]:
                return 'peak'
    
    # saturday_2 adds an additional cluster for middle of the day saturday 
    elif strategy == 'saturday_2':
        if weekday == 7:
            return 'off_peak'        
        elif weekday == 6:
            return 'off_peak'
        elif weekday in [0,1,2,3,4]:
            if time_window in ["06-09"]:
                return 'peak'
            elif time_window in ["09-12", "12-15", "15-18", "18-21"]:
                return 'middle'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'    
        elif weekday == 5:
            if time_window in ["06-09", "12-15", "15-18", "18-21"]:
                return 'saturday_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
            elif time_window in ["09-12"]:
                return 'saturday_busy'    
    
    # holiday_6 builds on saturday_2 and adds a new 'day' to the week for holidays
    # and a separate cluster for sundays. Total of 6 clusters
    elif strategy == 'holiday_6':
        if weekday == 7:
            return 'holiday'        
        elif weekday == 6:
            return 'sunday'
        elif weekday in [0,1,2,3,4]:
            if time_window in ["06-09"]:
                return 'peak'
            elif time_window in ["09-12", "12-15", "15-18", "18-21"]:
                return 'middle'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'    
        elif weekday == 5:
            if time_window in ["06-09", "12-15", "15-18", "18-21"]:
                return 'saturday_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
            elif time_window in ["09-12"]:
                return 'saturday_busy'      

    # has holidays but uses off peak for holidays and sundays
    elif strategy == 'holiday_simple':
        if weekday == 7:
            return 'off_peak_day'        
        elif weekday == 6:
            return 'off_peak_day'
        elif weekday in [0,1,2,3,4]:
            if time_window in ["06-09"]:
                return 'peak'
            elif time_window in ["09-12", "12-15", "15-18", "18-21"]:
                return 'middle'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'    
        elif weekday == 5:
            if time_window in ["06-09", "12-15", "15-18", "18-21"]:
                return 'saturday_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
            elif time_window in ["09-12"]:
                return 'saturday_busy'      
        # has holidays but uses off peak for holidays and sundays

    elif strategy == 'off_peak_split':
        if weekday == 7:
            if time_window in ["06-09", "09-12", "12-15", "15-18", "18-21"]:
                return 'sunday_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
        elif weekday == 6:
            if time_window in ["06-09", "09-12", "12-15", "15-18", "18-21"]:
                return 'sunday_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
        elif weekday in [0,1,2,3,4]:
            if time_window in ["06-09"]:
                return 'peak'
            elif time_window in ["09-12", "12-15", "15-18", "18-21"]:
                return 'middle'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'    
        elif weekday == 5:
            if time_window in ["06-09", "12-15", "15-18", "18-21"]:
                return 'saturday_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
            elif time_window in ["09-12"]:
                return 'saturday_busy'      

    # Weekend_day is an attempt to improve based on the loss scores of the model
    elif strategy == 'weekend_day':
        if weekday == 7:
            if time_window in ["06-09", "09-12", "12-15", "15-18", "18-21"]:
                return 'weekend_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
        elif weekday == 6:
            if time_window in ["06-09", "09-12", "12-15", "15-18", "18-21"]:
                return 'weekend_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
        elif weekday in [0,1,2,3,4]:
            if time_window in ["06-09"]:
                return 'peak'
            elif time_window in ["09-12", "12-15", "15-18", "18-21"]:
                return 'busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'    
        elif weekday == 5:
            if time_window in ["06-09", "12-15", "15-18", "18-21"]:
                return 'weekend_busy'
            elif time_window in ["00-03", "03-06", "21-24"]:
                return 'off_peak'
            elif time_window in ["09-12"]:
                return 'weekend_busy'     
            
    # no_cluster returns a individual cluster name for each weekday, time window and holiday combination
    elif strategy == 'no_cluster':
        return (str(weekday)+str(time_window)+str(holiday))
    
    elif strategy == 'time_window':
        return time_window
            
def create_cluster_feature(crash_df, strategy='baseline', verbose=0):
    '''
    Function takes crash df and creates new column with tw cluster labels.
    If verbose is increased, the time window clusters will be visualised.
    '''
    crash_df['cluster'] = crash_df.apply(lambda x: assign_TW_cluster(weekday=x.weekday, time_window=x.time_window_str, strategy=strategy) ,axis=1)
    if verbose > 0:    
        print(f'{crash_df.cluster.nunique()} clusters created')
    if verbose > 1:
        tb_clusters = sns.FacetGrid(crash_df,hue='cluster', height=5)
        tb_clusters.map(sns.stripplot,'weekday', 'time_window_str', s=20, 
                                       order = ['00-03', '03-06', '06-09', '09-12', 
                                                '12-15', '15-18', '18-21', '21-24'],
                                    label = 'Time Window Clusters')
    return crash_df
            

def create_cluster_centroids(crash_df_with_cluster, test_df, verbose=0, method='k_means', lr=3e-2, n_epochs=400, batch_size=50):
    if method == 'k_means':
        centroids_dict = create_k_means_centroids(crash_df_with_cluster, verbose=verbose)
    elif method == 'agglomerative':
        centroids_dict = create_AgglomerativeClustering_centroids(crash_df_with_cluster, verbose=verbose)
    elif method == 'gradient_descent':
        centroids_dict = create_gradient_descent_centroids(crash_df_with_cluster, test_df, verbose=verbose,
                                                           lr=lr, n_epochs=n_epochs, batch_size=batch_size)
    elif method == 'k_medoids':
        centroids_dict = create_k_medoids_centroids(crash_df_with_cluster, verbose=verbose)
    
    elif method == 'baseline':
        centroids_dict = create_baseline_centroids(crash_df_with_cluster, verbose=verbose)
    
    if verbose > 0:    
        print(f'{len(centroids_dict)} placement sets created')
    
    return centroids_dict
    

def create_baseline_centroids(crash_df_with_cluster, verbose=0):
    if verbose > 0:    
        print('using star grid for placement')
    centroids_dict = {}
    for i in crash_df_with_cluster.cluster.unique():
        data_slice = crash_df_with_cluster.query('cluster==@i')
        lat_centroid = list(data_slice.latitude.quantile(q=[1/5,2/5,3/5,4/5]))
        lon_centroid = list(data_slice.longitude.quantile(q=[1/4,2/4,3/4]))
        centroids=[(lat_centroid[1],lon_centroid[0]),(lat_centroid[2],lon_centroid[0]),
               (lat_centroid[0],lon_centroid[1]),(lat_centroid[3],lon_centroid[1]),
               (lat_centroid[1],lon_centroid[2]),(lat_centroid[2],lon_centroid[2])]
        centroids_dict[i] = np.array(centroids).flatten()          
        if verbose > 5:
            print(centroids)
    return centroids_dict

    
def create_k_means_centroids(crash_df_with_cluster, verbose=0):
    if verbose > 0:    
        print('using k-means clustering')
    centroids_dict = {}
    for i in crash_df_with_cluster.cluster.unique():
        data_slice = crash_df_with_cluster.query('cluster==@i')
        kmeans = KMeans(n_clusters=6, verbose=0, tol=1e-5, max_iter=500, n_init=20 ,random_state=42)
        kmeans.fit(data_slice[['latitude','longitude']])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        centroids_dict[i] = centroids.flatten()        
        if verbose > 2:
            plot_centroids(data_slice, centroids, cluster=i, labels=labels)
        if verbose > 5:
            print(centroids)
    return centroids_dict

def create_k_medoids_centroids(crash_df_with_cluster, verbose=0):
    if verbose > 0:    
        print('using k-medoids clustering')
    centroids_dict = {}
    for i in crash_df_with_cluster.cluster.unique():
        data_slice = crash_df_with_cluster.query('cluster==@i')
        kmedoids = KMedoids(n_clusters=6, init='k-medoids++', max_iter=500, random_state=42)
        kmedoids.fit(data_slice[['latitude','longitude']])
        labels = kmedoids.labels_
        centroids = kmedoids.cluster_centers_
        centroids_dict[i] = centroids.flatten()        
        if verbose > 2:
            plot_centroids(data_slice, centroids, cluster=i, labels=labels)
        if verbose > 5:
            print(centroids)
    return centroids_dict


def create_AgglomerativeClustering_centroids(crash_df_with_cluster, verbose=0):
    if verbose > 0:    
        print('using agglomerative clustering')
    centroids_dict = {}   
    for i in crash_df_with_cluster.cluster.unique():
        data_slice = crash_df_with_cluster.query('cluster==@i')
        
        hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
        y_predict = hc.fit_predict(data_slice[['latitude','longitude']])
        clf = NearestCentroid()
        clf.fit(data_slice[['latitude','longitude']], y_predict)
        
        centroids = clf.centroids_
        centroids_dict[i] = centroids.flatten()
        if verbose > 5:
            plot_centroids(data_slice, centroids, cluster=i)
        if verbose > 14:
            print(centroids)
    return centroids_dict

def create_gradient_descent_centroids(crash_df_with_cluster, test_df, verbose=0, lr=3e-3, n_epochs=400, batch_size=50):
    if verbose > 0:    
        print('using gradient descent clustering')
    
    # Copy dataframes and standardize lat lon values on train and test set
    scaler = StandardScaler()
    crash_df_scaled = crash_df_with_cluster.copy()
    test_df_scaled = test_df.copy()
    crash_df_scaled[['latitude', 'longitude']] = scaler.fit_transform(crash_df_scaled[['latitude', 'longitude']])
    test_df_scaled[['latitude', 'longitude']] = scaler.transform(test_df_scaled[['latitude', 'longitude']])
    
    centroids_dict = {}   
    for i in crash_df_scaled.cluster.unique():
        data_slice = crash_df_scaled.query('cluster==@i')
        test_slice = test_df_scaled.query('cluster==@i')
        train_locs = tensor(data_slice[['latitude', 'longitude']].values) # To Tensor
        val_locs = tensor(test_slice[['latitude', 'longitude']].values) # To Tensor
        
        # Load crash locs from train into a dataloader
        batches = DataLoader(train_locs, batch_size=batch_size, shuffle=True)

        # Set up ambulance locations
        amb_locs = torch.randn(6, 2)*.5
        amb_locs = amb_locs + tensor(data_slice.latitude.mean(), data_slice.longitude.mean())
        amb_locs.requires_grad_()
                
        # Set vars
        lr=lr
        n_epochs = n_epochs

        # Store loss over time
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(n_epochs):
           # Run through batches
            for crashes in batches:
                loss = loss_fn(crashes, amb_locs) # Find loss for this batch of crashes
                loss.backward() # Calc grads
                amb_locs.data -= lr * amb_locs.grad.data # Update locs
                amb_locs.grad = None # Reset gradients for next step
                train_losses.append(loss.item())                
                if verbose > 2:
                    val_loss = loss_fn(val_locs, amb_locs)
                    val_losses.append(val_loss.item()) # Can remove as this slows things down
            if verbose > 2 and epoch % 100  == 0: # show progress
                print(f'Val loss for cluster {i}: {val_loss.item()}')
        
        centroids = amb_locs.detach().numpy()

        #show output
        if verbose > 5:
            plot_centroids(data_slice, centroids, cluster=i)
        if verbose > 14:
            print(centroids) 
        if verbose > 9:
            plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(train_losses, label='train_loss')
            plt.plot(val_losses, c='red', label='val loss')
            plt.legend()
        
        #scale back to lat lon
        centroids = scaler.inverse_transform(centroids)
        centroids_dict[i] = centroids.flatten()
        
    return centroids_dict

def loss_fn(crash_locs, amb_locs):
    """
      Used for gradient descent model. 
      For each crash we find the dist to the closest ambulance, and return the mean of these dists.
    """
    # Dists to first ambulance
    dists_split = crash_locs - amb_locs[0]
    dists = (dists_split[:,0]**2 + dists_split[:,1]**2)**0.5
    min_dists = dists
    for i in range(1, 6):
        # Update dists so they represent the dist to the closest ambulance
        dists_split = crash_locs-amb_locs[i]
        dists = (dists_split[:,0]**2 + dists_split[:,1]**2)**0.5
        min_dists = torch.min(min_dists, dists)
    return min_dists.mean()


def centroid_to_submission(centroids_dict, date_start='2019-07-01', date_end='2020-01-01', tw_cluster_strategy='baseline', verbose=0):
    '''Takes dictionary of clusters and centroids and creates a data frame in the format needed for submission'''

    # Create Date range covering submission period set
    dates = pd.date_range(date_start, date_end, freq='3h')
    submission_df = pd.DataFrame({'date':dates})
    submission_df = create_temporal_features(submission_df,'date')
    submission_df["cluster"] = submission_df.apply(lambda x: assign_TW_cluster(x.weekday, x.time_window_str, strategy=tw_cluster_strategy) ,axis=1)
    ambulance_columns = ['A0_Latitude', 'A0_Longitude', 'A1_Latitude','A1_Longitude', 'A2_Latitude', 'A2_Longitude', 
                         'A3_Latitude', 'A3_Longitude', 'A4_Latitude', 'A4_Longitude', 'A5_Latitude', 'A5_Longitude']
    for i in submission_df["cluster"].unique():
        submission_df["placements"] = submission_df["cluster"].apply(lambda x: centroids_dict.get(x))
        submission_df[ambulance_columns] = pd.DataFrame(submission_df.placements.tolist(), index=submission_df.index)
    submission_df = submission_df.drop('placements', axis=1)
    submission_df = drop_temporal(submission_df)
    submission_df = submission_df.drop(["cluster"], axis=1)
    if verbose > 0:
        print('submission dataframe created')
    return submission_df

def create_submission_csv(submission_df, crash_source, outlier_filter, tw_cluster_strategy, placement_method, path='../Outputs/', verbose=0):
    '''Takes dataframe in submission format and outputs a csv file with matching name'''
    # current_time = dt.datetime.now()
    current_time = dt.datetime.now()
    filename = f'{current_time.year}{current_time.month}{current_time.day}_{crash_source}_{outlier_filter}_{tw_cluster_strategy}_{placement_method}.csv'
    submission_df.to_csv(path+filename,index=False)
    if verbose > 0:
        print(f'{filename} saved in {path}') 
    
    
def score(train_placements_df, crash_df, test_start_date='2018-01-01', test_end_date='2019-12-31', verbose=0):
          
    '''
    Can be used to score the ambulance placements against a set of crashes. 
    Can be used on all crash data, train_df or holdout_df as crash_df.
    '''
    test_df = crash_df.loc[(crash_df.datetime >= test_start_date) & (crash_df.datetime <= test_end_date)]
    if verbose > 0:    
        print(f'Data points in test period: {test_df.shape[0]}' )
    total_distance = 0
    for crash_date, c_lat, c_lon in test_df[['datetime', 'latitude', 'longitude']].values:
        row = train_placements_df.loc[train_placements_df.date < crash_date].tail(1)
        dists = []
        for a in range(6):
            dist = ((c_lat - row[f'A{a}_Latitude'].values[0])**2+(c_lon - row[f'A{a}_Longitude'].values[0])**2)**0.5 
            dists.append(dist)
        total_distance += min(dists)
    return total_distance



def ambulance_placement_pipeline(input_path='../Inputs/', output_path='../Outputs/', crash_source_csv='Train',
                                 outlier_filter=0, holdout_strategy='random', holdout_test_size=0.3,
                                 test_period_date_start='2019-01-01', test_period_date_end='2019-07-01',
                                 tw_cluster_strategy='saturday_2', placement_method='k_means', verbose=0,
                                 lr=3e-2, n_epochs=400, batch_size=50):  
    '''
    load crash data (from train or prediction) and apply feautre engineering, run tw clustering (based on strategy choice) 
    create ambulance placements, create output file.
    '''
    # load crash data into dataframe
    crash_df = create_crash_df(train_file = input_path+crash_source_csv+'.csv')
    # in case of loading file with hex bins instead of lat/long
    if 'latitude' not in crash_df.columns:
        crash_df['latitude'] = crash_df.hex_bins.apply(lambda x : h3.h3_to_geo(x)[0])
        crash_df['longitude'] = crash_df.hex_bins.apply(lambda x : h3.h3_to_geo(x)[1])  
        crash_df.drop("hex_bins", axis=1, inplace=True)
    # create individual date and time features from date column
    crash_df = create_temporal_features(crash_df)
    # split data into train and test sets
    train_df, test_df = split_accident_df(data=crash_df, strategy=holdout_strategy,
                                          test_size=holdout_test_size)
    
    # remove outliers from test set based on lat and lon
    train_df = outlier_removal(train_df, filter=outlier_filter)
    # apply time window cluster labels to df based on strategy specified
    train_df = create_cluster_feature(train_df, strategy=tw_cluster_strategy, verbose=verbose)
    # Run clustering model to get placement set centroids for each TW cluster
    test_df_with_clusters = create_cluster_feature(test_df, strategy=tw_cluster_strategy, verbose=0)
    centroids_dict = create_cluster_centroids(train_df, test_df=test_df_with_clusters, verbose=verbose, method=placement_method,
                                             lr=lr, n_epochs=n_epochs, batch_size=batch_size)
    
    # create df in format needed for submission
    train_placements_df = centroid_to_submission(centroids_dict, date_start='2018-01-01', date_end='2019-12-31',
                                                 tw_cluster_strategy=tw_cluster_strategy)
    
    # Run scoring functions
    if verbose > 0:    
        print(f'Total size of test set: {test_df.shape[0]}')
    test_score = score(train_placements_df, test_df, test_start_date=test_period_date_start,
                       test_end_date=test_period_date_end)
    if verbose > 0:    
        print(f'Total size of train set: {crash_df.shape[0]}')
    train_score = score(train_placements_df,train_df,
                        test_start_date=test_period_date_start, test_end_date=test_period_date_end)
    if verbose > 0:    
        print(f'Score on test set: {test_score / max(test_df.shape[0],1)}')
    if verbose > 0:    
        print(f'Score on train set: {train_score / train_df.shape[0] } (avg distance per accident)')

    # Create file for submitting to zindi
    submission_df = centroid_to_submission(centroids_dict, date_start='2019-07-01', date_end='2020-01-01',
                                           tw_cluster_strategy=tw_cluster_strategy)
    create_submission_csv(submission_df, crash_source=crash_source_csv, outlier_filter=outlier_filter,
                          tw_cluster_strategy=tw_cluster_strategy, placement_method=placement_method, path=output_path ,verbose=verbose)

    
    
    
### Prediction functions from here on

def convert_h3_to_lat_lon(df):
    """
    Convert hex bins back to latitude and longitude
    """
    df['latitude'] = df.hex_bins.apply(lambda x: h3.h3_to_geo(x)[0])
    df['longitude'] = df.hex_bins.apply(lambda x: h3.h3_to_geo(x)[1])  
    df = df.drop("hex_bins", axis=1)
    return df

def create_pred_template(df):
    '''Based on hex bin resolution creates an empty data frame for each 3 hour time window for each hex bin.
     This results in a n * 2 dataframe (columns: time_windows, hex_bins) where number of rows equals hex_bins * 4369.
     4369 is the result of days between start and end date (in days) * 8 time windows per day (24 / 3 hours)'''
    #Create dataframe to get the accurate amount of 3-hour time windows for the desired time frame
    date_start = '2018-01-01'
    date_end = '2019-07-01'
    dates = pd.date_range(date_start, date_end, freq='3h')
    all_days_df = pd.DataFrame(dates, columns=["dates"])

    time_windows = list(all_days_df["dates"])
    len_windows = all_days_df.shape[0]
    list_unique_hexbins = list(df["h3_zone_6"].unique())
    
    list_bins_per_window = []
    list_time_windows = []
    
    for i in range(0, len(list_unique_hexbins)):
        list_bins_per_window += len_windows * [list_unique_hexbins[i]]
        list_time_windows += time_windows
        
    input_df = {"time_windows": list_time_windows, "hex_bins": list_bins_per_window}
    df_pred_template = pd.DataFrame(data=input_df)
    
    return df_pred_template

def rta_per_time_window_hex_bin(df):
    '''
    Add up RTA's per time window and hex bin
    '''
    df["time_window_key"] = df["datetime"].apply(lambda x: str(x.year) + "-" + str(x.month) + "-" + str(x.day) + "-" + str(math.floor(x.hour / 3)))
    df_tw_hex = df.groupby(["time_window_key", "h3_zone_6"]).agg({"uid": "count"}).reset_index()
    col_names = ["time_window_key"] + ["hex_bins"] + ["RTA"]
    df_tw_hex.columns = col_names
    return df_tw_hex

def fill_overall_df(df_pred_template, df_tw_hex):
    '''
    Join road traffic accidents onto empty data frame that consists of time windows (8 per day) for all days (1.5 years) for all hex bins. 
    For combinations with no accidents, NaNs will be converted into 0.
    '''
    df_pred_template["time_window_key"] = df_pred_template["time_windows"].apply(lambda x: str(x.year) + "-" + str(x.month) + "-" + str(x.day) + "-" + str(math.floor(x.hour / 3)))
    df_merged = pd.merge(df_pred_template, df_tw_hex, on=["time_window_key", "hex_bins"], how="outer")
    df_merged = df_merged.fillna(0)
    
    list_of_c = list(df_merged.columns)
    list_of_c[0] = "datetime"
    df_merged.columns = list_of_c
    return df_merged

def generate_outlier_list(df, frequency_cutoff=1):
    """
    Based on the minimum frequency of occurrence, cut off all hex bins that do not exceed that value over 1.5 years. Returns list of hex bins to exclude.
    """
    if frequency_cutoff == 0:
        return []
    else:
        df_outliers = df.groupby("hex_bins")
        df_outliers = df_outliers.agg({'RTA': np.count_nonzero})
        df_outliers = df_outliers.reset_index()
        df_outliers.columns = ["hex_bins", "RTA_nonzero"]
        df_freq_outliers = df_outliers.loc[df_outliers["RTA_nonzero"] <= frequency_cutoff]

        # Get list of frequency outliers
        list_freq_outliers = list(df_freq_outliers["hex_bins"].values)

        return list_freq_outliers

def filter_df_for_pred_a(df, list_freq_outliers):
    """
    Exclude frequency outliers according to list and drop all hex bin / time window combinations with 0 RTA's
    """
    df_pred_a = df.loc[~df["h3_zone_6"].isin(list_freq_outliers)]
    df_pred_a = df_pred_a.reset_index()
    df_pred_a = df_pred_a.drop(["uid", "latitude", "longitude", "time", "time_window", "time_window_str", "day", "weekday", "month", "half_year", "rainy_season",
                                "year", "date_trunc", "holiday", "time_window_key", "index"], axis=1)
    df_pred_a.columns = ["datetime", "hex_bins"]
    return df_pred_a

def filter_df_for_pred_b(df_merged, list_freq_outliers):
    """
    Exclude frequency outliers according to list and drop all hex bin / time window combinations with 0 RTA's
    """
    
    # Filters overall dataframe to exclude hex bins with only one RTA occurrence in the whole timeframe (according to input list)
    df_merged = df_merged.loc[~df_merged["hex_bins"].isin(list_freq_outliers)]
    
    # Also filters out all hex bin and time window combinations where no RTA occurred
    df_pred_b = df_merged.loc[df_merged["RTA"] > 0]
    
    return df_pred_b

def clean_pred_b(df_pred_b):
    """Dropping all redundant rows, fixing indices and making sure the time windows are hit."""

    # Remove some redundant rows and fix indices
    df_predictions = df_pred_b.drop(["time_window_key", "RTA"], axis=1)
    df_predictions = df_predictions.reset_index()
    df_predictions.drop("index", axis=1, inplace=True)
    
    # Add 1 minute to have the RTA's lie inside the time window rather than on the verge, sort values and reset the index
    df_predictions["datetime"] = df_predictions["datetime"].apply(lambda x: x + pd.Timedelta(minutes=1))
    df_predictions = df_predictions.sort_values(by="datetime").reset_index()
    
    # Drop redundant columns
    df_predictions = df_predictions.drop("index", axis=1)
    
    return df_predictions

def create_samples(df, list_freq_outliers):
    """
    Creates a sort of distribution from which hex bin and time window combinations can be drawn, subject to the predicted RTA's per day
    """
    
    dict_windows = {1: "00-03", 2: "03-06", 3: "06-09", 4: "09-12", 5: "12-15", 
                    6: "15-18", 7: "18-21", 8: "21-24"}

    df["time_window"] = df["datetime"].apply(lambda x: math.floor(x.hour / 3) + 1)
    df["time_window_str"] = df["time_window"].apply(lambda x: dict_windows.get(x))
    df["weekday"] = df["datetime"].apply(lambda x: x.weekday())

    # Filtering for hex bins with only one occurrence
    df_filter = df.loc[~df["hex_bins"].isin(list_freq_outliers)]
    
    # Prepare data frame for sample generation
    df_freq = df_filter.groupby(["hex_bins", "weekday", "time_window_str"])
    df_samples = df_freq.agg({'RTA': [np.count_nonzero]})
    df_samples = df_samples.reset_index()
    df_samples.columns = ["hex_bins", "weekday", "time_window", "RTA_freq"]
    
    return df_samples

def generate_predictions(df, predicted_rta):
    """
    Takes a dataframe containing the RTA frequency per weekday and time window and the predicted RTA's per day and turns this into a prediction dataframe.
    """

    df_monday = df.loc[df["weekday"] == 0].sort_values(by="RTA_freq", ascending=False)
    df_tuesday = df.loc[df["weekday"] == 1].sort_values(by="RTA_freq", ascending=False)
    df_wednesday = df.loc[df["weekday"] == 2].sort_values(by="RTA_freq", ascending=False)
    df_thursday = df.loc[df["weekday"] == 3].sort_values(by="RTA_freq", ascending=False)
    df_friday = df.loc[df["weekday"] == 4].sort_values(by="RTA_freq", ascending=False)
    df_saturday = df.loc[df["weekday"] == 5].sort_values(by="RTA_freq", ascending=False)
    df_sunday = df.loc[df["weekday"] == 6].sort_values(by="RTA_freq", ascending=False)
    
    # Split overall predictions into predictions per weekday
    lst_mon = predicted_rta[0::7]
    lst_tue = predicted_rta[1::7]
    lst_wed = predicted_rta[2::7]
    lst_thu = predicted_rta[3::7]
    lst_fri = predicted_rta[4::7]
    lst_sat = predicted_rta[5::7]
    lst_sun = predicted_rta[6::7]
    
    # The evaluation period 2019-07-01 to 2019-12-31 conveniently starts with a Monday but end with a Tuesday - hence the loop has to run 
    # one iteration more for Monday and Tuesday.
    # This generates a list of lists of predictions for each weekday

    monday_bins = tuesday_bins = wednesday_bins = thursday_bins = friday_bins = saturday_bins = sunday_bins = []
    monday_tw = tuesday_tw = wednesday_tw = thursday_tw = friday_tw = saturday_tw = sunday_tw = []
    
    for i in range(len(lst_mon)):
        monday_bins.append(list(*[df_monday["hex_bins"][0:lst_mon[i]]]))
        monday_tw.append(list(*[df_monday["time_window"][0:lst_mon[i]]]))
        tuesday_bins.append(list(*[df_tuesday["hex_bins"][0:lst_tue[i]]]))
        tuesday_tw.append(list(*[df_tuesday["time_window"][0:lst_tue[i]]]))
    for i in range(len(lst_wed)):
        wednesday_bins.append(list(*[df_wednesday["hex_bins"][0:lst_wed[i]]]))
        wednesday_tw.append(list(*[df_wednesday["time_window"][0:lst_wed[i]]]))
        thursday_bins.append(list(*[df_thursday["hex_bins"][0:lst_thu[i]]]))
        thursday_tw.append(list(*[df_thursday["time_window"][0:lst_thu[i]]]))
        friday_bins.append(list(*[df_friday["hex_bins"][0:lst_fri[i]]]))
        friday_tw.append(list(*[df_friday["time_window"][0:lst_fri[i]]]))
        saturday_bins.append(list(*[df_saturday["hex_bins"][0:lst_sat[i]]]))
        saturday_tw.append(list(*[df_saturday["time_window"][0:lst_sat[i]]]))
        sunday_bins.append(list(*[df_sunday["hex_bins"][0:lst_sun[i]]]))
        sunday_tw.append(list(*[df_sunday["time_window"][0:lst_sun[i]]]))    
    
    
    # Turn list of lists into an overall list for each weekday's predictions
    flat_monday_bins = [item for sublist in monday_bins for item in sublist]
    flat_monday_tw = [item for sublist in monday_tw for item in sublist]
    flat_tuesday_bins = [item for sublist in tuesday_bins for item in sublist]
    flat_tuesday_tw = [item for sublist in tuesday_tw for item in sublist]
    flat_wednesday_bins = [item for sublist in wednesday_bins for item in sublist]
    flat_wednesday_tw = [item for sublist in wednesday_tw for item in sublist]
    flat_thursday_bins = [item for sublist in thursday_bins for item in sublist]
    flat_thursday_tw = [item for sublist in thursday_tw for item in sublist]
    flat_friday_bins = [item for sublist in friday_bins for item in sublist]
    flat_friday_tw = [item for sublist in friday_tw for item in sublist]
    flat_saturday_bins = [item for sublist in saturday_bins for item in sublist]
    flat_saturday_tw = [item for sublist in saturday_tw for item in sublist]
    flat_sunday_bins = [item for sublist in sunday_bins for item in sublist]
    flat_sunday_tw = [item for sublist in sunday_tw for item in sublist]
    
    # Generate list with hex bins and time windows as input for prediction
    flat_bins = flat_monday_bins + flat_tuesday_bins + flat_wednesday_bins + flat_thursday_bins + flat_friday_bins + flat_saturday_bins + flat_sunday_bins
    flat_tw = flat_monday_tw + flat_tuesday_tw + flat_wednesday_tw + flat_thursday_tw + flat_friday_tw + flat_saturday_tw + flat_sunday_tw

    # Generate list with day of the week entries for each prediction as input for dataframe
    weekdays = [0] * sum(lst_mon) + [1] * sum(lst_tue) + [2] * sum(lst_wed) + [3] * sum(lst_thu) + [4] * sum(lst_fri) + [5] * sum(lst_sat) + [6] * sum(lst_sun)
    
    # Generate list with week entries for each prediction as input for dataframe
    list_of_days_list = [lst_mon, lst_tue, lst_wed, lst_thu, lst_fri, lst_sat, lst_sun]
    lst_weeks = []
    for lst_days in list_of_days_list:
        i = 0
        for number in lst_days:
            lst_weeks += [i] * number
            i += 1
    
    # Create dataframe
    df_pred_c = pd.DataFrame(list(zip(flat_bins, flat_tw, weekdays, lst_weeks)), columns=["hex_bins", "time_window", "weekday", "week"])
    
    return df_pred_c

def generate_predictions_first_half_2019(df, predicted_rta):
    """
    Takes a dataframe containing the RTA frequency per weekday and time window and the predicted RTA's per day and turns this into a prediction dataframe.
    """

    df_monday = df.loc[df["weekday"] == 0].sort_values(by="RTA_freq", ascending=False)
    df_tuesday = df.loc[df["weekday"] == 1].sort_values(by="RTA_freq", ascending=False)
    df_wednesday = df.loc[df["weekday"] == 2].sort_values(by="RTA_freq", ascending=False)
    df_thursday = df.loc[df["weekday"] == 3].sort_values(by="RTA_freq", ascending=False)
    df_friday = df.loc[df["weekday"] == 4].sort_values(by="RTA_freq", ascending=False)
    df_saturday = df.loc[df["weekday"] == 5].sort_values(by="RTA_freq", ascending=False)
    df_sunday = df.loc[df["weekday"] == 6].sort_values(by="RTA_freq", ascending=False)
    
    # Split overall predictions into predictions per weekday
    lst_mon = predicted_rta[6::7]
    lst_tue = predicted_rta[0::7]
    lst_wed = predicted_rta[1::7]
    lst_thu = predicted_rta[2::7]
    lst_fri = predicted_rta[3::7]
    lst_sat = predicted_rta[4::7]
    lst_sun = predicted_rta[5::7]
    
    # The evaluation period 2019-07-01 to 2019-12-31 conveniently starts with a Monday but end with a Tuesday - hence the loop has to run 
    # one iteration more for Monday and Tuesday.
    # This generates a list of lists of predictions for each weekday

    monday_bins = tuesday_bins = wednesday_bins = thursday_bins = friday_bins = saturday_bins = sunday_bins = []
    monday_tw = tuesday_tw = wednesday_tw = thursday_tw = friday_tw = saturday_tw = sunday_tw = []
    
    for i in range(len(lst_mon)):
        monday_bins.append(list(*[df_monday["hex_bins"][0:lst_mon[i]]]))
        monday_tw.append(list(*[df_monday["time_window"][0:lst_mon[i]]]))

    for i in range(len(lst_wed)):
        tuesday_bins.append(list(*[df_tuesday["hex_bins"][0:lst_tue[i]]]))
        tuesday_tw.append(list(*[df_tuesday["time_window"][0:lst_tue[i]]]))
        wednesday_bins.append(list(*[df_wednesday["hex_bins"][0:lst_wed[i]]]))
        wednesday_tw.append(list(*[df_wednesday["time_window"][0:lst_wed[i]]]))
        thursday_bins.append(list(*[df_thursday["hex_bins"][0:lst_thu[i]]]))
        thursday_tw.append(list(*[df_thursday["time_window"][0:lst_thu[i]]]))
        friday_bins.append(list(*[df_friday["hex_bins"][0:lst_fri[i]]]))
        friday_tw.append(list(*[df_friday["time_window"][0:lst_fri[i]]]))
        saturday_bins.append(list(*[df_saturday["hex_bins"][0:lst_sat[i]]]))
        saturday_tw.append(list(*[df_saturday["time_window"][0:lst_sat[i]]]))
        sunday_bins.append(list(*[df_sunday["hex_bins"][0:lst_sun[i]]]))
        sunday_tw.append(list(*[df_sunday["time_window"][0:lst_sun[i]]]))    
    
    
    # Turn list of lists into an overall list for each weekday's predictions
    flat_monday_bins = [item for sublist in monday_bins for item in sublist]
    flat_monday_tw = [item for sublist in monday_tw for item in sublist]
    flat_tuesday_bins = [item for sublist in tuesday_bins for item in sublist]
    flat_tuesday_tw = [item for sublist in tuesday_tw for item in sublist]
    flat_wednesday_bins = [item for sublist in wednesday_bins for item in sublist]
    flat_wednesday_tw = [item for sublist in wednesday_tw for item in sublist]
    flat_thursday_bins = [item for sublist in thursday_bins for item in sublist]
    flat_thursday_tw = [item for sublist in thursday_tw for item in sublist]
    flat_friday_bins = [item for sublist in friday_bins for item in sublist]
    flat_friday_tw = [item for sublist in friday_tw for item in sublist]
    flat_saturday_bins = [item for sublist in saturday_bins for item in sublist]
    flat_saturday_tw = [item for sublist in saturday_tw for item in sublist]
    flat_sunday_bins = [item for sublist in sunday_bins for item in sublist]
    flat_sunday_tw = [item for sublist in sunday_tw for item in sublist]
    
    # Generate list with hex bins and time windows as input for prediction
    flat_bins = flat_monday_bins + flat_tuesday_bins + flat_wednesday_bins + flat_thursday_bins + flat_friday_bins + flat_saturday_bins + flat_sunday_bins
    flat_tw = flat_monday_tw + flat_tuesday_tw + flat_wednesday_tw + flat_thursday_tw + flat_friday_tw + flat_saturday_tw + flat_sunday_tw

    # Generate list with day of the week entries for each prediction as input for dataframe
    weekdays = [0] * sum(lst_mon) + [1] * sum(lst_tue) + [2] * sum(lst_wed) + [3] * sum(lst_thu) + [4] * sum(lst_fri) + [5] * sum(lst_sat) + [6] * sum(lst_sun)
    
    # Generate list with week entries for each prediction as input for dataframe
    list_of_days_list = [lst_mon, lst_tue, lst_wed, lst_thu, lst_fri, lst_sat, lst_sun]
    lst_weeks = []
    for lst_days in list_of_days_list:
        i = 0
        for number in lst_days:
            lst_weeks += [i] * number
            i += 1
    
    # Create dataframe
    df_pred_c = pd.DataFrame(list(zip(flat_bins, flat_tw, weekdays, lst_weeks)), columns=["hex_bins", "time_window", "weekday", "week"])
    
    return df_pred_c

def reduce_to_time_windows(df, predict_period):
    """
    Takes a data frame of predicted RTA's and brings it into the correct format for clustering.
    """
    
    # Set start of prediction period
    if predict_period == '2019_h2':
        start = pd.to_datetime("2019-07-01")
    if predict_period == '2019_h1':
        start = pd.to_datetime("2019-01-01")
    
    # Creates a datetime column that counts the days upwards and then sets all entries to the starting day, 2019-07-01, plus that day
    df["help"] = (df["week"]) * 7 + df["weekday"]
    df["datetime"] = df["help"].apply(lambda x: start + pd.Timedelta(days=x))
    
    # Convert time windows strings back to datetime objects and add 1 minute to have them lie inside the time window rather than on the verge
    df.loc[df["time_window"] == "00-03", "datetime"] = df["datetime"] + pd.Timedelta(minutes=1)
    df.loc[df["time_window"] == "03-06", "datetime"] = df["datetime"] + pd.Timedelta(hours=3, minutes=1)
    df.loc[df["time_window"] == "06-09", "datetime"] = df["datetime"] + pd.Timedelta(hours=6, minutes=1)
    df.loc[df["time_window"] == "09-12", "datetime"] = df["datetime"] + pd.Timedelta(hours=9, minutes=1)
    df.loc[df["time_window"] == "12-15", "datetime"] = df["datetime"] + pd.Timedelta(hours=12, minutes=1)
    df.loc[df["time_window"] == "15-18", "datetime"] = df["datetime"] + pd.Timedelta(hours=15, minutes=1)
    df.loc[df["time_window"] == "18-21", "datetime"] = df["datetime"] + pd.Timedelta(hours=18, minutes=1)
    df.loc[df["time_window"] == "21-24", "datetime"] = df["datetime"] + pd.Timedelta(hours=21, minutes=1)
    
    # Remove redundant columns
    df_pred_c_clean = df.drop(["time_window", "weekday", "week", "help"], axis=1)
    
    df_pred_c_clean = df_pred_c_clean.sort_values(by="datetime")
    
    if predict_period=="2019_h1":
        df_pred_c_clean = df_pred_c_clean.loc[df_pred_c_clean["datetime"] < "2019-7-1"]
    
    return df_pred_c_clean


def export_df_to_csv(df,path_file='../Inputs/train_h3.csv'):
    """
    Exporting the dataframe back to csv
    """
    df.to_csv(path_file,index=False)
    print(f'file created {path_file}') 

def rta_prediction_pipeline(type_of_pred="a", frequency_cutoff=1, predict_period='2019_h1'):
    """
    Choose type of prediction:
        - 'a' for a simple frequency outlier removal based on parameter 'frequency_cutoff'. Outputs data for 2018-01-01 to 2019-06-30.
        - 'b' for only accounting for the frequency of accidents in a hex bin / time window combination, also subject to parameter 'frequency_cutoff'. 
        Outputs data for 2018-01-01 to 2019-06-30.
        - 'c' for using predicted overall daily RTA's and sampling from a list of hex bins most likely to produce an RTA for the selected time window.
        Outputs data for 2019-07-01 to 2019-12-31.
    """
    df_raw = create_crash_df()
    df = create_temporal_features(df_raw)
    df = assign_hex_bin(df)

    df_pred_template = create_pred_template(df)
    df_tw_hex = rta_per_time_window_hex_bin(df)
        
    df_merged = fill_overall_df(df_pred_template, df_tw_hex)

    list_freq_outliers = generate_outlier_list(df_merged, frequency_cutoff=frequency_cutoff)
    
    # Prediction type A
    if type_of_pred == "a":

        df_pred_a = filter_df_for_pred_a(df, list_freq_outliers)

        export_df_to_csv(df_pred_a,path_file='../Inputs/predictions_for_clustering_a.csv')
        
        return df_pred_a
    
    # Prediction type B
    if type_of_pred == "b":
        
        df_pred_b = filter_df_for_pred_b(df_merged, list_freq_outliers)

        df_pred_b_clean = clean_pred_b(df_pred_b)        
        
        export_df_to_csv(df_pred_b_clean,path_file='../Inputs/predictions_for_clustering_b.csv')
        
        return df_pred_b_clean

    # Prediction type C
    if type_of_pred == "c":
        
        df_samples = create_samples(df_merged, list_freq_outliers)
        
        df_weather = pd.read_csv('../Inputs/Weather_Nairobi_Daily_GFS.csv', parse_dates=['Date'])
        df_raw = create_crash_df()
        
        predicted_rta = predict_accidents_on_weather(df_raw, df_weather, predict_period=predict_period)
        predicted_rta_round = [int(round(i, 0)) for i in predicted_rta]
        #predicted_rta = predict_accidents_on_weather(df_raw, df_weather)
        if predict_period == '2019_h2':
            df_pred_c = generate_predictions(df_samples, predicted_rta_round)
        elif predict_period == '2019_h1':
            df_pred_c = generate_predictions_first_half_2019(df_samples, predicted_rta_round)
        
        df_pred_c_clean = reduce_to_time_windows(df_pred_c, predict_period)
        
        export_df_to_csv(df_pred_c_clean,path_file='../Inputs/predictions_for_clustering_c.csv')
        
        return df_pred_c_clean        
        
# Call pipeline function! Best results so far:
'''
ambulance_placement_pipeline(input_path='../Inputs/', output_path='../Outputs/', crash_source_csv='Train',
                             outlier_filter=0.005, 
                             holdout_strategy='random', holdout_test_size=0.005,
                             test_period_date_start='2018-01-01', test_period_date_end='2019-12-31',
                             tw_cluster_strategy='holiday_simple', placement_method='gradient_descent', verbose=0,
                             lr=3e-3, n_epochs=400)
'''


def full_pipeline(frequency_cutoff=1, predict_period='2019_h1', outlier_filter=0, 
                  test_period_date_start='2019-07-01', test_period_date_end='2019-12-01',
                  tw_cluster_strategy='baseline', placement_method='k_means', verbose=0,
                  lr=3e-2, n_epochs=400, batch_size=50):  
    '''
    load crash data (from prediction) and apply feautre engineering, 
    run tw clustering (based on strategy choice) 
    create ambulance placements, create output file.
    '''
    
    # load predicted crash data into dataframe
    crash_df = rta_prediction_pipeline(type_of_pred="c", frequency_cutoff=frequency_cutoff, predict_period=predict_period)
    # in case of loading file with hex bins instead of lat/long
    if 'latitude' not in crash_df.columns:
        crash_df['latitude'] = crash_df.hex_bins.apply(lambda x : h3.h3_to_geo(x)[0])
        crash_df['longitude'] = crash_df.hex_bins.apply(lambda x : h3.h3_to_geo(x)[1])  
        crash_df.drop("hex_bins", axis=1, inplace=True)
    
    # create train dataset from predictions for 2019
    crash_df = create_temporal_features(crash_df)
    train_df = crash_df
    # create test dataset from actual 2019
    test_df = create_crash_df(train_file = '../Inputs/Train.csv')
    test_df = test_df.loc[test_df["datetime"] >= "2019-01-01"]
    test_df = create_temporal_features(test_df)
    
    # remove outliers from test set based on lat and lon
    train_df = outlier_removal(train_df, filter=outlier_filter)
    # apply time window cluster labels to df based on strategy specified
    train_df = create_cluster_feature(train_df, strategy=tw_cluster_strategy, verbose=verbose)

    # Run clustering model to get placement set centroids for each TW cluster
    test_df_with_clusters = create_cluster_feature(test_df, strategy=tw_cluster_strategy, verbose=0)
    centroids_dict = create_cluster_centroids(train_df, test_df=test_df_with_clusters,
                                              verbose=verbose, method=placement_method,
                                              lr=lr, n_epochs=n_epochs, batch_size=batch_size)
    
    # create df in format needed for submission
    train_placements_df = centroid_to_submission(centroids_dict, date_start='2019-01-01', date_end='2020-01-01',
                                                 tw_cluster_strategy=tw_cluster_strategy)
    # Run scoring functions
    if verbose > 0:    
        print(f'Total size of test set: {test_df.shape[0]}') 
        print(f'Total size of train set: {crash_df.shape[0]}')
    test_score = score(train_placements_df, test_df, test_start_date=test_period_date_start,
                       test_end_date=test_period_date_end)
    train_score = score(train_placements_df,train_df,
                        test_start_date=test_period_date_start, test_end_date=test_period_date_end)
    if verbose > 0:    
        print(f'Score on test set: {test_score / max(test_df.shape[0],1)}')  
        print(f'Score on train set: {train_score / train_df.shape[0] } (avg distance per accident)')

    # Create df for submitting to zindi
    submission_df = centroid_to_submission(centroids_dict, date_start='2019-07-01', date_end='2020-01-01',
                                           tw_cluster_strategy=tw_cluster_strategy)
    # Create file for submitting to zindi
    create_submission_csv(submission_df, crash_source='prediction', outlier_filter=outlier_filter,
                          tw_cluster_strategy=tw_cluster_strategy, placement_method=placement_method, path='../Outputs/' ,verbose=verbose)

    
    