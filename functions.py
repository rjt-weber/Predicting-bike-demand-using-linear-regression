import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error, make_scorer, r2_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan



def create_predictions(fitted_logYmodel, data_fe, y):
    """
    returns a dataframe of the transformed predictions of the model, which is passed as an argument, and the actual values
    
    arguments
    ---------
    fitted_logYmodel: linear regression model of the logarithm of y vs. x
    data_fe: prepared features for prediction
    y: target variable
    
    return
    ------
    dataframe
    
    """
    
    ylog_pred = fitted_logYmodel.predict(data_fe)
    y_pred = np.exp(ylog_pred)
    y_pred_s = pd.Series(y_pred)

    datetime = pd.Series(data_fe.index)
    result = pd.concat([datetime, y_pred_s], axis=1)
    result.columns = ['datetime', 'count']
    result.set_index('datetime', inplace=True)

    df = pd.concat([y, result], axis=1)
    df.columns = ['truth', 'pred']
    return df


def create_predictions_simplelr(fitted_model, data_fe, y):
    """
    returns a dataframe of the predictions of the model, which is passed as an argument, and the actual values
    
    arguments
    ---------
    fitted_model: linear regression model of y vs. x
    data_fe: prepared features for prediction
    y: target variable
    
    return
    ------
    dataframe
    
    """
    
    y_pred = fitted_model.predict(data_fe)
    y_pred_s = pd.Series(y_pred)

    datetime = pd.Series(data_fe.index)
    result = pd.concat([datetime, y_pred_s], axis=1)
    result.columns = ['datetime', 'count']
    result.set_index('datetime', inplace=True)

    df = pd.concat([y, result], axis=1)
    df.columns = ['truth', 'pred']
    return df


def plot_diagnostics_timescales(df):
    """
    plots multi-panel time series of predictions and actuals on different timescales
    
    arguments
    ---------
    df: dataframe of truth and prediction columns labeled "truth" and "pred"
    
    
    return
    ------
    2 figures
    
    """
    
    

    fig1, ax1 = plt.subplots(2, 1, figsize=(6.4*2.5, 4.8*2.2))

    ax1[0].plot(df['truth'].rolling(window=24 * 30).mean(), label='Actual', linewidth=3)
    ax1[0].plot(df['pred'].rolling(window=24 * 30).mean(), label='Predicted', linewidth=3)
    ax1[0].legend(loc = 'upper left')
    ax1[0].set_title("Seasonality with 30-Day-Smoothing", fontweight = "bold")#, size=24, fontweight="bold")
    ax1[0].set_ylabel("Bike Rental Count")#, fontsize=20, fontweight="bold")
    
    ax1[1].plot(df.loc['2012-12-01':'2012-12-08'], linewidth=3)
    ax1[1].set_title("Weekly cycle", fontweight = "bold")#, fontsize=24, fontweight="bold")
    ax1[1].set_ylabel("Bike Rental Count")#, fontsize=20, fontweight="bold")
    
    
    fig2, ax2 = plt.subplots(2, 2, figsize=(6.4*2.5, 4.8*2.2), sharey=True)

    ax2[0, 0].plot(df['truth'].loc['2012-12-17'], label='Actual', linewidth=3)
    ax2[0, 0].plot(df['pred'].loc['2012-12-17'], label='Predicted', linewidth=3)
    ax2[0, 0].set_title("Monday", fontweight = "bold")
    ax2[0, 0].set_ylabel("Bike Rental Count")
    ax2[0, 0].legend(loc = 'upper left')
    
    ax2[0, 1].plot(df.loc['2012-12-19'], linewidth=3)
    ax2[0, 1].set_title("Wednesday", fontweight = "bold")
    
    ax2[1, 0].plot(df.loc['2012-12-15'], linewidth=3)
    ax2[1, 0].set_title("Saturday",fontweight = "bold")
    ax2[1, 0].set_ylabel("Bike Rental Count")

    ax2[1, 1].plot(df.loc['2012-12-16'], linewidth=3)
    ax2[1, 1].set_title("Sunday",fontweight = "bold")

    plt.setp(ax2[0,0].get_xticklabels(), rotation=20, ha='right')
    plt.setp(ax2[0,1].get_xticklabels(), rotation=20, ha='right')
    plt.setp(ax2[1,0].get_xticklabels(), rotation=20, ha='right')
    plt.setp(ax2[1,1].get_xticklabels(), rotation=20, ha='right')
    
    return fig1, fig2
    
    
def check_lr_assumptions(df, data_fe):
    """
    prints multiple statistical tests and returns a dataframe containing residuals
    
    arguments
    ---------
    df: dataframe of truth and prediction columns labeled "truth" and "pred"
    data_fe: prepared features for prediction
    
    return
    ------
    dataframe
    
    """
    
    df['residuals'] = df['pred'] - df['truth']
    
    print("mean of residuals:", df['residuals'].mean())
    print("variance of residuals:", df['residuals'].var())
    print("skewness of residuals:", stats.skew(df.residuals))
    print("kurtosis of residuals:", stats.kurtosis(df.residuals))
    print("kurtosis test of residuals:", stats.kurtosistest(df.residuals))
    print("normal test of residuals (scipy stats):", stats.normaltest(df.residuals))
    print("Jarque Bera test for normality of residuals:", stats.jarque_bera(df.residuals))
    print("Breusch Pagan test for heteroscedasticity:", het_breuschpagan(df.residuals, data_fe))

    return df
    
    
def plot_residuals_hist(df):
    """
    plots histogram of residuals
    
    arguments
    ---------
    df: dataframe of truth and prediction columns labeled "truth" and "pred"
    
    return
    ------
    none
    
    """
    
    
    df['residuals'] = df['pred'] - df['truth']

    plt.figure(figsize = [8.0, 6.0])
    plt.hist(df.residuals, bins=30, density=True)
    plt.title("Residuals histogram")

    mean = np.mean(df.residuals)
    variance = np.var(df.residuals)
    sigma = np.sqrt(variance)
    x = np.linspace(min(df.residuals), max(df.residuals), 100)
    plt.plot(x, stats.norm.pdf(x, mean, sigma), label='Gaussian', linewidth=3)
    plt.legend()

    
def plot_diagnostics_lr(df, data_fe):
    """
    plots multi-panel diagnostics of linear regression assumptions
    
    arguments
    ---------
    df: dataframe containing residuals columns
    
    
    return
    ------
    none
    
    """

    fig3, ax3 = plt.subplots(3, 3, figsize=(20, 15))
    ax3[0, 0].hist(df.residuals, bins=30, density=True)
    ax3[0, 0].set_title("Residuals histogram")

    mean = np.mean(df.residuals)
    variance = np.var(df.residuals)
    sigma = np.sqrt(variance)
    x = np.linspace(min(df.residuals), max(df.residuals), 100)
    
    ax3[0, 0].plot(x, stats.norm.pdf(x, mean, sigma), label='Gaussian', linewidth=3)
    ax3[0, 0].legend()
    
    ax3[0, 1].scatter(data_fe["yr_2012"], df.residuals, s = 2)
    ax3[0, 1].set_title("year_2012 yes/no")
    
    ax3[0, 2].scatter(data_fe["humidity"], df.residuals, s = 2)
    ax3[0, 2].set_title("humidity")
    
    ax3[1, 0].scatter(data_fe["windspeed"], df.residuals, s = 2)
    ax3[1, 0].set_title("windspeed")
    
    ax3[1, 1].scatter(data_fe["atemp"], df.residuals, s = 2)
    ax3[1, 1].set_title("atemp")
    
    ax3[1, 2].scatter(data_fe["weather"], df.residuals, s = 2)
    ax3[1, 2].set_title("weather")
    
    ax3[2, 0].scatter(data_fe["hr_8_workingday"], df.residuals, s = 2)
    ax3[2, 0].set_title("hr_8_workingday")
    
    ax3[2, 1].scatter(data_fe["windspeed_Sq"], df.residuals, s = 2)
    ax3[2, 1].set_title("windspeed_Sq")
    
    ax3[2, 2].scatter(data_fe["humidity_Sq"], df.residuals, s = 2)
    ax3[2, 2].set_title("humidity_Sq")




def save_figure(fig, filename):
    """
    saves figure to filename path
    
    arguments
    ---------
    fig: figure
    filename: string containing path to file
    
    
    return
    ------
    none
    
    """
    
    fig.savefig(filename)
    

        
def feature_engineer(data, scaler_weatherFeatures, scaler_allFeatures, kind = "train"):
    """
    prepares and scales features for linear regression including 
    - interaction terms, 
    - second-order features of weather data, 
    - weather variables: humidity, windspeed, atemp, weather,
    - one-hot-encoded hours, months, years
    
    arguments
    ---------
    data: raw training data
    scaler_weatherFreatures: a scaler initialized using sklearn's StandardScaler() which scales the weather features prior to calculating their second-order terms
    scaler_allFeatures: a scaler initialized using sklearn's StandardScaler() which scales all features after engineering 
    kind: a string containing either "train" or "test" to indicate the type of dataset passed to the function
    
    return
    ------
    dataframe of engineered features
    
    """

    df = data[['humidity', 'windspeed', 'atemp', 'weather', 'workingday']]
    
    df['yr'] = data.index.year
    df['mon'] = data.index.month
    df['hr'] = data.index.hour
    
    variables_to_be_encoded = df[['yr', 'mon', 'hr']]
    df_notEncoded = df.drop(columns=['yr', "mon", 'hr'])
    df_encoded = pd.get_dummies(variables_to_be_encoded, columns=['yr', 'mon', 'hr'], drop_first=True)
    df_unscaled = pd.concat([df_encoded, df_notEncoded], axis=1)
    
    if kind == "train":
        
        scaled_weatherFeatures = scaler_weatherFeatures.fit_transform(df_unscaled[['windspeed','humidity','atemp']])
        
    elif kind == "test":
        
        scaled_weatherFeatures = scaler_weatherFeatures.transform(df_unscaled[['windspeed', 'humidity', 'atemp']])
    
    df_weather_scaled = pd.DataFrame(scaled_weatherFeatures)
    df_weather_scaled.columns = ['windspeed', 'humidity', 'atemp']
    df_weather_scaled.index = df_unscaled.index
    
    df_unscaledFeatures = df_unscaled.drop(columns=['windspeed', 'humidity', 'atemp'])
    df_features = df_weather_scaled.merge(df_unscaledFeatures, left_index=True, right_index=True)
    
    df_features['hr_8_workingday'] = df_features.apply(lambda row: row['hr_8'] * row['workingday'], axis=1)
    df_features['hr_18_workingday'] = df_features.apply(lambda row: row['hr_18'] * row['workingday'], axis=1)
    df_features['hr_13_workingday'] = df_features.apply(lambda row: row['hr_13'] * row['workingday'], axis=1)
    df_features['hr_15_workingday'] = df_features.apply(lambda row: row['hr_15'] * row['workingday'], axis=1)
    df_features['hr_21_workingday'] = df_features.apply(lambda row: row['hr_21'] * row['workingday'], axis=1)
    df_features['hr_1_workingday'] = df_features.apply(lambda row: row['hr_1'] * row['workingday'], axis=1)
    df_features['hr_14_workingday'] = df_features.apply(lambda row: row['hr_14'] * row['workingday'], axis=1)
    df_features['hr_16_workingday'] = df_features.apply(lambda row: row['hr_16'] * row['workingday'], axis=1)

    df_features['windspeed_Sq'] = df_features.apply(lambda row: row['windspeed'] * row['windspeed'], axis=1)
    df_features['humidity_Sq'] = df_features.apply(lambda row: row['humidity'] * row['humidity'], axis=1)
    df_features['atemp_Sq'] = df_features.apply(lambda row: row['atemp'] * row['atemp'], axis=1)
    
    df_features.drop(columns = "workingday", inplace = True)
    
    if kind == "train":

        scaled_features = scaler_allFeatures.fit_transform(df_features)
        
    elif kind == "test":
        
        scaled_features = scaler_allFeatures.transform(df_features)
        
    df_features_scaled = pd.DataFrame(scaled_features)
    df_features_scaled.columns = df_features.columns
    df_features_scaled.index = df_features.index

    return df_features_scaled, scaler_weatherFeatures, scaler_allFeatures


def feature_engineer_rbf(data):
    """
    prepares features for linear regression including 
    - interaction terms of rbfs and non-workingdays, 
    - weather variables: humidity, windspeed, atemp, weather,
    - one-hot-encoded hours, months, years
    
    arguments
    ---------
    data: raw training data
    
    
    return
    ------
    dataframe of engineered features
    
    """

    df = data[
        ['humidity', 'windspeed', 'atemp', 'weather', 'workingday', 'rbf_22', 'rbf_8', 'rbf_3', 'rbf_11', 'rbf_15']]
    df['yr'] = data.index.year
    df['mon'] = data.index.month
    df['hr'] = data.index.hour
    
    variables_to_be_encoded = df[['yr', 'mon', 'hr']]
    df_notEncoded = df.drop(columns=['yr', "mon", 'hr'])
    df_encoded = pd.get_dummies(variables_to_be_encoded, columns=['yr', 'mon', 'hr'], drop_first=True)
    df_features = pd.concat([df_encoded, df_notEncoded], axis=1)

    df_features['hr_8_weekendHol_rbf'] = df_features.apply(lambda row: row['rbf_22']*(not(row['workingday'])),axis=1)
    df_features['hr_18_weekendHol_rbf'] = df_features.apply(lambda row: row['rbf_8']*(not(row['workingday'])),axis=1)
    df_features['hr_13_weekendHol_rbf'] = df_features.apply(lambda row: row['rbf_3']*(not(row['workingday'])),axis=1)
    df_features['hr_21_weekendHol_rbf'] = df_features.apply(lambda row: row['rbf_11']*(not(row['workingday'])),axis=1)
    df_features['hr_1_weekendHol_rbf'] = df_features.apply(lambda row: row['rbf_15']*(not(row['workingday'])),axis=1)
    
    df_features.drop(columns=['rbf_22', 'rbf_8', 'rbf_3', 'rbf_11', 'rbf_15','workingday'], inplace = True)

    return df_features


def feature_engineer_interaction_terms(data):
    """
    prepares features for linear regression including 
    - interaction terms, 
    - weather variables: humidity, windspeed, atemp, weather,
    - one-hot-encoded hours, months, years
    
    arguments
    ---------
    data: raw training data
    
    
    return
    ------
    dataframe of engineered features
    
    """

    df = data[['humidity', 'windspeed', 'atemp', 'weather', 'workingday']]
    
    df['yr'] = data.index.year
    df['mon'] = data.index.month
    df['hr'] = data.index.hour
    
    variables_to_be_encoded = df[['yr', 'mon', 'hr']]
    df_notEncoded = df.drop(columns=['yr', "mon", 'hr'])
    df_encoded = pd.get_dummies(variables_to_be_encoded, columns=['yr', 'mon', 'hr'], drop_first=True)
    df_features = pd.concat([df_encoded, df_notEncoded], axis=1)
    
    df_features['hr_8_workingday'] = df_features.apply(lambda row: row['hr_8'] * row['workingday'], axis=1)
    df_features['hr_18_workingday'] = df_features.apply(lambda row: row['hr_18'] * row['workingday'], axis=1)
    df_features['hr_13_workingday'] = df_features.apply(lambda row: row['hr_13'] * row['workingday'], axis=1)
    df_features['hr_21_workingday'] = df_features.apply(lambda row: row['hr_21'] * row['workingday'], axis=1)
    df_features['hr_1_workingday'] = df_features.apply(lambda row: row['hr_1'] * row['workingday'], axis=1)
    
    df_features.drop(columns = "workingday", inplace = True)

    return df_features


def feature_engineer_single_interaction_term(data):
    """
    prepares features for linear regression including 
    - single interaction term between 8th hour and workingday booleans, 
    - weather variables: humidity, windspeed, atemp, weather,
    - one-hot-encoded hours, months, years
    
    arguments
    ---------
    data: raw training data
    
    
    return
    ------
    dataframe of engineered features
    
    """

    df = data[['humidity', 'windspeed', 'atemp', 'weather', 'workingday']]
    df['yr'] = data.index.year
    df['mon'] = data.index.month
    df['hr'] = data.index.hour
    
    variables_to_be_encoded = df[['yr', 'mon', 'hr']]
    df_notEncoded = df.drop(columns=['yr', "mon", 'hr'])
    df_encoded = pd.get_dummies(variables_to_be_encoded, columns=['yr', 'mon', 'hr'], drop_first=True)
    df_features = pd.concat([df_encoded, df_notEncoded], axis=1)
    
    df_features['hr_8_workingday'] = df_features.apply(lambda row: row['hr_8'] * row['workingday'], axis=1)
    
    df_features.drop(columns=['workingday'], inplace = True)

    return df_features


def feature_engineer_baseline(data):
    """
    prepares features for linear regression including 
    - weather variables: humidity, windspeed, atemp, weather,
    - one-hot-encoded hours, day of the week, months, years
    
    arguments
    ---------
    data: raw training data
    
    
    return
    ------
    dataframe of engineered features
    
    """

    df = data[['humidity', 'windspeed', 'atemp', 'weather']]
    
    df['yr'] = data.index.year
    df['mon'] = data.index.month
    df['dayOfWeek'] = data.index.dayofweek
    df['hr'] = data.index.hour
    
    variables_to_be_encoded = df[['yr', 'mon', 'dayOfWeek', 'hr']]
    df_notEncoded = df.drop(columns=['yr', "mon", "dayOfWeek", 'hr'])
    df_encoded = pd.get_dummies(variables_to_be_encoded, columns=['dayOfWeek', 'yr', 'mon', 'hr'], drop_first=True)
    df_features = pd.concat([df_encoded, df_notEncoded], axis=1)
    
    return df_features


def rbf(x, width, mean):
    """
    returns a numpy array of the probability of normally distributed x-variable
    
    arguments
    ---------
    x: normally distributed variable
    width: variance of x
    mean: mean of x
    
    return
    ------
    numpy array
    
    """
    
    return np.exp(-(x - mean) ** 2 / (2 * width))


def rbf_transform(df, freq, width):
    """
    returns radial based function-encoded features based on the normal distribution
    
    arguments
    ---------
    df: dataframe of features to be rbf-encoded
    freq: frequency
    width: chosen width of the normal distribution
    
    return
    ------
    dataframe
    
    """
    
    
    x = np.arange(df.shape[0])
    for i in range(0, freq):
        df[f'rbf_{i}'] = 0
        j = -freq
        while j <= df.shape[0]:
            df[f'rbf_{i}'] += rbf(x, width, i + j)
            j += freq
            
    return df


def print_model_scores(model, X_train, X_test, y_train, y_test):
    """
    prints model scores
    
    arguments
    ---------
    model: machine learning model
    X_train: feature engineered training data 
    X_test: feature engineedred test data
    y_train: target y-variable of the training data set
    y_test: target y-variable of the test data set
    
    return
    ------
    none
    
    """

    print(f'Training Score rSq: {model.score(X_train, y_train)}')
    print(f'Testing Score rSq: {model.score(X_test, y_test)}')


def print_logmodel_scores(model, X_train, X_test, y_train, y_test):
    """
    prints multiple scores of a model trained against the logarithm of the target variable
    
    arguments
    ---------
    model: machine learning model
    X_train: feature engineered training data 
    X_test: feature engineedred test data
    y_train: target y-variable of the training data set
    y_test: target y-variable of the test data set
    
    return
    ------
    none
    
    """

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    rSq_train = r2_score(y_train, np.exp(y_pred_train))
    rSq_test = r2_score(y_test, np.exp(y_pred_test))

    mSqE_test = mean_squared_error(y_test, np.exp(y_pred_test))
    mSqE_train = mean_squared_error(y_train, np.exp(y_pred_train))
    
    rmSqE_test = np.sqrt(mSqE_test)
    rmSqE_train = np.sqrt(mSqE_train)

    mSqLE_test = mean_squared_log_error(y_test, np.exp(y_pred_test))
    mSqLE_train = mean_squared_log_error(y_train, np.exp(y_pred_train))

    rmSqLE_test = np.sqrt(mSqLE_test)
    rmSqLE_train = np.sqrt(mSqLE_train)
    
    result_dict = {'Train rSq': rSq_train, 'Test rSq': rSq_test, 'Train rmSqE': rmSqE_train, 'Test rmSqE': rmSqE_test, 'Train mSqLE': mSqLE_train, 'Test mSqLE': mSqLE_test}
    
    result = pd.DataFrame(result_dict, index=[0])
    print(result)


def print_cross_val_results(model, X_train, y_train):
    """
    prints multiple cross-validation scores of a model
    
    arguments
    ---------
    model: machine learning model
    X_train: feature engineered training data 
    y_train: target y-variable of the training data set
    
    return
    ------
    none
    
    """

    mse = make_scorer(mean_squared_error)
    rmse = make_scorer(mean_squared_error, squared = False)
    
    rSq = cross_val_score(model, X_train, y_train, cv=5)
    mSqLE = cross_val_score(model, X_train, y_train, cv=5, scoring=mse)
    rmSqLE = np.sqrt(mSqLE)
    
    result_dict = {'rSq': rSq, 'mean rSq': rSq.mean(), 'std rSq': rSq.std(), 'mSqLE': mSqLE, 'rmSqLE': rmSqLE}
    result = pd.DataFrame(result_dict)

    print(result)


def bootstrapping(model, X_train, y_train):
    """
    prints bootstrapped confidence intervals of the mean squared log error
    
    *** function under construction: currently predicts on the training data, but should predict on the out-of-sample test data
    
    arguments
    ---------
    model: machine learning model
    X_train: feature engineered training data 
    y_train: target y-variable of the training data set
    
    return
    ------
    none
    
    """
    
    boots = []
    for i in range(1000):
        Xb, yb = resample(X_train, y_train)
        model.fit(Xb, np.log(yb))
        y_pred_train = model.predict(Xb)
        mSqLE = mean_squared_log_error(yb, np.exp(y_pred_train))
        rmSqLE = np.sqrt(mSqLE)
        boots.append(rmSqLE)

    boots.sort()
    ci80 = boots[100:-100]
    print(f"80% confidence interval: {ci80[0]:5.2} -{ci80[-1]:5.2}")
    ci90 = boots[50:-50]
    print(f"90% confidence interval: {ci90[0]:5.2} -{ci90[-1]:5.2}")
    ci95 = boots[25:-25]
    print(f"95% confidence interval: {ci95[0]:5.2} -{ci95[-1]:5.2}")
    ci99 = boots[5:-5]
    print(f"99% confidence interval: {ci99[0]:5.2} -{ci99[-1]:5.2}")




def submit(model, train_data, test_data, y, file_name="submission_default.csv"):
    """
    creates kaggle submission file from entire pipeline:
    - feature engineering of test and training data
    - fits model to logarithm of target y-variable and training data
    - makes predictions of the test data set with trained model
    - generates csv file from predictions
    
    arguments
    ---------
    model: unfitted machine learning model
    train_data: raw training data
    test_data: raw test data
    y: target y-variable of the training data set
    file_name: string containing path to file
    
    return
    ------
    none
    
    """

    data_fe = feature_engineer(train_data)
    model.fit(data_fe, np.log(y))
    data_test_fe = feature_engineer(test_data)
    y_pred = model.predict(data_test_fe)
    y_pred_s = pd.Series(np.exp(y_pred))
    datetime = pd.Series(test_data.index)
    result = pd.concat([datetime, y_pred_s], axis=1)
    submission_data = result
    submission_data.columns = ['datetime', 'count']
    submission_data.to_csv(file_name, index=False)
    print(f"Successfully generated prediction: {file_name}.")
