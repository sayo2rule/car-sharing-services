# 1a.
# Import the CarSharing table into a CSV file.
# Import pandas for reading and manipulating the data and csv for writing the data to a CSV file.
import pandas as pd
import csv

# Read CSV file into a DataFrame.
df = pd.read_csv('/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharing.csv', header=0, index_col=0)

# Set the display option for the columns to obtain a broader view.
pd.set_option('display.max_columns', 50)

# Call out the first 5 rows of the DataFrame to see if the output is exactly what one wants.
print(df.head())

# 1b.
# Data preprocessing.

# 1bi.
# Get the shape of the DataFrame and obtain the number of rows and columns.
rows_initial = df.shape[0]
print("\nNumbers of rows prior to cleaning: {}".format(rows_initial))
print("Numbers of columns excluding the id column: {}".format(df.shape[1]))

# 1bii.
# Check for duplicates using the duplicated() method.
num_duplicates = df.duplicated().sum()
print("\nNumber of duplicates: {}".format(num_duplicates))

# 1biii.
# Check for null values in the DataFrame using the isnull function.
null_values = df.isnull().sum()
print("\nColumns and the numbers of their corresponding null values are:\n{}".format(null_values))

# Get a description of the numeric columns for a better understanding of the best approach
# to handling the null values.
num_desc = df.describe()
print("\nDescription of the numerical columns are:\n{}".format(num_desc))

# Find the mode of the numerical columns.
columns = ["temp", "temp_feel", "humidity", "windspeed"]
modes = df[columns].apply(lambda x: x.mode())
print("\nModes of the numerical columns are:\n{}".format(modes))

# Handle the null values each column separately.
# Fill in missing values in the temp column with the mean of the column.
df["temp"].fillna(df["temp"].mean(), inplace=True)

# Drop rows with null values in "temp_feel", "humidity" and "windspeed" columns.
# "columns" was specified earlier above.
df.dropna(subset=columns, inplace=True)

# Check out the number of rows dropped and print out the differences.
rows_after = df.shape[0]
rows_dropped = rows_initial - rows_after
print("\nNumber of rows dropped: {}".format(rows_dropped))
print("Number of rows left after dropping: {}\nInitial number of rows: {}".format(rows_after, rows_initial))

# Confirm there are no more missing values
null_values = df.isnull().sum()
print("\nColumns showing no more missing values:\n{}".format(null_values))

# 1c.
# Save the processed data as "CarSharingProcessed" to a csv file.
df.to_csv("/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharingProcessed.csv")

# Reload the processed CarSharing table into a Pandas dataframe using the read_csv function.
df = pd.read_csv('/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharingProcessed.csv')

# 1d.
# Handle categorical columns and data in the dataset.
# Import the necessary additional libraries; scikit-learn to apply encoding methods.
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# List the categorical columns and assign it a variable.
cat_columns = ["season", "holiday", "workingday", "weather"]

# Obtain the number of unique values in each of the categorical column
print("\nThe season, holiday, workingday and weather column has {}, {}, {} and {} number of unique values respectively"
      .format((df["season"].nunique()), (df["holiday"].nunique()), (df["workingday"].nunique()), (df["weather"].nunique())))

# Use appropriate encoding method, based on the number of unique values in each column.
# For the season column, weather column, use one-hot encoding.
# For the holiday and workingday columns, use label encoding.

# Create the encoding objects.
onehot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

# Reshape the columns that will go through onehot_encoding among the categorical columns.
season_reshaped = np.array(df["season"]).reshape(-1, 1)
weather_reshaped = np.array(df["weather"]).reshape(-1, 1)

# Apply the appropriate encoding method to each of the categorical columns.
season_values = onehot_encoder.fit_transform(season_reshaped)
holiday_values = label_encoder.fit_transform(df["holiday"])
workingday_values = label_encoder.fit_transform(df["workingday"])
weather_values = onehot_encoder.fit_transform(weather_reshaped)

# Check the resulting data to ensure that it is in the desired format and that all categories are correctly encoded.
print("\nFirst five values of season column before encoding:\n", df["season"][:5])
print("\nFirst five values of season column after encoding:\n", season_values.toarray()[:5])
print("\nFirst five values of holiday column before encoding:\n", df["holiday"][:5])
print("\nFirst five values of holiday column after encoding:\n", list(holiday_values)[:5])
print("\nFirst five values of workingday column before encoding:\n", df["workingday"][:5])
print("\nFirst five values of workingday column after encoding:\n", list(workingday_values)[:5])
print("\nFirst five values of weather column before encoding:\n", df["weather"][:5])
print("\nFirst five values of weather column after encoding:\n", weather_values.toarray()[:5])

# Obtain the unique values in season and weather column, having more than two values.
print("\nThe unique values in the season column: {}".format(list(df["season"].unique())))
print("\nThe unique values in the weather column: {}".format(list(df["weather"].unique())))

# Create the encoded DataFrame by initializing the DataFrame object for each categorical column
season = pd.DataFrame(season_values.toarray(), columns=(list(df["season"].unique())))
holiday = pd.DataFrame(holiday_values, columns=["holiday"])
workingday = pd.DataFrame(workingday_values, columns=["workingday"])
weather = pd.DataFrame(weather_values.toarray(), columns=(list(df["weather"].unique())))

# Combine all categorical columns as one DataFrame.
df_cat_encoded = pd.concat([df["id"], season, holiday, workingday, weather], axis=1)
print("\nThe DataFrame of the concatenated categorical columns have ", df_cat_encoded.shape[0],
      " rows and ",  df_cat_encoded.shape[1], " columns including the id column")

# Save the categorical DataFrame to a csv file.
df_cat_encoded.to_csv("/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharingCatEncoded.csv", index=False)

# Reload the encoded categorical columns table into a Pandas dataframe using the read_csv function.
df_cat_encoded = pd.read_csv('/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharingCatEncoded.csv', index_col=0)
print("\nThe first five rows of the DataFrame are:\n", df_cat_encoded.head())

# 2a.
# Determine if there is a significant relationship between each column and the demand rate.

# Formulate research questions and research hypothesis for each column against the demand rate.
# Research question:
# What is the relationship between each column of the CarSharing dataset and demand rate.

# Research hypothesis:
# Null hypothesis: There is no difference between a named column and demand rate of the CarSharing dataset.
# Alternative hypothesis: There is a difference between a named column and the demand rate of the CarSharing dataset.

# 2ai.
# Determine if there is significant relationship between the numerical columns and the demand rate.
# numerical column and the demand rate.
num_columns = df[["timestamp", "temp", "temp_feel", "humidity", "windspeed"]]

# Use simple linear regression to test for a linear relationship of the numerical columns and demand rate.
# Import the pearsonr and f_oneway module from the scipy.stats library for numerical and categorical columns respectively.
from scipy.stats import pearsonr, f_oneway

# Import the ols module from the statsmodels.formula.api library.
from statsmodels.formula.api import ols

# Iterate over the numerical columns of the DataFrame.
for col in num_columns:
    if col != "timestamp":
        # Fit a simple linear regression model using the ols function from the statsmodels module.
        model = ols(formula=f"demand ~ {col}", data=df)
        results = model.fit()

        # Print the regression equation and the p-value for the model.
        print(f"Y' = {results.params[0]:.3f} + {results.params[1]:.3f}X")
        print(f"Where Y' is the outcome 'demand' and X is the independent variable, {col}.")
        print(f"p-value = {results.pvalues[1]}")

        # Report the test results
        if results.pvalues[1] < 0.05:
            if results.params[1] >= 0:
                print(f"There is a positive significant relationship between {col} and the demand rate (p-value ~ {results.pvalues[1]:.3f})")
            else:
                print(f"There is a negative significant relationship between {col} and the demand rate (p-value ~ {results.pvalues[1]:.3f})")
        else:
            print(f"There is no significant relationship between {col} and the demand rate (p-value ~ {results.pvalues[1]:.3f})")

        # Make a prediction for a new value of the independent variable
        X_new = 1
        Y_pred = results.params[0] + results.params[1] * X_new
        print(f"Prediction for X = {X_new}: Y = {Y_pred:.3f}.\n")

# 2aii.
# Alternatively, use the pearson correlation regression to test for a linear relationship of the numerical columns and demand rate.
# Iterate over the numerical columns of the DataFrame.
for col in num_columns:
    if col != "timestamp":
        # Calculate the p-value using the Pearson correlation coefficient.
        r, p = pearsonr(df[col], df["demand"])

        # Report the test results.
        if p < 0.05:
            if r > 0:
                print(f"There is a positive significant relationship between {col} and the demand rate r ~ {r:.3f}, (p-value ~ {p:.3f})")
            else:
                print(f"There is a negative significant relationship between {col} and the demand rate r ~ {r:.3f}, (p-value ~ {p:.3f})")
        else:
            print(f"There is no significant relationship between {col} and the demand rate r ~ {r:.3f}, (p-value ~ {p:.3f})")

# 2bi.
# Determine if there is significant relationship between the categorical columns and the demand rate.
print(f"\nThe categorical columns in the CarSharing dataset are {cat_columns}.")

# Iterate over the categorical columns of the DataFrame.
for col in cat_columns:
    # Group data by the column name and calculate mean demand rate for each group.
    grouped = df.groupby(col)["demand"].mean().reset_index()

    # Extract the demand rate values for each category of the column.
    categories = [df[df[col] == category]["demand"] for category in grouped[col]]

    # Perform one-way ANOVA test.
    f, p = f_oneway(*categories)

    # Print results.
    print(f"\nFor {col}, the following were obtained:")
    print(f"F-statistic: {f}")
    print(f"p-value: {p}")

    # Interpret and report results.
    if p < 0.05:
        print(f"There is a significant relationship between the demand rate and the {col}.\n"
              f"The F-statistic value of {f} also indicates that the model is a good fit for the data\n"
              f"and that the observed relationship between the {col} and the demand rate is likely not due to chance.")
    else:
        print(f"There is no significant relationship between the demand rate and the {col}.")

# 3a.
# Import the matplotlib.pyplot library for plotting the data.
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DateFormatter

# Load the processed CarSharing table into a Pandas dataframe using the read_csv function.
df = pd.read_csv("/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharingProcessed.csv")

# Extract rows where the year of the timestamp column is 2017.
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_2017 = df.loc[df["timestamp"].dt.strftime('%Y') == '2017']

# Extract the temp, humidity, windspeed, and demand columns from the dataframe in the year 2017.
temp = df_2017['temp']
humidity = df_2017['humidity']
windspeed = df_2017['windspeed']
demand = df_2017['demand']

# 3ai.
# Use the subplots function to create a grid of subplots within a single figure.
# Create a figure with subplots.
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))

# Plot the data for each column using the plot function from matplotlib.pyplot.
# Set the x-axis to the timestamps column and the y-axis to the respective column.
# The first subplot is the top left plot.
ax[0, 0].plot(df_2017['timestamp'], temp)
ax[0, 0].set_title("Temp")
ax[0, 0].set_xlabel("Timestamp")
ax[0, 0].set_ylabel("Value")
ax[0, 0].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

# The second subplot is the top right plot.
ax[0, 1].plot(df_2017['timestamp'], humidity)
ax[0, 1].set_title("Humidity")
ax[0, 1].set_xlabel("Timestamp")
ax[0, 1].set_ylabel("Value")
ax[0, 1].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

# The third subplot is the bottom left plot.
ax[1, 0].plot(df_2017['timestamp'], windspeed)
ax[1, 0].set_title("Windspeed")
ax[1, 0].set_xlabel("Timestamp")
ax[1, 0].set_ylabel("Value")
ax[1, 0].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

# The fourth subplot is the bottom right plot.
ax[1, 1].plot(df_2017['timestamp'], demand)
ax[1, 1].set_title("Demand")
ax[1, 1].set_xlabel("Timestamp")
ax[1, 1].set_ylabel("Value")
ax[1, 1].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

# Set the tick locations and labels for the x-axis of each subplot
for row in ax:
    for col in row:
        col.xaxis.set_major_locator(MultipleLocator(int(30)))
        col.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

# Observe the plots and look for patterns in the data.

# 3aii.
# Import necessary additional libraries
from statsmodels.tsa.seasonal import seasonal_decompose

# Convert the data for each variable into a time series object
temp = pd.Series(df_2017['temp'].values, index=df_2017['timestamp'])
humidity = pd.Series(df_2017['humidity'].values, index=df_2017['timestamp'])
windspeed = pd.Series(df_2017['windspeed'].values, index=df_2017['timestamp'])
demand = pd.Series(df_2017['demand'].values, index=df_2017['timestamp'])

# Create a figure with subplots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))

# Plot the data for each column in a separate subplot
ax[0, 0].plot(temp)
ax[0, 0].set_title("Temp")
ax[0, 0].set_xlabel("Timestamp")
ax[0, 0].set_ylabel("Value")
ax[0, 0].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

ax[0, 1].plot(humidity)
ax[0, 1].set_title("Humidity")
ax[0, 1].set_xlabel("Timestamp")
ax[0, 1].set_ylabel("Value")
ax[0, 1].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

ax[1, 0].plot(windspeed)
ax[1, 0].set_title("Windspeed")
ax[1, 0].set_xlabel("Timestamp")
ax[1, 0].set_ylabel("Value")
ax[1, 0].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

ax[1, 1].plot(demand)
ax[1, 1].set_title("Demand")
ax[1, 1].set_xlabel("Timestamp")
ax[1, 1].set_ylabel("Value")
ax[1, 1].set_xticklabels(df_2017['timestamp'].dt.strftime('%Y'), rotation=45)

# Set the tick locations and labels for the x-axis of each subplot
for row in ax:
    for col in row:
        col.xaxis.set_major_locator(MultipleLocator(int(30)))
        col.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

# Alternatively, use a single plot that will contain all the four plots for each column.
# Set the x-axis to the timestamps column and the y-axis to the respective column.
plt.figure(figsize=(10, 6))
plt.plot(df_2017['timestamp'], temp, label='temp')
plt.plot(df_2017['timestamp'], humidity, label='humidity')
plt.plot(df_2017['timestamp'], windspeed, label='windspeed')
plt.plot(df_2017['timestamp'], demand, label='demand')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.title("Data for 2017")
plt.xticks(rotation=90)
plt.show()

# Observe the plots and look for patterns in the data.

# 3aiii.
# Another way of looking for patterns is to employ the seaborn's line plot function.
# Import the seaborn library for plotting the data.
import seaborn as sns

# Use seaborn's line plot function to plot the data for each column.
# Set the x-axis to the timestamp column and the y-axis to the respective columns.
sns.lineplot(x='timestamp', y='temp', data=df_2017, label='temp')
sns.lineplot(x='timestamp', y='humidity', data=df_2017, label='humidity')
sns.lineplot(x='timestamp', y='windspeed', data=df_2017, label='windspeed')
sns.lineplot(x='timestamp', y='demand', data=df_2017, label='demand')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title("Data for 2017")
plt.xticks(rotation=45)
plt.show()

# Observe the plots and look for patterns in the data.

# 4a.
# Import the statsmodels.tsa.arima_model for fitting the ARIMA model and
# sklearn.model_selection.train_test_split for splitting the data into training and testing sets.
import statsmodels.tsa.arima_model as arima_model
from sklearn.model_selection import train_test_split

# Extract the timestamp and demand columns from the dataframe.
timestamp = df['timestamp']
demand = df['demand']

# Convert the timestamps column to a datetime data type using the to_datetime function from pandas.
timestamp = pd.to_datetime(timestamp)

# Set the index of the dataframe to the timestamps' column. This will allow us to easily resample the
# data to a weekly frequency.

df.index = timestamp

# Resample the demand column to a weekly frequency and calculate the mean for each week.
demand_weekly_mean = demand.resample('W').mean()

# Split the data into training and testing sets using the train_test_split function from sklearn.model_selection.
# Set the test set size to 30%.
X_train, X_test, y_train, y_test = train_test_split(demand_weekly_mean.index, demand_weekly_mean.values, test_size=0.3)

# Convert the training and testing sets to a 1-dimensional array using the values attribute.
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Fit the ARIMA model to the training data. Set the order of the model to (1,1,1).
model = arima_model.ARIMA(y_train, order=(1, 1, 1))

# Make predictions on the test data using the predict method of the fitted model.
predictions = model.predict(X_test)

# Calculate the mean squared error between the predictions and the true values
# using the mean_squared_error function from sklearn.metrics.
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)

# Report the results of the prediction in a report, including the mean squared error and any observations about the performance of the model.

# Alternatively, to use an ARIMA model to predict the weekly
# average demand rate, you can follow these steps:
# Import the necessary libraries for building the ARIMA model:
from statsmodels.tsa.arima_model import ARIMA

# Split the data into a training set and a testing set, using 30% of the data for testing:
n = len(df)
split_point = int(n * 0.7)

train_data = df['demand'][:split_point]
test_data = df['demand'][split_point:]

# Determine the optimal hyperparameters for the ARIMA model using the auto_arima function:
model = pm.auto_arima(train_data, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

# Fit the ARIMA model using the fit method and the training data:
model.fit(train_data)

# Use the predict method to make predictions on the testing data:
predictions = model.predict(n_periods=len(test_data))

# Calculate the weekly average
# demand rate by taking the mean of the demand data for each week
weekly_average_demand = df['demand'].resample('W').mean()

# Calculate the mean squared error between the predicted demand rate and the actual
# weekly average demand rate

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(weekly_average_demand, predictions)
# You can then report the results of the prediction in a report, including the mean
# squared error and any observations about the performance of the model.

# Use a random forest regressor and a deep neural network to predict the
# demand rate and report the minimum square error for
# each model. Which one is working better? Why? Please describe the reason.

# 5. To use a random forest regressor and a deep neural network to predict the demand rate
# and report the minimum square error for each model, you can follow these steps:
# Import the necessary libraries for building the models:
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Split the data into input features and the target variable:
X = df.drop(columns=['demand', 'timestamps'])
y = df['demand']

# Split the data into a training set and a testing set, using 30% of the data for testing:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Build the random forest regressor model:
model_rf = RandomForestRegressor(n_estimators=100)

# Train the model using the fit method and the training data:
model_rf.fit(X_train, y_train)

# Make predictions on the testing data using the predict method
predictions_rf = model_rf.predict(X_test)

# Calculate the mean
# squared error between the predicted demand rate and the actual demand rate
from sklearn.metrics import mean_squared_error

mse_rf = mean_squared_error(y_test, predictions_rf)

# Build the deep neural network model:
model_nn = Sequential()
model_nn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1))

# Compile the model using the compile method and specify the loss function and optimizer:
model_nn.compile(loss='mean_squared_error', optimizer='adam')

# Train the model using the fit method and the training data:
model_nn.fit(X_train, y_train, epochs=10)

# Make predictions on the testing data using the predict method:
predictions_nn = model_nn.predict(X_test)

# Calculate the mean squared error between
# the predicted demand rate and the actual demand rate:
mse_nn = mean_squared_error(y_test, predictions_nn)

# You can then compare the mean squared errors of the two models to determine which one
# is performing better. If the random forest regressor has a lower mean squared error,
# it may be performing better than the deep neural network. On the other hand, if the
# deep neural network has a lower mean squared error, it may be performing better.
#
# You can describe the reason for the better performance in a report, considering factors
# such as the complexity of the model, the amount of training data, and the features of the data. For example

# Alternatively, predict the demand rate using a random forest regressor and a deep neural network
# import the necessary libraries

# Pandas to read and manipulate the data,
# scikit-learn to train the random forest regressor, and
# tensorflow or keras to train the deep neural network.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

# Read the data into a DataFrame
df = pd.read_csv('/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharingCleaned.csv')

# Split the data into features and target
X = df.drop(['demand', 'timestamps'], axis=1)
y = df['demand']

# 6. To categorize the demand rate into two groups and use three different
# classifiers to predict the demand rates' labels, you can follow these steps:

# Import the necessary libraries for building the classifiers:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Calculate the average demand rate:
average_demand = df['demand'].mean()

# Create a new column in the DataFrame to hold the labels for the demand rate groups
df['demand_group'] = df['demand'].apply(lambda x: 1 if x > average_demand else 2)

# Split the data into input features and the target variable:
X = df.drop(columns=['demand', 'demand_group', 'timestamps'])
y = df['demand_group']

# Split the data into a training set and a testing set, using 30% of the data for testing:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Build the random forest classifier model:
model_rf = RandomForestClassifier(n_estimators=100)

# Train the model using the fit method and the training data:
model_rf.fit(X_train, y_train)

# Make predictions on the testing data using the predict method:
predictions_rf = model_rf.predict(X_test)

# Calculate the accuracy of the model using the accuracy_score
# function from the sklearn.metrics library:
from sklearn.metrics import accuracy_score

accuracy_rf = accuracy_score(y_test, predictions_rf)

# Build the logistic regression model:
model_lr = LogisticRegression()

# Train the model using the fit method and the training data:
model_lr.fit(X_train, y_train)

# Make predictions on the testing data using the predict method:
predictions_lr = model_lr.predict(X_test)

# Calculate the accuracy of the model using the accuracy_score function:
accuracy_lr = accuracy_score(y_test, predictions_lr)

# Build the SVM model:
model_svm = SVC()

# Train the model using the fit method and the training data:
model_svm.fit(X_train, y_train)

# Make predictions on the testing data using the `predict` method:
predictions_svm = model_svm.predict(X_test)

# Calculate the accuracy of the model using the accuracy_score function:
accuracy_svm = accuracy_score(y_test, predictions_svm)

# You can then compare the accuracy of the three models to determine which one is
# performing better. If the random forest classifier has a higher accuracy,
# it may be performing better than the other two models. If the logistic
# regression model has a higher accuracy, it may be performing better.
# If the SVM model has a higher accuracy, it may be performing better.

# You can report the accuracy of all
# models in a report, along with any observations about the performance of the models.

# 7. To determine which k value gives the most uniform clusters when clustering
# the temp data in 2017 using 2 different methods, you can follow these steps:

# Import the necessary libraries for clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Extract the temp data for 2017
df_2017 = df[df['timestamps'].dt.year == 2017]['temp']

# Standardize the data using the StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_2017.values.reshape(-1, 1))

# Iterate over the different values of k
for k in [2, 3, 4, 12]:

# Use the KMeans algorithm to cluster the data:
    model_kmeans = KMeans(n_clusters=k)
    model_kmeans.fit(X)
    labels_kmeans = model_kmeans.labels_

# Calculate the number of samples in each cluster:
cluster_counts_kmeans = [len(labels_kmeans[labels_kmeans == i]) for i in range(k)]

# Use the DBSCAN algorithm to cluster the data
model_dbscan = DBSCAN()
model_dbscan.fit(X)
labels_dbscan = model_dbscan.labels_

# Calculate the number of samples in each cluster
cluster_counts_dbscan = [len(labels_dbscan[labels_dbscan == i]) for i in range(k)]

# Calculate the variance of the cluster counts for each method:
variance_kmeans = np.var(cluster_counts_kmeans)
variance_dbscan = np.var(cluster_counts_dbscan)

# Compare the variance of the cluster counts for each
# method and determine which k value gives the most uniform clusters:
if variance_kmeans < variance_dbscan:
    print(f"KMeans with k = {k} gives the most uniform clusters with variance {variance_kmeans}")
else:
    print(f"DBSCAN with k = {k} gives the most uniform clusters with variance {variance_dbscan}")

# The k value that gives the most uniform clusters will be the one with the lowest
# variance of the cluster counts. You can
# report the k value and the method that gives the most uniform clusters in a report.
