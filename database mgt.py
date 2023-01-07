# 1a.
# Create an SQLite database and import the data into a table named “CarSharing”.
# Import the sqlite3 and csv modules.
import sqlite3 as sq
import csv

# Create a connection to the SQLite database by calling the "connect()" function.
conn = sq.connect("CarSharing.db", isolation_level=None)

# Create a cursor object to execute the SQLite commands using the cursor() method of the connection object.
cur = conn.cursor()

# use execute() method of the cursor object to create a new table in the database.
cur.execute('''
CREATE TABLE CarSharing (
    id INTEGER PRIMARY KEY, 
    timestamp TIMESTAMP, 
    season TEXT, 
    holiday TEXT, 
    workingday TEXT, 
    weather TEXT, 
    temp REAL, 
    temp_feel REAL, 
    humidity REAL, 
    windspeed REAL, 
    demand REAL);
''')

# Open the csv file and perform some preprocessing techniques on the csv file.
with open("/Users/user/Documents/Keele AI & DS/CSC-40054 (Data Analytics and Databases)/Coursework/CarSharing.csv", "r") as file:

    # Use the csv reader to read the file.
    reader = csv.reader(file)

    # skip the header row.
    next(reader)

    # iterate over the rows.
    for row in reader:
        # insert data into the table CarSharing.
        cur.execute("INSERT INTO CarSharing VALUES (?,?,?,?,?,?,?,?,?,?,?);", row)

# Fetch and print the data in the table
print("\nQ1: CarSharing table (top 5 rows)")

# Get column names
cur.execute("PRAGMA table_info(CarSharing);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name
column_names = [column[1] for column in columns]

# Print column names
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator

cur.execute("SELECT * FROM CarSharing LIMIT 5;")
CarSharing = cur.fetchall()

for row in CarSharing:
    print({row})

# 1b.
# Create a backup table and copy the whole table into it.

cur.execute('''
CREATE TABLE CarSharing_backup AS 
SELECT * 
FROM CarSharing;
''')

# Fetch and print the data in the table
print("\nQ1: CarSharing back-up table (top 5 rows)")

# Get column names
cur.execute("PRAGMA table_info(CarSharing_backup);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name
column_names = [column[1] for column in columns]

# Print column names
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator

cur.execute("SELECT * FROM CarSharing_backup LIMIT 5;")
CarSharing_backup = cur.fetchall()

for row in CarSharing_backup:
    print({row})

# 2a.
# Add a column to the CarSharing table named "temp_category".
cur.execute("ALTER TABLE CarSharing ADD COLUMN temp_category TEXT(3);")

# 2b.
# Update the values in the temp_category.
cur.execute('''
    UPDATE CarSharing
    SET temp_category = (
        CASE
            WHEN temp_feel < 10 THEN 'Cold'
            WHEN temp_feel BETWEEN 10 AND 25 THEN 'Mild'
            ELSE 'Hot'
        END
    );
''')

# Fetch and print the data in the table
print("\nQ2: CarSharing updated table (top 5 rows)")

# Get column names
cur.execute("PRAGMA table_info(CarSharing);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name
column_names = [column[1] for column in columns]

# Print column names
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator

cur.execute("SELECT * FROM CarSharing LIMIT 5;")
CarSharing = cur.fetchall()

for row in CarSharing:
    print({row})

# 3a.
# Create another table named "temperature" featuring temp, temp_feel and temp_category columns.

cur.execute('''
    CREATE TABLE temperature (
        temp REAL,
        temp_feel REAL,
        temp_category TEXT
    );
''')

# Select the temp, temp_feel, and temp_category columns from the CarSharing table and insert them
# into the temperature table.
cur.execute('''
    INSERT INTO temperature (temp, temp_feel, temp_category)
    SELECT temp, temp_feel, temp_category
    FROM CarSharing;
''')

# Fetch and print the data in the table
print("\nQ3: Temperature table (top 5 rows)")

# Get column names
cur.execute("PRAGMA table_info(temperature);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name
column_names = [column[1] for column in columns]

# Print column names
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator

cur.execute("SELECT * FROM temperature LIMIT 5;")
temperature = cur.fetchall()

for row in temperature:
    print({row})

# 3b.
# Drop the temp and temp_feel columns from the CarSharing table using the executescript() method
# of the cursor object.
cur.executescript('''
BEGIN;

ALTER TABLE CarSharing 
DROP COLUMN temp;

ALTER TABLE CarSharing 
DROP COLUMN temp_feel;

COMMIT;
''')

# Fetch and print the data in the table
print("\nQ3: Altered CarSharing table (top 5 rows)")

# Get column names
cur.execute("PRAGMA table_info(CarSharing);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name
column_names = [column[1] for column in columns]

# Print column names
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator

# Print the first 5 rows of the table
cur.execute("SELECT * FROM CarSharing LIMIT 5;")
CarSharing = cur.fetchall()

for row in CarSharing:
    print({row})

# 4a.
# Find the distinct values of the weather column
cur.execute("SELECT DISTINCT weather FROM CarSharing;")

# Fetch the result.
distinct_weather = cur.fetchall()

# Print the distinct values.
print("\nQ4: Distinct weather of the weather column")
for row in distinct_weather:
    print(row[0])

# 4b.
# Assign a number to each weather value
weather_code = {}
for i, weather in enumerate(distinct_weather):
    weather_code[weather[0]] = i

# 4c.
# Add the weather_code column to the CarSharing table
cur.execute("ALTER TABLE CarSharing ADD COLUMN weather_code INTEGER;")

# Update the weather_code column based on the weather column
for weather, code in weather_code.items():
    cur.execute('''
    UPDATE CarSharing
    SET weather_code = ?
    WHERE weather = ?;
    ''', (code, weather))

# Fetch and print the data
print("\nQ4: CarSharing table updated with weather_code")

# Get column names
cur.execute("PRAGMA table_info(CarSharing);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name
column_names = [column[1] for column in columns]

# Print column names
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator

# Print the first 5 rows
cur.execute("SELECT * FROM CarSharing LIMIT 5;")
rows = cur.fetchall()
for row in rows:
    print(row)

# Alternatively, query the column information for the CarSharing table to confirm the newly added columns
cur.execute("PRAGMA table_info(CarSharing)")

# Fetch and print the column names
columns = cur.fetchall()
print("\nQ4: Column details of the updated CarSharing table")
for column in columns:
    print(f"{column[0]}: {column[1]}: {column[2]}")

# 5a.
# Create the weather table
cur.execute('''
CREATE TABLE weather(
weather TEXT,
weather_code INTEGER
);''')

# 5b.
# Copy the weather and weather_code's data from the CarSharing table into the table weather
cur.execute('''
INSERT INTO weather (weather, weather_code)
SELECT weather, weather_code 
FROM CarSharing;
''')

# 5c. Drop the weather column from the CarSharing table
cur.execute('''
ALTER TABLE CarSharing 
DROP COLUMN weather;
''')

# Query the column information for the CarSharing table to confirm if the column "weather" has been dropped
cur.execute("PRAGMA table_info(CarSharing)")

# Fetch and print the column names
columns = cur.fetchall()
print("\nQ5: Column details of the altered CarSharing table")
for column in columns:
    print(f"{column[0]}: {column[1]}: {column[2]}")

# 6a.
# Create the table named "time" with four columns
cur.execute('''
CREATE TABLE time(
    timestamp TIMESTAMP,
    hour INTEGER,
    weekday TEXT,
    month TEXT);
''')

# Insert rows into the table "time"
cur.execute('''
INSERT INTO time (timestamp, hour, weekday, month)
SELECT
    strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
    strftime('%H', timestamp) as hour,
    CASE strftime('%w', timestamp)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday' 
    END as weekday,
    CASE strftime('%m', timestamp) 
        WHEN '01' THEN 'January'
        WHEN '02' THEN 'February'
        WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'
        WHEN '05' THEN 'May'
        WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'
        WHEN '08' THEN 'August'
        WHEN '09' THEN 'September'
        WHEN '10' THEN 'October'
        WHEN '11' THEN 'November'
        WHEN '12' THEN 'December'
    END as month
FROM CarSharing; 
''')

# Fetch and print the data.
print("\nQ6: Table named 'time' with the columns; timestamp, hour, weekday and month.")

# Get column names.
cur.execute("PRAGMA table_info(time);")
columns = cur.fetchall()

# Get the second element of each tuple, which is the column name.
column_names = [column[1] for column in columns]

# Print column names.
print(", ".join(column_names))  # Join the column names with a 'comma with a space' separator.

# Query and print the first 5 rows of the table.
cur.execute("SELECT * FROM time LIMIT 5;")
rows = cur.fetchall()
for row in rows:
    print(row)

# 7a.
# Find the date and time with the highest demand rate in 2017.
cur.execute('''
SELECT t.timestamp, c.demand
FROM time AS t
INNER JOIN CarSharing AS c 
ON t.timestamp = c.timestamp
WHERE strftime('%Y', t.timestamp) = '2017'
ORDER BY c.demand DESC
LIMIT 1;
''')

# Fetch and print the result.
result = cur.fetchone()
print(f"\nQ7a:\nThe date and time with the highest demand rate in 2017 was {result[0]} with a demand rate of {result[1]}.")

# 7b.
# Provide a table containing the name of the weekday, month and season
# in which we had the highest and lowest average demand rates throughout 2017.

# 7bi.
# For the highest average demand rate throughout 2017, write a query that to call the
# executescript() method on and assign it to a variable.
query_highest_demand2017 = '''
BEGIN;

CREATE TABLE highest_demand2017(
    weekday TEXT,
    month TEXT,
    season TEXT,
    avg_demand REAL
);

INSERT INTO highest_demand2017
SELECT 
    CASE strftime('%w', timestamp)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday' 
    END as weekday,
    CASE strftime('%m', timestamp) 
        WHEN '01' THEN 'January'
        WHEN '02' THEN 'February'
        WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'
        WHEN '05' THEN 'May'
        WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'
        WHEN '08' THEN 'August'
        WHEN '09' THEN 'September'
        WHEN '10' THEN 'October'
        WHEN '11' THEN 'November'
        WHEN '12' THEN 'December'
    END AS month,
    season,
    AVG(demand) AS avg_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
GROUP BY weekday, month, season
ORDER BY avg_demand DESC
LIMIT 1;

COMMIT;
'''

# 7bii.
# For the lowest average demand rate throughout 2017, write a query that to call on the
# executescript() method and assign it to a variable it.
query_lowest_demand2017 = '''
BEGIN;

CREATE TABLE lowest_demand2017(
    weekday TEXT,
    month TEXT,
    season TEXT,
    avg_demand REAL
);

INSERT INTO lowest_demand2017
SELECT 
    CASE strftime('%w', timestamp)
        WHEN '0' THEN 'Sunday'
        WHEN '1' THEN 'Monday'
        WHEN '2' THEN 'Tuesday'
        WHEN '3' THEN 'Wednesday'
        WHEN '4' THEN 'Thursday'
        WHEN '5' THEN 'Friday'
        WHEN '6' THEN 'Saturday' 
    END as weekday,
    CASE strftime('%m', timestamp) 
        WHEN '01' THEN 'January'
        WHEN '02' THEN 'February'
        WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'
        WHEN '05' THEN 'May'
        WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'
        WHEN '08' THEN 'August'
        WHEN '09' THEN 'September'
        WHEN '10' THEN 'October'
        WHEN '11' THEN 'November'
        WHEN '12' THEN 'December'
    END AS month,
    season,
    avg(demand) as avg_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
GROUP BY weekday, month, season
ORDER BY avg_demand ASC
LIMIT 1;

COMMIT;
'''

# 7biii.
# Perform the executescript() method on the variable assigned to the query in 7bi.
cur.executescript(query_highest_demand2017)

# Fetch and print the data.
cur.execute("SELECT * FROM highest_demand2017")
highest_demand2017 = cur.fetchall()
print("\nQ7b: The highest demand rate in 2017 ")
for row in highest_demand2017:
    print(row)

# 7biv.
# Perform the executescript() method on the variable assigned to the query in 7bii.
cur.executescript(query_lowest_demand2017)

# Fetch and print the data.
cur.execute("SELECT * FROM lowest_demand2017")
lowest_demand2017 = cur.fetchall()
print("\nQ7b: The lowest demand rate in 2017 ")
for row in lowest_demand2017:
    print(row)

# 7ci.
# Give a table showing the avg demand rate on different hours of that weekday in 7biii when
# we have the highest demand rates throughout 2017.

# Firstly, fetch the weekday of the highest demand rate throughout 2017.
cur.execute("SELECT weekday FROM highest_demand2017")
highest_weekday = cur.fetchone()
print("\nQ7c: Weekday with highest demand rate in 2017 ")
for row in highest_weekday:
    print(row)

# Then, create the table to show the average demand rate for different hours of the weekday
highest_weekday = '0'
cur.execute("""
CREATE TABLE hour_highest_demand AS
SELECT strftime('%H', timestamp) AS hour,
       AVG(demand) AS avg_demand,
       CASE strftime('%w', timestamp)
           WHEN '0' THEN 'Sunday'
           WHEN '1' THEN 'Monday'
           WHEN '2' THEN 'Tuesday'
           WHEN '3' THEN 'Wednesday'
           WHEN '4' THEN 'Thursday'
           WHEN '5' THEN 'Friday'
           WHEN '6' THEN 'Saturday'
       END AS weekday
FROM CarSharing
WHERE strftime('%Y', timestamp) LIKE '2017%'
    AND strftime('%w', timestamp) = ?
GROUP BY hour, weekday
ORDER BY avg_demand DESC
""", highest_weekday,)

# Fetch and print the data in the table
cur.execute("SELECT * FROM hour_highest_demand")
hour_highest_demand = cur.fetchall()

print("\nQ7c: Hours and average demand from weekday with highest demand rate in 2017")
for row in hour_highest_demand:
    hour = row[0]
    avg_demand = row[1]
    print(f"Hour: {hour}, Average Demand: {avg_demand}")

# 7cii.
# Give a table showing the avg demand rate on different hours of that weekday in 7biv when
# we have the lowest demand rates throughout 2017.

# Fetch the weekday of the lowest demand rate throughout 2017.
cur.execute("SELECT weekday FROM lowest_demand2017")
lowest_weekday = cur.fetchone()
print("\nQ7c: Weekday with lowest demand rate in 2017 ")
for row in lowest_weekday:
    print(row)

# Then, create the table to show the average demand rate for different hours of the weekday
lowest_weekday = '1'
cur.execute("""
CREATE TABLE hour_lowest_demand AS
SELECT strftime('%H', timestamp) AS hour,
       AVG(demand) AS avg_demand,
       CASE strftime('%w', timestamp)
           WHEN '0' THEN 'Sunday'
           WHEN '1' THEN 'Monday'
           WHEN '2' THEN 'Tuesday'
           WHEN '3' THEN 'Wednesday'
           WHEN '4' THEN 'Thursday'
           WHEN '5' THEN 'Friday'
           WHEN '6' THEN 'Saturday'
       END AS weekday
FROM CarSharing
WHERE strftime('%Y', timestamp) LIKE '2017%'
    AND strftime('%w', timestamp) = ?
GROUP BY hour, weekday
ORDER BY avg_demand DESC
""", lowest_weekday,)

# Fetch and print the data in the table
cur.execute("SELECT * FROM hour_lowest_demand")
hour_lowest_demand = cur.fetchall()

print("\nQ7c: Hours and average demand from weekday with lowest demand rate in 2017")
for row in hour_lowest_demand:
    hour = row[0]
    avg_demand = row[1]
    print(f"Hour: {hour}, Average Demand: {avg_demand}")

# 7d.
# Nature and frequency (prevalence) of the weather condition in 2017 with reference to its
# temp_category being cold, mild or hot.

# 7di.
# Considering both the weather condition and temp_category in the grouping
cur.execute('''
CREATE TABLE weather_temp_condition AS
SELECT cb.weather, c.temp_category, COUNT(*) AS count
FROM CarSharing_backup cb
JOIN CarSharing c
ON c.id = cb.id
WHERE strftime('%Y', c.timestamp) LIKE '2017%'
GROUP BY cb.weather, c.temp_category
ORDER BY count DESC
''')

# Fetch and print the most prevalent weather_condition considering both the
# weather and temp_category in the table
cur.execute("SELECT * FROM weather_temp_condition")
row = cur.fetchall()
print(f"\nQ7d: \nConsidering both the weather and temp_category of weather condition,\n"
      f"the most prevalent weather condition is one with \n{row[0]}")

# 7dii.
# Considering only the weather condition in the grouping
cur.execute('''
CREATE TABLE weather_condition AS
SELECT cb.weather, c.temp_category, COUNT(*) AS count
FROM CarSharing_backup cb
JOIN CarSharing c
ON c.id = cb.id
WHERE strftime('%Y', c.timestamp) LIKE '2017%'
GROUP BY cb.weather
ORDER BY count DESC
''')

# Fetch and print the most prevalent weather_condition considering only the weather and
# temp_category in the table
cur.execute("SELECT * FROM weather_condition")
row = cur.fetchall()
print(f"\nConsidering only the weather component of the weather condition,\n"
      f"the most prevalent weather condition is one with \n{row[0]}")

# 7diii.
# Table for average, highest and lowest wind speed for each month in 2017.
cur.execute("""
CREATE TABLE windspeed_summary AS
SELECT strftime('%m', timestamp) AS month, AVG(windspeed) AS avg_windspeed, MAX(windspeed) AS max_windspeed, MIN(windspeed) AS min_windspeed
FROM CarSharing
WHERE strftime('%Y', timestamp) LIKE '2017%'
AND windspeed != ""
GROUP BY month
""")

# Fetch and print the data in the table
cur.execute("SELECT * FROM windspeed_summary")
windspeed_summary = cur.fetchall()

print("\nQ7d: Average, minimum and maximum summary of windspeed for each month in 2017")
for row in windspeed_summary:
    print({row})

# 7div.
# Table for average, highest and lowest humidity for each month in 2017.
cur.execute("""
CREATE TABLE humidity_summary AS
SELECT strftime('%m', timestamp) AS month, AVG(humidity) AS avg_humidity, MAX(humidity) AS max_humidity, MIN(humidity) AS min_humidity
FROM CarSharing
WHERE strftime('%Y', timestamp) LIKE '2017%'
AND humidity != ""
GROUP BY month
""")

# Fetch and print the data in the table
cur.execute("SELECT * FROM humidity_summary")
humidity_summary = cur.fetchall()

print("\nQ7d: Average, minimum and maximum summary of humidity for each month in 2017")
for row in humidity_summary:
    print({row})

# 7dv.
# Create table showing the average demand rate for each cold, mild and hot weather in 2017
# sorted in descending order based on their average demand rates.
cur.execute("""
CREATE TABLE temp_category_demand AS
SELECT temp_category, AVG(demand) AS avg_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) LIKE '2017%'
GROUP BY temp_category
ORDER BY avg_demand DESC
""")

# Fetch and print the data in the table
cur.execute("SELECT * FROM temp_category_demand")
temp_category_demand = cur.fetchall()

print("\nQ7d: Average demand rate for each cold, mild and hot weather in 2017")
for row in temp_category_demand:
    print({row})

# 7e.
# Create table showing the information in 7d for the month with the highest average demand rate
# in 2017 and compare it with other months

# 7ei.
# Create the table to show the information summary for all the months in year 2017.
cur.execute("""
CREATE TABLE info_summary_demand2017 AS
SELECT c.temp_category, cb.weather, cb.humidity, cb.windspeed, AVG(c.demand) AS avg_demand, 
    CASE strftime('%m', c.timestamp)
        WHEN '01' THEN 'January'
        WHEN '02' THEN 'February'
        WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'
        WHEN '05' THEN 'May'
        WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'
        WHEN '08' THEN 'August'
        WHEN '09' THEN 'September'
        WHEN '10' THEN 'October'
        WHEN '11' THEN 'November'
        WHEN '12' THEN 'December'
    END AS month
FROM CarSharing_backup cb
JOIN CarSharing c
ON c.id = cb.id
WHERE strftime('%Y', c.timestamp) LIKE '2017%'
GROUP BY month
ORDER BY avg_demand DESC
""")

# Firstly, fetch the month of the highest demand rate throughout 2017.
cur.execute("SELECT month FROM highest_demand2017")
highest_month = cur.fetchone()
print("\nQ7e: Month with highest demand rate in 2017 ")
for row in highest_month:
    print(row)

# Fetch the table to show the information summary for all the months in year 2017 for comparison.
cur.execute("SELECT * FROM info_summary_demand2017 WHERE month = ?", highest_month,)
highest_month_summary_demand2017 = cur.fetchall()
print("\nQ7e: Information summary for the month with highest demand in year 2017.")
for row in highest_month_summary_demand2017:
    print(row)

# Fetch the table to show the information summary for all the months in year 2017 for comparison.
cur.execute("SELECT * FROM info_summary_demand2017")
info_summary_demand2017 = cur.fetchall()
print("\nQ7e: Information summary for all the months in year 2017.")
for row in info_summary_demand2017:
    print(row)