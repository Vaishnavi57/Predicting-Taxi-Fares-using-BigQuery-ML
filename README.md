## **Objective**

- Predict taxi fares in New York City using **BigQuery ML**.
- Build a predictive model based on historical NYC taxi trip data.
- Use key features like:
  - Time of day
  - Pickup and dropoff locations
  - Passenger count
- Improve fare prediction accuracy through:
  - Data preprocessing
  - Model tuning
- Provide a foundation for more complex systems such as:
  - Dynamic pricing
  - Route optimization
  - Real-time fare prediction for ride-hailing services

## **Dataset**
- The dataset used for this project is the **NYC Yellow Taxi Trip Data**, which contains detailed information about taxi rides in New York City. The dataset is available publicly through **Google Cloud's BigQuery Public Datasets** and can be accessed directly within BigQuery.
- (https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=nyc_tlc&t=yellow_tripdata&page=table)

##  **Step 1: Explore NYC Taxi Cab Data**

In this step, you will query the NYC Yellow Taxi dataset to find out how many trips Yellow taxis took each month in 2015.

### SQL Query:
```sql
#standardSQL
SELECT
  TIMESTAMP_TRUNC(pickup_datetime, MONTH) AS month,
  COUNT(*) AS trips
FROM
  `bigquery-public-data.new_york.tlc_yellow_trips_2015`
GROUP BY
  month
ORDER BY
  month
```

##  **Step 2: Calculate the Average Speed of Yellow Taxi Trips in 2015**

In this step, we will calculate the average speed of Yellow taxi trips in 2015. The average speed is calculated using the trip distance and the time difference between pickup and dropoff.

### 1. **SQL Query**:
   Copy and paste the following SQL code into the BigQuery query editor:

   ```sql
   #standardSQL
   SELECT
     EXTRACT(HOUR FROM pickup_datetime) AS hour,
     ROUND(AVG(trip_distance / TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND)) * 3600, 1) AS speed
   FROM
     `bigquery-public-data.new_york.tlc_yellow_trips_2015`
   WHERE
     trip_distance > 0
     AND fare_amount / trip_distance BETWEEN 2 AND 10
     AND dropoff_datetime > pickup_datetime
   GROUP BY
     hour
   ORDER BY
     hour
```
##  **Step 3: Select Features and Create Your Training Dataset**

In this step, we will select useful features from the **NYC Yellow Taxi Dataset** and create a training dataset to forecast taxi fares. The selected features will help the machine learning model understand the relationship between historical cab ride data and fare prices.

### 1. **SQL Query**:
   Copy and paste the following SQL code into the BigQuery query editor:

   ```sql
   #standardSQL
   WITH params AS (
       SELECT
         1 AS TRAIN,
         2 AS EVAL
   ),
   
   daynames AS (
       SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek
   ),
   
   taxitrips AS (
   SELECT
     (tolls_amount + fare_amount) AS total_fare,
     daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
     EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
     pickup_longitude AS pickuplon,
     pickup_latitude AS pickuplat,
     dropoff_longitude AS dropofflon,
     dropoff_latitude AS dropofflat,
     passenger_count AS passengers
   FROM
     `nyc-tlc.yellow.trips`, daynames, params
   WHERE
     trip_distance > 0 AND fare_amount > 0
     AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.TRAIN
   )
   
   SELECT *
   FROM taxitrips
```
### Explanation of the SQL Query:

- **Features Selected**: 
   - We are selecting features like `total_fare`, `dayofweek`, `hourofday`, pickup and dropoff latitudes/longitudes, and `passengers` to predict the taxi fare.
  
- **Sampling and Split**:
   - The query uses a **sampling clause** to pick a small random subset of data, and the `TRAIN` variable is used to split the dataset into training and evaluation datasets.
  
- **Grouping and Transformation**:
   - `EXTRACT` is used to extract the day of the week and hour of the day from the `pickup_datetime`.
   - 
##  **Step 4: Creating a BigQuery Dataset to Store Models**

A **BigQuery dataset** named **`taxi`** was created to store the machine learning models for the taxi fare prediction project. This dataset will be used to store models and other relevant outputs throughout the project.

## 🔧 **Creating a Machine Learning Model for Taxi Fare Prediction**

To build a model for predicting taxi fares, a **linear regression model** was created using the **`taxi.taxifare_model`** dataset. The model is trained using historical taxi trip data, with the **`total_fare`** as the target label. 

### SQL Query to Create the Model:
The following SQL query was executed to create and train the model:

```sql
CREATE OR REPLACE MODEL taxi.taxifare_model
OPTIONS
  (model_type='linear_reg', labels=['total_fare']) AS

WITH params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL
),
  
daynames AS (
    SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek
),

taxitrips AS (
  SELECT
    (tolls_amount + fare_amount) AS total_fare,
    daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
    EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
    pickup_longitude AS pickuplon,
    pickup_latitude AS pickuplat,
    dropoff_longitude AS dropofflon,
    dropoff_latitude AS dropofflat,
    passenger_count AS passengers
  FROM
    `nyc-tlc.yellow.trips`, daynames, params
  WHERE
    trip_distance > 0 AND fare_amount > 0
    AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.TRAIN
)

SELECT *
FROM taxitrips
```
## **Step 5: Evaluating Model Performance**

After training the `taxifare_model`, the next step involved evaluating its performance on unseen data to determine how well it generalizes. The evaluation focused on the **Root Mean Squared Error (RMSE)**, a commonly used metric for regression tasks that indicates how far predictions deviate from actual values.

BigQuery ML provides `ML.EVALUATE` for this purpose. The RMSE was calculated using the square root of the `mean_squared_error` field returned from the evaluation.

### SQL Query for Evaluation:

```sql
#standardSQL
SELECT
  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL taxi.taxifare_model,
  (
    WITH params AS (
      SELECT 1 AS TRAIN, 2 AS EVAL
    ),
  
    daynames AS (
      SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek
    ),
  
    taxitrips AS (
      SELECT
        (tolls_amount + fare_amount) AS total_fare,
        daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
        EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
        pickup_longitude AS pickuplon,
        pickup_latitude AS pickuplat,
        dropoff_longitude AS dropofflon,
        dropoff_latitude AS dropofflat,
        passenger_count AS passengers
      FROM
        `nyc-tlc.yellow.trips`, daynames, params
      WHERE
        trip_distance > 0 AND fare_amount > 0
        AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.EVAL
    )
  
    SELECT * FROM taxitrips
  ))

##  **Predicting Taxi Fare Amounts**

With the model trained and evaluated, the final step involved generating fare predictions using the **`taxifare_model`**. Predictions were made on evaluation data that was not used during training, ensuring unbiased results.

BigQuery ML’s `ML.PREDICT` function was used to apply the model to new data and return predicted fare amounts along with the original features.

### SQL Query for Prediction:

```sql
#standardSQL
SELECT
  *
FROM
  ML.PREDICT(MODEL `taxi.taxifare_model`,
  (
    WITH params AS (
      SELECT 1 AS TRAIN, 2 AS EVAL
    ),
  
    daynames AS (
      SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek
    ),
  
    taxitrips AS (
      SELECT
        (tolls_amount + fare_amount) AS total_fare,
        daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
        EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
        pickup_longitude AS pickuplon,
        pickup_latitude AS pickuplat,
        dropoff_longitude AS dropofflon,
        dropoff_latitude AS dropofflat,
        passenger_count AS passengers
      FROM
        `nyc-tlc.yellow.trips`, daynames, params
      WHERE
        trip_distance > 0 AND fare_amount > 0
        AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.EVAL
    )
  
    SELECT * FROM taxitrips
  ))
```
##  **Step 5: Improving the Model with Feature Engineering**

Machine learning model development is an iterative process. After evaluating the performance of the initial model, steps were taken to refine the input data by applying **feature engineering** and **row-level filtering**. The goal was to improve model accuracy by reducing noise and focusing on more consistent and reliable data points.

A statistical analysis of the fare data was performed to understand its distribution and identify potential outliers. This helped in setting thresholds for more relevant training data.

### SQL Query for Fare Distribution Analysis:

```sql
SELECT
  COUNT(fare_amount) AS num_fares,
  MIN(fare_amount) AS low_fare,
  MAX(fare_amount) AS high_fare,
  AVG(fare_amount) AS avg_fare,
  STDDEV(fare_amount) AS stddev
FROM
  `nyc-tlc.yellow.trips`
WHERE 
  trip_distance > 0 
  AND fare_amount BETWEEN 6 AND 200
  AND pickup_longitude > -75
  AND pickup_longitude < -73
  AND dropoff_longitude > -75
  AND dropoff_longitude < -73
  AND pickup_latitude > 40
  AND pickup_latitude < 42
  AND dropoff_latitude > 40
  AND dropoff_latitude < 42
```
##  **Step 6: Retraining the Model with Refined Features**

Following the initial evaluation and data refinement, the model was retrained using an updated dataset and enhanced feature set. The new model, **`taxi.taxifare_model_2`**, aims to improve prediction accuracy by incorporating calculated features such as **Euclidean distance** between pickup and drop-off locations.

### Key Enhancements:
- Introduced new features:  
  - `dist`: Straight-line (Euclidean) distance  
  - `longitude`: Distance in longitude  
  - `latitude`: Distance in latitude  
- Filtered dataset to include only relevant geographic and fare ranges.
- Continued using **linear regression** to predict the `total_fare`.

### SQL Query to Retrain the Model:

```sql
CREATE OR REPLACE MODEL taxi.taxifare_model_2
OPTIONS
  (model_type='linear_reg', labels=['total_fare']) AS

WITH params AS (
    SELECT 1 AS TRAIN, 2 AS EVAL
),
daynames AS (
    SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek
),
taxitrips AS (
  SELECT
    (tolls_amount + fare_amount) AS total_fare,
    daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
    EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
    SQRT(POW((pickup_longitude - dropoff_longitude), 2) + POW((pickup_latitude - dropoff_latitude), 2)) AS dist,
    SQRT(POW((pickup_longitude - dropoff_longitude), 2)) AS longitude,
    SQRT(POW((pickup_latitude - dropoff_latitude), 2)) AS latitude,
    passenger_count AS passengers
  FROM
    `nyc-tlc.yellow.trips`, daynames, params
  WHERE 
    trip_distance > 0 AND fare_amount BETWEEN 6 AND 200
    AND pickup_longitude > -75 AND pickup_longitude < -73
    AND dropoff_longitude > -75 AND dropoff_longitude < -73
    AND pickup_latitude > 40 AND pickup_latitude < 42
    AND dropoff_latitude > 40 AND dropoff_latitude < 42
    AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.TRAIN
)

SELECT * FROM taxitrips
```
## **Step 7: Evaluating the Improved Model**

After retraining the model using refined features and filtered data, the updated model `taxifare_model_2` was evaluated to measure performance improvements. As with the previous model, **Root Mean Squared Error (RMSE)** was used to assess prediction accuracy.

### SQL Query to Evaluate the New Model:

```sql
SELECT
  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL taxi.taxifare_model_2,
  (
    WITH params AS (
      SELECT 1 AS TRAIN, 2 AS EVAL
    ),
    daynames AS (
      SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek
    ),
    taxitrips AS (
      SELECT
        (tolls_amount + fare_amount) AS total_fare,
        daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
        EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
        SQRT(POW((pickup_longitude - dropoff_longitude), 2) + POW((pickup_latitude - dropoff_latitude), 2)) AS dist,
        SQRT(POW((pickup_longitude - dropoff_longitude), 2)) AS longitude,
        SQRT(POW((pickup_latitude - dropoff_latitude), 2)) AS latitude,
        passenger_count AS passengers
      FROM
        `nyc-tlc.yellow.trips`, daynames, params
      WHERE 
        trip_distance > 0 AND fare_amount BETWEEN 6 AND 200
        AND pickup_longitude > -75 AND pickup_longitude < -73
        AND dropoff_longitude > -75 AND dropoff_longitude < -73
        AND pickup_latitude > 40 AND pickup_latitude < 42
        AND dropoff_latitude > 40 AND dropoff_latitude < 42
        AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.EVAL
    )
    SELECT * FROM taxitrips
  ))
```

