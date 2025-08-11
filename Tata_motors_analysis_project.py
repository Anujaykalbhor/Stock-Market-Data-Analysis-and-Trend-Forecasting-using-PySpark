# Databricks notebook source
# MAGIC %md
# MAGIC Step 1: Reload the file (to get raw data)

# COMMAND ----------

# MAGIC %md
# MAGIC  STEP 1: Reload cleanly with all columns as strings

# COMMAND ----------

file_path = "/FileStore/tables/TATAMOTORS-3.csv"

df_raw = spark.read.option("header", True).option("inferSchema", False).csv(file_path)


# COMMAND ----------

# MAGIC %md
# MAGIC  STEP 2: Remove the invalid row (second header row)

# COMMAND ----------

df_cleaned = df_raw.filter(~df_raw["Close"].contains("TATAMOTORS.NS"))


# COMMAND ----------

# MAGIC %md
# MAGIC STEP 3: Rename Price to DateString

# COMMAND ----------

df_cleaned = df_cleaned.withColumnRenamed("Price", "DateString")


# COMMAND ----------

# MAGIC %md
# MAGIC STEP 4: Extract date using substring

# COMMAND ----------

from pyspark.sql.functions import substring, col

df_cleaned = df_cleaned.withColumn("Date", substring("DateString", 1, 10))


# COMMAND ----------

# MAGIC %md
# MAGIC  STEP 5: Convert string to proper date format

# COMMAND ----------

from pyspark.sql.functions import to_date

df_cleaned = df_cleaned.withColumn("Date", to_date("Date", "yyyy-MM-dd"))


# COMMAND ----------

# MAGIC %md
# MAGIC STEP 6: Cast numeric columns

# COMMAND ----------

numeric_columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
for c in numeric_columns:
    df_cleaned = df_cleaned.withColumn(c, col(c).cast("double"))


# COMMAND ----------

df_cleaned.select("Date", "Close").show(5)
df_cleaned.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC Step 1: Analyze Daily Returns
# MAGIC

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, col, round

# Define a window specification ordered by Date
window_spec = Window.orderBy("Date")

# Use lag to get the previous day's Close price
df_with_prev = df_cleaned.withColumn("Prev_Close", lag("Close").over(window_spec))

# Calculate Daily_Return as percentage change
df_with_return = df_with_prev.withColumn(
    "Daily_Return",
    round(((col("Close") - col("Prev_Close")) / col("Prev_Close")) * 100, 4)  # rounded to 4 decimals
)

# Show the result
df_with_return.select("Date", "Close", "Prev_Close", "Daily_Return").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC  Step 2: Compute 50-day and 200-day Moving Averages

# COMMAND ----------

from pyspark.sql.functions import avg

# Define moving average windows ordered by Date
window_50 = Window.orderBy("Date").rowsBetween(-49, 0)    # 50-day window
window_200 = Window.orderBy("Date").rowsBetween(-199, 0)  # 200-day window

# Compute SMAs
df_with_sma = df_with_return \
    .withColumn("SMA_50", avg("Close").over(window_50)) \
    .withColumn("SMA_200", avg("Close").over(window_200))

# Show sample output
df_with_sma.select("Date", "Close", "SMA_50", "SMA_200").show(10)


# COMMAND ----------

df_with_sma.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3: Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Show summary stats for key columns

# COMMAND ----------

# Describe basic stats
df_with_sma.select("Open", "Close", "Volume", "Daily_Return").describe().show()


# COMMAND ----------

# MAGIC %md
# MAGIC  3.2 Find high-volume days and analyze price movement
# MAGIC  

# COMMAND ----------

# Top 10 highest volume days
df_with_sma.orderBy(col("Volume").desc()).select("Date", "Volume", "Close").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC  3.3 Identify volatility using 30-day rolling standard deviation

# COMMAND ----------

from pyspark.sql.functions import stddev

# 30-day rolling window for standard deviation of Daily_Return
window_30 = Window.orderBy("Date").rowsBetween(-29, 0)

df_volatility = df_with_sma.withColumn("Volatility_30", stddev("Daily_Return").over(window_30))

# View volatility
df_volatility.select("Date", "Daily_Return", "Volatility_30").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC  3.4 Perform trend analysis using SMA crossovers (Golden/Death Cross)

# COMMAND ----------

from pyspark.sql.functions import lag, when

# Detect crossover events
df_trend = df_volatility.withColumn("Prev_SMA_50", lag("SMA_50").over(window_spec)) \
    .withColumn("Prev_SMA_200", lag("SMA_200").over(window_spec)) \
    .withColumn("Trend_Signal", when(
        (col("Prev_SMA_50") < col("Prev_SMA_200")) & (col("SMA_50") >= col("SMA_200")), "Golden Cross"
    ).when(
        (col("Prev_SMA_50") > col("Prev_SMA_200")) & (col("SMA_50") <= col("SMA_200")), "Death Cross"
    ).otherwise(None))

# Show crossover signals
df_trend.filter(col("Trend_Signal").isNotNull()).select("Date", "SMA_50", "SMA_200", "Trend_Signal").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC  Step 4: Correlation and Seasonal Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC  4.1 Calculate correlation between Close and Volume

# COMMAND ----------

# Correlation between Close and Volume
correlation = df_trend.stat.corr("Close", "Volume")
print(f"Correlation between Close and Volume: {correlation}")


# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Analyze monthly average prices and volume (seasonality)
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import year, month

# Add Year and Month columns
df_seasonal = df_trend.withColumn("Year", year("Date")).withColumn("Month", month("Date"))

# Group by Year and Month and get average Close and Volume
monthly_stats = df_seasonal.groupBy("Year", "Month") \
    .agg(
        avg("Close").alias("Avg_Close"),
        avg("Volume").alias("Avg_Volume")
    ) \
    .orderBy("Year", "Month")

# Show monthly trends
monthly_stats.show(12)


# COMMAND ----------

# MAGIC %md
# MAGIC 4.3 Detect anomalous Daily_Returns using mean ± 3 * stddev
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import mean, stddev

# Compute mean and stddev of Daily_Return
stats = df_seasonal.select(mean("Daily_Return").alias("mean_return"),
                           stddev("Daily_Return").alias("stddev_return")).collect()[0]

mean_val = stats["mean_return"]
stddev_val = stats["stddev_return"]

# Filter for anomalies (returns outside mean ± 3 * stddev)
anomalies = df_seasonal.filter(
    (col("Daily_Return") > mean_val + 3 * stddev_val) |
    (col("Daily_Return") < mean_val - 3 * stddev_val)
)

# Show anomalous return days
anomalies.select("Date", "Daily_Return", "Close", "Volume").orderBy("Date").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Export data to Pandas for forecasting

# COMMAND ----------

# Convert to Pandas DataFrame for modeling (if using smaller sample)
df_pandas = df_seasonal.select("Date", "Close").orderBy("Date").toPandas()
df_pandas.set_index("Date", inplace=True)


# COMMAND ----------

# MAGIC %md
# MAGIC  ARIMA Model using statsmodels

# COMMAND ----------

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fit ARIMA model
model = ARIMA(df_pandas["Close"], order=(5,1,0))  # (p,d,q) can be tuned
model_fit = model.fit()

# Forecast the next 30 days
forecast = model_fit.forecast(steps=30)

# Plot
plt.figure(figsize=(12,6))
plt.plot(df_pandas["Close"], label="Historical")
plt.plot(forecast.index, forecast, label="Forecast", color='red')
plt.legend()
plt.title("ARIMA Forecast - Close Prices")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Step 6: Strategic Insights and Visualization

# COMMAND ----------

# MAGIC %md 6.1 Identify Price Trends: Uptrend, Downtrend, Sideways
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import when

df_insights = df_trend.withColumn("Market_Trend", when(
    (col("Close") > col("SMA_50")) & (col("SMA_50") > col("SMA_200")), "Uptrend"
).when(
    (col("Close") < col("SMA_50")) & (col("SMA_50") < col("SMA_200")), "Downtrend"
).otherwise("Sideways"))

# Show sample trend classification
df_insights.select("Date", "Close", "SMA_50", "SMA_200", "Market_Trend").orderBy("Date").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC 6.2 High Volatility Detection
# MAGIC

# COMMAND ----------

df_insights = df_insights.withColumn("High_Volatility_Flag", when(col("Volatility_30") > 5.0, "Yes").otherwise("No"))

# Show where high volatility occurs
df_insights.select("Date", "Volatility_30", "High_Volatility_Flag").filter(col("High_Volatility_Flag") == "Yes").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC 6.3 Visualizations (if using Databricks)

# COMMAND ----------

# Create a temporary table
df_insights.createOrReplaceTempView("stock_insights")




# COMMAND ----------

# MAGIC %md
# MAGIC Volume vs Close

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Run this SQL:
# MAGIC SELECT Date, Close, Volume FROM stock_insights ORDER BY Date
# MAGIC -- Then visualize as Line or Area Chart
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Daily Returns Distribution

# COMMAND ----------

# Display Daily_Return as histogram (Databricks built-in display)
display(df_insights.select("Daily_Return"))


# COMMAND ----------

# MAGIC %md
# MAGIC Step 7: Presentation and Reporting
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 7.1 Export Cleaned and Enriched Data

# COMMAND ----------

# Save as CSV to Databricks FileStore
output_path = "/FileStore/tables/tatamotors_final_analysis.csv"
df_insights.coalesce(1).write.option("header", True).mode("overwrite").csv(output_path)


# COMMAND ----------

# MAGIC %md
# MAGIC 7.2 Visual Summary with SQL or Display()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use this in a SQL cell
# MAGIC SELECT Date, Close, SMA_50, SMA_200, Market_Trend FROM stock_insights ORDER BY Date
# MAGIC

# COMMAND ----------

