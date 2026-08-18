from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col, to_timestamp, dayofweek, month, avg, max, min

# Start Spark session
spark = SparkSession.builder.appName("BigDataAnalysis").getOrCreate()

# Load the dataset
df = spark.read.csv("project1_df.csv", header=True, inferSchema=True)

# Convert string to timestamp
df = df.withColumn("Purchase Date", to_timestamp(col("Purchase Date"), "dd/MM/yyyy HH:mm:ss"))

# Extract hour, day of week, and month
df = df.withColumn("purchase_hour", hour(col("Purchase Date")))
df = df.withColumn("purchase_day", dayofweek(col("Purchase Date")))
df = df.withColumn("purchase_month", month(col("Purchase Date")))

# Calculate average net amount by hour
hourly_avg = df.groupBy("purchase_hour").agg(avg("Net Amount").alias("avg_net_amount"))
hourly_avg = hourly_avg.orderBy("purchase_hour")
hourly_avg.show()

# Average net amount by day of the week
day_avg = df.groupBy("purchase_day").agg(avg("Net Amount").alias("avg_net_amount_day"))
day_avg = day_avg.orderBy("purchase_day")
day_avg.show()

# Average net amount by month
monthly_avg = df.groupBy("purchase_month").agg(avg("Net Amount").alias("avg_net_amount_month"))
monthly_avg = monthly_avg.orderBy("purchase_month")
monthly_avg.show()

# Maximum and minimum purchases per category
category_stats = df.groupBy("Product Category").agg(
    max("Net Amount").alias("max_net_amount"),
    min("Net Amount").alias("min_net_amount"),
    avg("Net Amount").alias("avg_net_amount")
)
category_stats = category_stats.orderBy("avg_net_amount", ascending=False)
category_stats.show()

# Top 10 highest purchases
top_purchases = df.orderBy(col("Net Amount").desc()).limit(10)
top_purchases.show()

# Top 5 purchase locations by frequency
location_freq = df.groupBy("Location").count().orderBy("count", ascending=False)
location_freq.show(5)

# Most used purchase method
purchase_method_freq = df.groupBy("Purchase Method").count().orderBy("count", ascending=False)
purchase_method_freq.show()

import matplotlib.pyplot as plt

#chart 1 : Convert Spark DF to Pandas for plotting
pdf = hourly_avg.toPandas()

plt.figure(figsize=(10, 6))
plt.plot(pdf["purchase_hour"], pdf["avg_net_amount"], marker='o', color='blue')
plt.title("Average Net Purchase Amount by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Net Amount (INR)")
plt.grid(True)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("chart_hourly_net_amount.png")
plt.show()


#Chart 2: Average Net Amount by Day of Week
day_avg = df.groupBy("purchase_day").agg(avg("Net Amount").alias("avg_net_amount_day"))
pdf_day = day_avg.toPandas()

plt.figure(figsize=(10, 6))
plt.bar(pdf_day["purchase_day"], pdf_day["avg_net_amount_day"], color='orange')
plt.title("Average Net Amount by Day of Week (1=Sun, 7=Sat)")
plt.xlabel("Day of Week")
plt.ylabel("Average Net Amount (INR)")
plt.tight_layout()
plt.savefig("chart_day_avg.png")
plt.show()

#Chart 3: Average Net Amount by Month
month_avg = df.groupBy("purchase_month").agg(avg("Net Amount").alias("avg_net_amount_month"))
pdf_month = month_avg.toPandas()

plt.figure(figsize=(10, 6))
plt.plot(pdf_month["purchase_month"], pdf_month["avg_net_amount_month"], marker='o', color='green')
plt.title("Average Net Amount by Month")
plt.xlabel("Month")
plt.ylabel("Average Net Amount (INR)")
plt.tight_layout()
plt.savefig("chart_month_avg.png")
plt.show()

#Chart 4: Avg Net Amount by Product Category
category_stats = df.groupBy("Product Category").agg(avg("Net Amount").alias("avg_net_amount_cat"))
pdf_cat = category_stats.toPandas()

plt.figure(figsize=(12, 6))
plt.barh(pdf_cat["Product Category"], pdf_cat["avg_net_amount_cat"], color='purple')
plt.title("Average Net Amount by Product Category")
plt.xlabel("Avg Net Amount (INR)")
plt.tight_layout()
plt.savefig("chart_category_avg.png")
plt.show()


#Chart 5: Top 10 Locations by Purchase Count
location_freq = df.groupBy("Location").count().orderBy("count", ascending=False).limit(10)
pdf_loc = location_freq.toPandas()

plt.figure(figsize=(12, 6))
plt.bar(pdf_loc["Location"], pdf_loc["count"], color='teal')
plt.title("Top 10 Locations by Purchase Count")
plt.xlabel("Location")
plt.ylabel("Number of Purchases")
plt.tight_layout()
plt.savefig("chart_top_locations.png")
plt.show()

#Chart 6: Purchase Method Usage (Bar Chart)
payment_freq = df.groupBy("Purchase Method").count().orderBy("count", ascending=False)
pdf_pay = payment_freq.toPandas()

plt.figure(figsize=(10, 6))
plt.barh(pdf_pay["Purchase Method"], pdf_pay["count"], color='brown')
plt.title("Purchase Method Frequency")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("chart_payment_method.png")
plt.show()

# Stop Spark session
spark.stop()