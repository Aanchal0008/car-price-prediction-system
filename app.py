from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
import os
import sys

app = Flask(__name__)

# ✅ Set Python interpreter path explicitly
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ✅ Start Spark Session
spark = SparkSession.builder \
    .appName("CarPricePrediction") \
    .master("local[*]") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.network.timeout", "300s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

# ✅ Load and preprocess data once
data_path = "Car details v3.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

df = df.dropna(subset=["name", "year", "selling_price", "km_driven", "fuel", "seller_type", 
                       "transmission", "owner", "mileage", "engine", "max_power", "seats"])

df = df.withColumn("mileage_kmpl", df["mileage"].substr(1, 5).cast("float"))
df = df.withColumn("engine_cc", df["engine"].substr(1, 4).cast("float"))
df = df.withColumn("max_power_bhp", df["max_power"].substr(1, 5).cast("float"))

df = df.select("year", "km_driven", "fuel", "seller_type", "transmission", "owner",
               "mileage_kmpl", "engine_cc", "max_power_bhp", "seats", "selling_price")

# ✅ Build pipeline components
imputer = Imputer(
    inputCols=["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"],
    outputCols=["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]
)

categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in categorical_cols]

assembler_inputs = ["year", "km_driven", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats"] + [col + "_vec" for col in categorical_cols]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="selling_price", maxIter=10)

pipeline = Pipeline(stages=indexers + encoders + [imputer, assembler, gbt])

# ✅ Train model on first request
model = None

@app.before_request
def before_request():
    global model
    model = pipeline.fit(df)

# ✅ Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            "year": int(request.form['year']),
            "km_driven": int(request.form['km_driven']),
            "fuel": request.form['fuel'],
            "seller_type": request.form['seller_type'],
            "transmission": request.form['transmission'],
            "owner": request.form['owner'],
            "mileage_kmpl": float(request.form['mileage']),
            "engine_cc": float(request.form['engine']),
            "max_power_bhp": float(request.form['max_power']),
            "seats": int(request.form['seats']),
        }

        input_df = spark.createDataFrame([input_data])
        prediction = model.transform(input_df).collect()[0]["prediction"]

        return render_template("result.html", price=round(prediction, 2))

    except Exception as e:
        return render_template("result.html", price="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
