import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
    IsotonicRegression,
)
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt


spark = SparkSession.builder.appName("FlightDelayPrediction").getOrCreate()

def preprocess_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    for col_name in df.columns:
        clean_name = col_name.replace('"', '')
        df = df.withColumnRenamed(col_name, clean_name)

    numeric_cols = [
        col for col, dtype in df.dtypes if dtype in ("double", "int") and col not in ["DEP_DELAY", "ARR_DELAY"]
    ]
    imputer = Imputer(inputCols=numeric_cols, outputCols=[f"{col}_imputed" for col in numeric_cols])
    df = imputer.fit(df).transform(df)

    df = df.withColumn(
        "TOTAL_DELAY",
        when(col("DEP_DELAY").isNull(), 0).otherwise(col("DEP_DELAY")) +
        when(col("ARR_DELAY").isNull(), 0).otherwise(col("ARR_DELAY")),
    )

    imputed_cols = [f"{col}_imputed" for col in numeric_cols]
    assembler = VectorAssembler(inputCols=imputed_cols, outputCol="features")
    df = assembler.transform(df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    df = scaler.fit(df).transform(df)

    return df


def predict_delay(file_path):
    df = preprocess_data(file_path)

    evaluator_rmse = RegressionEvaluator(labelCol="TOTAL_DELAY", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="TOTAL_DELAY", predictionCol="prediction", metricName="r2")

    models = {
        "Linear Regression": LinearRegression(featuresCol="scaled_features", labelCol="TOTAL_DELAY"),
        "Decision Tree Regressor": DecisionTreeRegressor(featuresCol="scaled_features", labelCol="TOTAL_DELAY"),
        "Random Forest Regressor": RandomForestRegressor(featuresCol="scaled_features", labelCol="TOTAL_DELAY"),
        "Gradient Boosting Regressor": GBTRegressor(featuresCol="scaled_features", labelCol="TOTAL_DELAY"),
        "Isotonic Regression": IsotonicRegression(featuresCol="scaled_features", labelCol="TOTAL_DELAY"),
    }

    results = {}
    for model_name, model in models.items():
        try:
            model_fitted = model.fit(df)
            predictions = model_fitted.transform(df)
            rmse = evaluator_rmse.evaluate(predictions)
            r2 = evaluator_r2.evaluate(predictions)
            results[model_name] = {"RMSE": rmse, "R²": r2}
        except Exception as e:
            results[model_name] = {"Error": str(e)}

    return results


def main():
    st.title("Flight Delay Prediction")
    st.write("Upload a dataset to evaluate regression models.")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        file_path = f"./uploaded_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = spark.read.csv(file_path, header=True, inferSchema=True)
        st.write("### Dataset Preview")
        st.dataframe(df.limit(5).toPandas())

        st.write("### Evaluating Models...")
        results = predict_delay(file_path)

        st.write("### Model Evaluation Results")
        results_df = pd.DataFrame.from_dict(results, orient="index")
        st.dataframe(results_df)

        st.write("### Model Performance Comparison")

        if "Error" not in results_df.columns:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))


            rmse_values = results_df["RMSE"]
            ax[0].bar(results_df.index, rmse_values, color="skyblue")
            ax[0].set_xlabel("Models")
            ax[0].set_ylabel("RMSE (Root Mean Squared Error)")
            ax[0].set_title("Model Comparison based on RMSE")
            ax[0].tick_params(axis="x", rotation=45)
            for i, v in enumerate(rmse_values):
                ax[0].text(i, v + 0.2, f"{v:.2f}", ha="center")

           
            r2_values = results_df["R²"]
            ax[1].bar(results_df.index, r2_values, color="lightgreen")
            ax[1].set_xlabel("Models")
            ax[1].set_ylabel("R² (Coefficient of Determination)")
            ax[1].set_title("Model Comparison based on R²")
            ax[1].tick_params(axis="x", rotation=45)
            for i, v in enumerate(r2_values):
                ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center")

            
            best_model = results_df["R²"].idxmax()
            best_r2 = results_df["R²"].max()
            st.write(f"### Best Model based on R²: {best_model} with R² = {best_r2:.2f}")

            st.pyplot(fig)
        else:
            st.warning("Some models encountered errors during training. Results may be incomplete.")


if __name__ == "__main__":
    main()
