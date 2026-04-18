from flask import Flask, render_template, jsonify, request
import sys

from dotenv import load_dotenv
load_dotenv()

from src.exception import VisibilityException
from src.logger import logging as lg

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train")
def train_route():
    try:
        lg.info("Training API triggered")

        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        return jsonify({
            "status": "success",
            "message": "Model training completed successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "failed",
            "message": str(e)
        }), 500

@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        if request.method == "POST":

            prediction_pipeline = PredictionPipeline(request=request)
            result = prediction_pipeline.run_pipeline()

            prediction_value = float(result["prediction"])  # ✅ KEEP FLOAT

            return render_template(
                "result.html",
                prediction=prediction_value,   # ✅ send as float
                status=result["status"],
                artifact_used=result.get("artifact_used")
            )

        return render_template("index.html")

    except Exception as e:
        print("Prediction Error:", e)

        return render_template(
            "result.html",
            prediction=None,
            status="failed",
            artifact_used=None
        )

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    print("🌐 Running on: http://localhost:8062/")

    app.run(host="0.0.0.0", port=8062, debug=True)