from flask import Flask, render_template, jsonify, request, send_file
from src.exception import VisibilityException
from src.logger import logging as lg
import os,sys

from src.pipeline.training_pipeline import TraininingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train_route():
    try:
        train_pipeline = TraininingPipeline()
        train_pipeline.run_pipeline()

        return jsonify("Training Successfull.")

    except Exception as e:
        raise VisibilityException(e,sys)
    

@app.route("/predict", methods = ['POST', 'GET'])
def predict():
    try:
        if request.method == "POST":
            prediction_pipeline = PredictionPipeline(request=request)
            predicted_visibility = prediction_pipeline.run_pipeline()
            print(predicted_visibility)
            return render_template("result.html", prediction= f"{predicted_visibility[0] :.3f}")
    except Exception as e:
        raise VisibilityException(e,sys)

    


if __name__ == "__main__":
    print("Starting the Flask server")
    print("Flask application running on port 8062")
    print("Click on the link to open the application: http://localhost:8062/")
    app.run(host="0.0.0.0", port=8062, debug= True)
