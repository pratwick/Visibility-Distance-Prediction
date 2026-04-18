import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from flask import request

from src.cloud_storage.aws_syncer import S3Sync
from src.utils.main_utils import MainUtils
from src.exception import VisibilityException

@dataclass
class PredictionPipelineConfig:
    artifacts_dir: str = os.path.join(os.getcwd(), "artifacts")


class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.s3_sync = S3Sync()
        self.utils = MainUtils()
        self.config = PredictionPipelineConfig()

        self.model_path = None

        print("[INIT] PredictionPipeline initialized")

    def get_latest_artifact_folder(self):
        try:
            print("Searching latest artifact folder...")

            base_dir = self.config.artifacts_dir

            if not os.path.exists(base_dir):
                raise Exception(f"Artifacts directory not found: {base_dir}")

            folders = [
                os.path.join(base_dir, d)
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]

            if not folders:
                raise Exception("No artifact folders found")

            latest_folder = max(folders, key=os.path.getmtime)

            print(f"Latest artifact: {latest_folder}")
            return latest_folder

        except Exception as e:
            print(f"Artifact Error: {e}")
            raise VisibilityException(e, sys)

    def download_models(self):
        try:
            print("Trying S3 sync...")
            self.s3_sync.sync_folder_from_s3(
                folder=self.config.artifacts_dir,
                aws_bucket_name=os.getenv("AWS_S3_BUCKET_NAME")
            )
            print("S3 sync successful")

        except Exception as e:
            print(f"S3 failed → using local artifacts: {e}")

    def get_input_dataframe(self):
        try:
            print("Reading input from form...")

            data = dict(self.request.form.items())
            print(f"Raw input: {data}")

            df = pd.DataFrame([data])
            df = df.apply(pd.to_numeric, errors="coerce")

            print(f"DataFrame:\n{df}")

            return df

        except Exception as e:
            print(f"Input Error: {e}")
            raise VisibilityException(e, sys)

    def apply_feature_engineering(self, df):
        try:
            print("Applying feature engineering...")

            # Rename according to training dataset
            df.rename(columns={
                "temp_c": "Temp_C",
                "dew_point_temp_c": "Dew_Point_Temp_C",
                "rel_hum": "Rel_Hum_%",
                "wind_speed": "Wind_Speed_km/h",
                "press_kpa": "Press_kPa"
            }, inplace=True)

            print(f"Columns after rename: {list(df.columns)}")

            # Add time features
            df["hour"] = pd.Timestamp.now().hour
            df["month"] = pd.Timestamp.now().month

            # Derived features (match training logic)
            df['temp_dew_diff'] = df['Temp_C'] - df['Dew_Point_Temp_C']
            df['humidity_temp'] = df['Rel_Hum_%'] * df['Temp_C']
            df['pressure_change'] = 0  # default (no history)

            # Lag features → set to 0 (since no past data)
            for lag in [1, 2, 3, 6, 12, 24, 48]:
                df[f"vis_lag_{lag}"] = 0

            # Rolling features → set to 0
            df["vis_roll_mean_6"] = 0
            df["vis_roll_mean_12"] = 0
            df["vis_roll_std_6"] = 0

            print(f"Final features: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"Feature Engineering Error: {e}")
            raise VisibilityException(e, sys)

    def predict_ml(self, df):
        try:
            print("Loading model...")

            model = self.utils.load_object(self.model_path)

            print("Model loaded successfully")

            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = df.columns.tolist()

            print(f"Expected features: {expected_features}")

            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0

            df = df[expected_features]

            print(f"Final input shape: {df.shape}")

            prediction = model.predict(df)[0]

            print(f"Prediction: {prediction}")

            return float(prediction)

        except Exception as e:
            print(f" Prediction Error: {e}")
            raise VisibilityException(e, sys)

    def run_pipeline(self):
        try:
            print("\n====== PREDICTION PIPELINE STARTED ======\n")

            # Step 1: S3 Sync (optional)
            self.download_models()

            # Step 2: Get latest artifact
            artifact_folder = self.get_latest_artifact_folder()

            self.model_path = os.path.join(
                artifact_folder,
                "model_trainer",
                "stacking_model.pkl"
            )

            print(f"Model path: {self.model_path}")

            # Step 3: Input
            df = self.get_input_dataframe()

            # Step 4: Feature engineering
            df = self.apply_feature_engineering(df)

            # Step 5: Prediction
            prediction = self.predict_ml(df)

            print("\n ====== PIPELINE COMPLETED ======\n")

            return {
                "prediction": prediction,
                "status": "success",
                "artifact_used": os.path.basename(artifact_folder)
            }

        except Exception as e:
            print(f"Pipeline Error: {e}")
            raise VisibilityException(e, sys)