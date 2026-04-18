import sys
import time
import pandas as pd
from pandas import DataFrame

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import VisibilityException
from src.logger import logging

class VisibilityEstimator:
    """
    Production-grade S3 model handler for visibility prediction
    """

    def __init__(self, bucket_name: str, model_name: str, model_version: str = "v1"):
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.model_version = model_version

        self.s3 = SimpleStorageService()
        self.loaded_model = None

    def _get_s3_key(self):
        return f"{self.model_name}/{self.model_version}/model.pkl"

    def is_model_present(self) -> bool:
        try:
            key = self._get_s3_key()
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=key
            )
        except Exception as e:
            logging.error(f"Model check failed: {e}")
            return False

    def load_model(self):
        try:
            key = self._get_s3_key()

            for attempt in range(3):
                try:
                    logging.info(f"Loading model from S3: {key}")

                    model = self.s3.load_model(
                        key,
                        bucket_name=self.bucket_name
                    )

                    logging.info("Model loaded successfully")
                    return model

                except Exception as e:
                    logging.warning(f"Retry {attempt + 1}/3 failed: {e}")
                    time.sleep(2)

            raise Exception("Failed to load model after 3 retries")

        except Exception as e:
            raise VisibilityException(e, sys)

    def save_model(self, from_file: str, remove: bool = False):
        try:
            key = self._get_s3_key()

            logging.info(f"Uploading model to S3: {key}")

            self.s3.upload_file(
                from_file,
                to_filename=key,
                bucket_name=self.bucket_name,
                remove=remove
            )

            logging.info("Model uploaded successfully")

        except Exception as e:
            raise VisibilityException(e, sys)

    def _validate_input(self, df: DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

    def _align_features(self, df: pd.DataFrame, model):
        """
        Ensures inference input matches training features
        Prevents silent feature mismatch bugs
        """
        try:
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_

                missing_cols = set(expected_features) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing features: {missing_cols}")

                df = df[expected_features]

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            logging.info("Starting prediction pipeline")

            self._validate_input(dataframe)

            # Load model once
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            model = self.loaded_model

            # Align features (CRITICAL FIX)
            dataframe = self._align_features(dataframe, model)

            # Predict
            predictions = model.predict(dataframe)

            logging.info("Prediction completed successfully")

            return predictions

        except Exception as e:
            raise VisibilityException(e, sys)