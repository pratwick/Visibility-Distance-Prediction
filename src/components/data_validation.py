import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.constant import *
from src.exception import VisibilityException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(artifact_folder, "data_validation")
    valid_data_dir: str = os.path.join(data_validation_dir, "validated")
    invalid_data_dir: str = os.path.join(data_validation_dir, "invalid")

class DataValidation:
    def __init__(self, raw_data_store_dir: str):
        self.raw_data_store_dir = raw_data_store_dir
        self.config = DataValidationConfig()
        self.utils = MainUtils()
        self.schema = self.utils.read_schema_config_file()

    def load_data(self) -> pd.DataFrame:
        try:
            files = os.listdir(self.raw_data_store_dir)

            df_list = [
                pd.read_csv(os.path.join(self.raw_data_store_dir, f))
                for f in files
            ]

            df = pd.concat(df_list, ignore_index=True)

            logging.info(f"Raw data shape: {df.shape}")

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        try:
            expected_cols = set(self.schema["columns"].keys())
            actual_cols = set(df.columns)

            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols

            if missing:
                logging.error(f"Missing columns: {missing}")
                return False

            if extra:
                logging.warning(f"Extra columns ignored: {extra}")

            return True

        except Exception as e:
            raise VisibilityException(e, sys)

    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        try:
            missing_ratio = df.isnull().mean()

            # allow up to 25% missing (real-world tolerance)
            if (missing_ratio > 0.25).any():
                logging.error(f"Too many missing values:\n{missing_ratio}")
                return False

            return True

        except Exception as e:
            raise VisibilityException(e, sys)

    def validate_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Validating time series data")

            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"])

            df = df.sort_values("DATE").drop_duplicates(subset=["DATE"])
            df = df.reset_index(drop=True)

            # optional sanity check
            time_diff = df["DATE"].diff().dt.total_seconds()

            if time_diff.max() > 86400 * 3:
                logging.warning("Large time gap detected (>3 days)")

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Validating feature ranges")

            # humidity (if exists)
            if "Rel Hum_%" in df.columns:
                df = df[df["Rel Hum_%"].between(0, 100)]

            # wind speed
            if "Wind_Speed_km/h" in df.columns:
                df = df[df["Wind_Speed_km/h"] >= 0]

            # pressure sanity
            if "Press_kPa" in df.columns:
                df = df[df["Press_kPa"].between(80, 120)]

            # visibility target
            if "Visibility_km" in df.columns:
                df = df[df["Visibility_km"] >= 0]

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def save_valid_data(self, df: pd.DataFrame):
        try:
            os.makedirs(self.config.valid_data_dir, exist_ok=True)

            path = os.path.join(self.config.valid_data_dir, "validated.csv")
            df.to_csv(path, index=False)

            logging.info(f"Validated data saved at {path}")

            return self.config.valid_data_dir

        except Exception as e:
            raise VisibilityException(e, sys)

    def initiate_data_validation(self):
        logging.info("Starting data validation pipeline")

        try:
            df = self.load_data()

            # schema check
            if not self.validate_schema(df):
                raise Exception("Schema validation failed")

            # missing values
            if not self.validate_missing_values(df):
                raise Exception("Missing value validation failed")

            # time series cleanup
            df = self.validate_time_series(df)

            # range validation
            df = self.validate_ranges(df)

            # save
            valid_dir = self.save_valid_data(df)

            logging.info("Data validation completed successfully")

            return valid_dir

        except Exception as e:
            raise VisibilityException(e, sys)