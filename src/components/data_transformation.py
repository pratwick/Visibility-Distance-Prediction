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
class DataTransformationConfig:
    data_transformation_dir = os.path.join(artifact_folder, "data_transformation")

class DataTransformation:
    def __init__(self, valid_data_dir):
        self.valid_data_dir = valid_data_dir
        self.config = DataTransformationConfig()
        self.utils = MainUtils()

    def get_data(self):
        try:
            files = os.listdir(self.valid_data_dir)
            df_list = [
                pd.read_csv(os.path.join(self.valid_data_dir, f))
                for f in files
            ]

            df = pd.concat(df_list, ignore_index=True)

            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"])
            df = df.sort_values("DATE").reset_index(drop=True)

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Feature engineering started")

            df["hour"] = df["DATE"].dt.hour
            df["day"] = df["DATE"].dt.day
            df["month"] = df["DATE"].dt.month
            df["season"] = (df["month"] % 12 + 3) // 3

            df['temp_dew_diff'] = df['Temp_C'] - df['Dew_Point_Temp_C']
            df['humidity_temp'] = df['Rel_Hum_%'] * df['Temp_C']
            df['pressure_change'] = df['Press_kPa'].diff()

            if "Weather" in df.columns:
                df["is_fog"] = df["Weather"].astype(str).str.contains("Fog", case=False).astype(int)
                weather_dummies = df["Weather"].str.get_dummies(sep=",")
                df = pd.concat([df, weather_dummies], axis=1)

            lags = [1, 2, 3, 6, 12, 24, 48]

            for lag in lags:
                df[f"vis_lag_{lag}"] = df["Visibility_km"].shift(lag)

            df["vis_roll_mean_6"] = df["Visibility_km"].rolling(6).mean()
            df["vis_roll_mean_12"] = df["Visibility_km"].rolling(12).mean()
            df["vis_roll_std_6"] = df["Visibility_km"].rolling(6).std()

            df = df.dropna().reset_index(drop=True)

            logging.info("Feature engineering completed")

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def split_time_series(self, df: pd.DataFrame):
        try:
            split_index = int(len(df) * 0.8)

            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()

            return train_df, test_df

        except Exception as e:
            raise VisibilityException(e, sys)

    def initiate_data_transformation(self):
        logging.info("Data transformation started")
        print("Data transformation started")

        try:
            df = self.get_data()

            # schema-driven drop
            schema = self.utils.read_schema_config_file()
            df = df.drop(columns=schema["drop_columns"], errors="ignore")

            # feature engineering
            df = self.feature_engineering(df)

            # split
            train_df, test_df = self.split_time_series(df)

            target = "Visibility_km"

            X_train = train_df.drop(columns=[target, "DATE"])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target, "DATE"])
            y_test = test_df[target]
            print("Data Tranformation completed successfully")
            logging.info("Data transformation completed")

            return {
                "X_train": X_train,
                "y_train": y_train.values,
                "X_test": X_test,
                "y_test": y_test.values,
                "train_df": train_df,
                "test_df": test_df
            }

        except Exception as e:
            raise VisibilityException(e, sys)