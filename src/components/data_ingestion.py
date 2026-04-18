import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dataclasses import dataclass

from src.constant import *
from src.exception import VisibilityException
from src.logger import logging
from src.data_access.visibility_data import VisibilityData

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder, "data_ingestion")
    raw_data_path: str = os.path.join(data_ingestion_dir, "raw_data.csv")
    train_data_path: str = os.path.join(data_ingestion_dir, "train.csv")
    test_data_path: str = os.path.join(data_ingestion_dir, "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def upload_csv_to_mongodb(self, db_name: str, collection_name: str, folder_path: str):
        try:
            logging.info("Uploading CSV data to MongoDB")

            client = MongoClient(os.getenv("MONGO_DB_URL"))
            collection = client[db_name][collection_name]

            if not os.path.exists(folder_path):
                logging.warning(f"Folder not found: {folder_path}")
                return

            files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

            if not files:
                logging.warning("No CSV files found")
                return

            all_data = []

            for file in files:
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                logging.info(f"Loaded {file} shape: {df.shape}")
                all_data.append(df)

            final_df = pd.concat(all_data, ignore_index=True).drop_duplicates()

            records = final_df.to_dict(orient="records")

            if len(records) > 0:
                # OPTIONAL: clear old data for clean pipeline
                collection.delete_many({})

                collection.insert_many(records)
                logging.info(f"Inserted {len(records)} records into MongoDB")

        except Exception as e:
            raise VisibilityException(e, sys)

    def export_collection_as_dataframe(self, collection_name, db_name):
        try:
            client = MongoClient(os.getenv("MONGO_DB_URL"))
            collection = client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def preprocess_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Preprocessing time series data")

            # DATE handling
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"])

            # SORT BEFORE ANY FEATURE ENGINEERING (IMPORTANT)
            df = df.sort_values("DATE").reset_index(drop=True)

            # REMOVE DUPLICATES
            df = df.drop_duplicates()

            logging.info(f"Final cleaned shape: {df.shape}")

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def split_data(self, df: pd.DataFrame):
        try:
            logging.info("Performing time-based split")

            split_index = int(len(df) * 0.8)

            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()

            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            raise VisibilityException(e, sys)


    def export_data(self):
        try:
            logging.info("Starting data ingestion pipeline")

            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

            visibility_data = VisibilityData(database_name=MONGO_DATABASE_NAME)

            final_df = pd.DataFrame()

            for collection_name, dataset in visibility_data.export_collections_as_dataframe():
                logging.info(f"{collection_name} shape: {dataset.shape}")
                final_df = pd.concat([final_df, dataset], ignore_index=True)

            # CLEAN
            final_df = self.preprocess_time_series(final_df)

            # SAVE RAW
            final_df.to_csv(self.config.raw_data_path, index=False)

            # SPLIT
            train_df, test_df = self.split_data(final_df)

            # SAVE
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return {
                "raw_data_path": self.config.raw_data_path,
                "train_data_path": self.config.train_data_path,
                "test_data_path": self.config.test_data_path
            }

        except Exception as e:
            raise VisibilityException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion")

        try:
            self.upload_csv_to_mongodb(
                db_name=MONGO_DATABASE_NAME,
                collection_name="VisibilityData",
                folder_path=os.path.join(os.getcwd(), "uploaded_data")
            )

            logging.info("Upload completed")

            data_paths = self.export_data()

            logging.info("Exited Data Ingestion")

            return data_paths

        except Exception as e:
            raise VisibilityException(e, sys)