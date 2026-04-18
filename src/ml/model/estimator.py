import sys
import os
import numpy as np
import pandas as pd
from typing import List, Generator, Optional, Dict, Any
from pymongo import MongoClient

from src.exception import VisibilityException
from src.logger import logging


class VisibilityData:
    """
    Handles MongoDB extraction for Visibility ML pipeline
    """

    def __init__(self, database_name: str):
        try:
            self.database_name = database_name

            self.mongo_url = os.getenv("MONGO_DB_URL")
            if not self.mongo_url:
                raise Exception("MONGO_DB_URL not found in environment variables")

            self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)

            # Force connection check
            self.client.server_info()

            self.db = self.client[self.database_name]

            logging.info("MongoDB connection established successfully")

        except Exception as e:
            raise VisibilityException(e, sys)

    def get_collection_names(self) -> List[str]:
        try:
            collections = self.db.list_collection_names()
            logging.info(f"Collections found: {collections}")
            return collections

        except Exception as e:
            raise VisibilityException(e, sys)

    def get_collection_data(
        self,
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        try:
            logging.info(f"Fetching collection: {collection_name}")

            collection = self.db[collection_name]

            cursor = collection.find(query or {}, projection)

            df = pd.DataFrame(list(cursor))

            if df.empty:
                logging.warning(f"No data found in {collection_name}")
                return df

            # Remove Mongo ID
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Normalize missing values
            df.replace({"na": np.nan, "NA": np.nan, "null": np.nan}, inplace=True)

            # Convert DATE safely
            if "DATE" in df.columns:
                df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
                df = df.dropna(subset=["DATE"])

            logging.info(f"{collection_name} shape: {df.shape}")

            return df

        except Exception as e:
            raise VisibilityException(e, sys)

    def export_collections_as_dataframe(self) -> Generator:
        """
        Yields (collection_name, dataframe)
        """
        try:
            collections = self.get_collection_names()

            for collection_name in collections:
                df = self.get_collection_data(collection_name)

                if df is None or df.empty:
                    continue

                yield collection_name, df

        except Exception as e:
            raise VisibilityException(e, sys)