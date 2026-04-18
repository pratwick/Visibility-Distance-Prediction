import sys
import os
import yaml
import pickle
from typing import Any, Dict

from src.exception import VisibilityException
from src.logger import logging

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"YAML file not found: {filename}")

            with open(filename, "r", encoding="utf-8") as yaml_file:
                content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError(f"YAML file is empty: {filename}")

            return content

        except Exception as e:
            raise VisibilityException(e, sys)

    def read_schema_config_file(self) -> dict:
        try:
            schema_path = os.path.join("config", "schema.yaml")

            return self.read_yaml_file(schema_path)

        except Exception as e:
            raise VisibilityException(e, sys)

    @staticmethod
    def save_object(file_path: str, obj: Any) -> None:
        logging.info("Saving object...")

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info(f"Object saved at: {file_path}")

        except Exception as e:
            raise VisibilityException(e, sys)

    @staticmethod
    def load_object(file_path: str) -> Any:
        logging.info("Loading object...")

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Object file not found: {file_path}")

            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info(f"Object loaded from: {file_path}")

            return obj

        except Exception as e:
            raise VisibilityException(e, sys)