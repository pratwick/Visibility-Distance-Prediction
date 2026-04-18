import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer_ml import ModelTrainerML

from src.exception import VisibilityException

class TrainingPipeline:

    def start_data_ingestion(self):
        data_ingestion = DataIngestion()
        return data_ingestion.initiate_data_ingestion()

    def start_data_validation(self, raw_data_dir):
        data_validation = DataValidation(raw_data_store_dir=raw_data_dir)
        return data_validation.initiate_data_validation()

    def start_data_transformation(self, valid_data_dir):
        data_transformation = DataTransformation(valid_data_dir=valid_data_dir)
        return data_transformation.initiate_data_transformation()

    def start_ml_training(self, X_train, y_train, X_test, y_test):
        trainer = ModelTrainerML()
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)
        return model, metrics

    def run_pipeline(self):
        try:
            print("🚀 Starting Training Pipeline")

            data_paths = self.start_data_ingestion()
            raw_data_dir = os.path.dirname(data_paths["raw_data_path"])

            valid_data_dir = self.start_data_validation(raw_data_dir)

            data = self.start_data_transformation(valid_data_dir)

            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]

            model, metrics = self.start_ml_training(
                X_train, y_train, X_test, y_test
            )

            print("\nMODEL PERFORMANCE")
            print(metrics)

            print("\n Training pipeline completed successfully")

            return model, metrics

        except Exception as e:
            raise VisibilityException(e, sys)