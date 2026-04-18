import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer_ml import ModelTrainerML

from src.exception import VisibilityException


# ======================================
# 🔥 TRAINING PIPELINE (ML ONLY)
# ======================================
class TrainingPipeline:

    # ======================================
    # 📥 INGESTION
    # ======================================
    def start_data_ingestion(self):
        data_ingestion = DataIngestion()
        return data_ingestion.initiate_data_ingestion()

    # ======================================
    # 🔍 VALIDATION
    # ======================================
    def start_data_validation(self, raw_data_dir):
        data_validation = DataValidation(raw_data_store_dir=raw_data_dir)
        return data_validation.initiate_data_validation()

    # ======================================
    # ⚙️ TRANSFORMATION
    # ======================================
    def start_data_transformation(self, valid_data_dir):
        data_transformation = DataTransformation(valid_data_dir=valid_data_dir)
        return data_transformation.initiate_data_transformation()

    # ======================================
    # 🤖 ML TRAINING (STACKING PIPELINE)
    # ======================================
    def start_ml_training(self, X_train, y_train, X_test, y_test):
        trainer = ModelTrainerML()
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)
        return model, metrics

    # ======================================
    # 🚀 MAIN PIPELINE
    # ======================================
    def run_pipeline(self):
        try:
            print("🚀 Starting Training Pipeline")

            # -------------------------
            # 1. INGESTION
            # -------------------------
            data_paths = self.start_data_ingestion()
            raw_data_dir = os.path.dirname(data_paths["raw_data_path"])

            # -------------------------
            # 2. VALIDATION
            # -------------------------
            valid_data_dir = self.start_data_validation(raw_data_dir)

            # -------------------------
            # 3. TRANSFORMATION
            # -------------------------
            data = self.start_data_transformation(valid_data_dir)

            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]

            # -------------------------
            # 4. ML TRAINING (STACKING)
            # -------------------------
            model, metrics = self.start_ml_training(
                X_train, y_train, X_test, y_test
            )

            print("\n📊 MODEL PERFORMANCE")
            print(metrics)

            print("\n✅ Training pipeline completed successfully")

            return model, metrics

        except Exception as e:
            raise VisibilityException(e, sys)