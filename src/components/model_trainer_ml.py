import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.constant import *
from src.exception import VisibilityException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class ModelTrainerConfig:
    model_dir = os.path.join(artifact_folder, "model_trainer")
    model_path = os.path.join(model_dir, "stacking_model.pkl")

class ModelTrainerML:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()

    def evaluate(self, y_true, y_pred):
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred)
        }

    def train(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training pipeline")

            tscv = TimeSeriesSplit(n_splits=5)

            xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

            xgb_params = {
                "n_estimators": [400, 600],
                "max_depth": [6, 8],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "gamma": [0, 0.1]
            }

            xgb_grid = GridSearchCV(
                xgb,
                xgb_params,
                cv=tscv,
                scoring="r2",
                n_jobs=-1,
                verbose=1
            )

            xgb_grid.fit(X_train, y_train)
            best_xgb = xgb_grid.best_estimator_

            xgb_pred = best_xgb.predict(X_test)
            xgb_metrics = self.evaluate(y_test, xgb_pred)

            logging.info(f"XGBoost metrics: {xgb_metrics}")

            lgb = LGBMRegressor(random_state=42, verbose=-1)

            lgb_params = {
                "n_estimators": [400, 600],
                "learning_rate": [0.03, 0.05],
                "num_leaves": [31, 64],
                "max_depth": [-1, 10],
                "subsample": [0.8],
                "colsample_bytree": [0.8]
            }

            lgb_grid = GridSearchCV(
                lgb,
                lgb_params,
                cv=tscv,
                scoring="r2",
                n_jobs=-1,
                verbose=1
            )

            lgb_grid.fit(X_train, y_train)
            best_lgb = lgb_grid.best_estimator_

            lgb_pred = best_lgb.predict(X_test)
            lgb_metrics = self.evaluate(y_test, lgb_pred)

            logging.info(f"LightGBM metrics: {lgb_metrics}")

            rf = RandomForestRegressor(
                n_estimators=400,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_metrics = self.evaluate(y_test, rf_pred)

            logging.info(f"RandomForest metrics: {rf_metrics}")

            stack = StackingRegressor(
                estimators=[
                    ("xgb", best_xgb),
                    ("lgb", best_lgb),
                    ("rf", rf)
                ],
                final_estimator=Ridge()
            )

            stack.fit(X_train, y_train)
            stack_pred = stack.predict(X_test)
            stack_metrics = self.evaluate(y_test, stack_pred)

            logging.info(f"Stacking metrics: {stack_metrics}")

            os.makedirs(self.config.model_dir, exist_ok=True)
            self.utils.save_object(self.config.model_path, stack)

            logging.info(f"Model saved at {self.config.model_path}")

            results = pd.DataFrame({
                "Model": ["XGBoost", "LightGBM", "RandomForest", "Stacking"],
                "R2": [
                    xgb_metrics["r2"],
                    lgb_metrics["r2"],
                    rf_metrics["r2"],
                    stack_metrics["r2"]
                ],
                "RMSE": [
                    xgb_metrics["rmse"],
                    lgb_metrics["rmse"],
                    rf_metrics["rmse"],
                    stack_metrics["rmse"]
                ]
            })

            logging.info(f"\n{results}")

            return stack, results

        except Exception as e:
            raise VisibilityException(e, sys)