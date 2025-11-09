
"""
ML Model Training Pipeline
Trains multiple model types on synthetic HLD dataset
"""


import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
	from xgboost import XGBRegressor  # type: ignore
except ImportError:
	XGBRegressor = None  # Optional dependency; proceed without it

logger = logging.getLogger(__name__)

class LargeScaleMLTrainer:
	def __init__(self):
		self.models = {
			'RandomForest': RandomForestRegressor(),
			'GradientBoosting': GradientBoostingRegressor(),
			'SVR': SVR(),
			'LinearRegression': LinearRegression(),
		}
		if XGBRegressor:
			self.models['XGBoost'] = XGBRegressor()
		self.results = {}
		self.df = None
		self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

	def load_dataset(self, filepath: str):
		logger.info(f"Loading dataset from {filepath}")
		self.df = pd.read_csv(filepath)
		logger.info(f"Loaded dataset with shape {self.df.shape}")
		assert self.df.shape[1] >= 2, "Dataset must have features and target."

	def prepare_data(self):
		logger.info("Preparing data for training (splitting train/test)")
		features = [col for col in self.df.columns if col != 'quality_score']
		X = self.df[features]
		y = self.df['quality_score']
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		logger.info(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

	def train_models(self):
		logger.info("Training models...")
		for name, model in self.models.items():
			logger.info(f"Training {name} model.")
			model.fit(self.X_train, self.y_train)
			self.models[name] = model
			logger.info(f"{name} model trained.")

	def evaluate_models(self):
		logger.info("Evaluating models...")
		for name, model in self.models.items():
			y_pred = model.predict(self.X_test)
			r2 = r2_score(self.y_test, y_pred)
			# Compatibility with older scikit-learn versions: compute RMSE manually
			mse = mean_squared_error(self.y_test, y_pred)
			rmse = float(np.sqrt(mse))
			mae = mean_absolute_error(self.y_test, y_pred)
			mape = np.mean(np.abs((self.y_test - y_pred) / np.maximum(np.abs(self.y_test), 1e-8))) * 100
			self.results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}
			logger.info(f"Metrics for {name}: {self.results[name]}")

	def save_models(self, out_dir: str):
		logger.info(f"Saving models to {out_dir}")
		os.makedirs(out_dir, exist_ok=True)
		for name, model in self.models.items():
			with open(os.path.join(out_dir, f"{name}_model.pkl"), 'wb') as f:
				pickle.dump(model, f)
			logger.info(f"Saved {name} model.")

	def get_feature_importance(self, model_name: str):
		model = self.models.get(model_name)
		if hasattr(model, 'feature_importances_'):
			return model.feature_importances_
		return None

	def cross_validation_score(self, model_name: str, cv=5):
		from sklearn.model_selection import cross_val_score
		model = self.models.get(model_name)
		scores = cross_val_score(model, self.X_train, self.y_train, cv=cv)
		return float(np.mean(scores))
