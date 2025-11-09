
"""
ML Quality Model - Quality prediction models

Backwards compatibility:
This module defines `QualityPredictionModel` as the core class and exposes
`MLQualityModel` as an alias to satisfy legacy imports used elsewhere.
"""


import pickle
import logging
from typing import Any, Dict
import numpy as np

logger = logging.getLogger(__name__)

class QualityPredictionModel:
	def __init__(self, model, model_type: str = "RandomForest"):
		self.model = model
		self.model_type = model_type
		self.trained = False

	def train(self, X_train, y_train):
		logger.info(f"Training {self.model_type} model.")
		self.model.fit(X_train, y_train)
		self.trained = True
		logger.info(f"{self.model_type} model trained.")

	def predict(self, X_test):
		logger.info(f"Predicting with {self.model_type} model.")
		preds = self.model.predict(X_test)
		logger.debug(f"Predictions: {preds}")
		return preds

	def evaluate(self, X_test, y_test) -> Dict[str, float]:
		logger.info(f"Evaluating {self.model_type} model.")
		y_pred = self.predict(X_test)
		from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
		r2 = r2_score(y_test, y_pred)
		# Compute RMSE manually for compatibility with older sklearn
		mse = mean_squared_error(y_test, y_pred)
		rmse = float(np.sqrt(mse))
		mae = mean_absolute_error(y_test, y_pred)
		mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
		metrics = {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}
		logger.info(f"Evaluation metrics: {metrics}")
		return metrics

	def get_feature_importance(self):
		if hasattr(self.model, 'feature_importances_'):
			return self.model.feature_importances_
		return None

	def save(self, filepath: str):
		logger.info(f"Saving model to {filepath}")
		with open(filepath, 'wb') as f:
			pickle.dump(self, f)
		logger.info("Model saved.")

	@staticmethod
	def load(filepath: str):
		logger.info(f"Loading model from {filepath}")
		with open(filepath, 'rb') as f:
			model = pickle.load(f)
		logger.info("Model loaded.")
		return model

	def get_model_type(self) -> str:
		return self.model_type

	def cross_validate(self, X, y, cv=5):
		from sklearn.model_selection import cross_val_score
		scores = cross_val_score(self.model, X, y, cv=cv)
		return float(np.mean(scores))

# Legacy alias expected by imports in models.__init__
MLQualityModel = QualityPredictionModel
