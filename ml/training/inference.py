
"""
HLD Quality Predictor - Inference interface for trained ML models
Predicts HLD quality using ensemble of models
"""


import pickle
import os
import numpy as np
import logging
from typing import Dict, List, Optional

# Lazy imports inside methods to avoid heavy startup cost


logger = logging.getLogger(__name__)

class HLDQualityPredictor:
	"""Inference interface for trained ML models.

	Enhanced to support:
	 - Default initialization (tests expect no-arg constructor).
	 - Automatic small training routine if models are absent.
	"""

	def __init__(self, model_dir: Optional[str] = None, feature_names: Optional[List[str]] = None):
		if model_dir is None:
			# Default to local models directory beside this file
			model_dir = os.path.join(os.path.dirname(__file__), 'models')
		self.model_dir = model_dir
		if feature_names is None:
			try:
				from ml.models.feature_extractor import FeatureExtractor
				feature_names = FeatureExtractor().get_feature_names()
			except Exception:
				feature_names = []
		self.feature_names = feature_names
		self.models: Dict[str, object] = {}

	def load_models_from_disk(self) -> bool:
		"""Load previously trained models.

		Gracefully handles missing directory by creating it and returning False
		rather than raising an exception.
		"""
		logger.info(f"Loading models from {self.model_dir}")
		if not os.path.isdir(self.model_dir):
			logger.warning(f"Model directory '{self.model_dir}' does not exist. Creating it.")
			try:
				os.makedirs(self.model_dir, exist_ok=True)
			except Exception as e:
				logger.error(f"Could not create model directory: {e}")
				return False
		try:
			for fname in os.listdir(self.model_dir):
				if fname.endswith('_model.pkl'):
					model_name = fname.split('_')[0]
					with open(os.path.join(self.model_dir, fname), 'rb') as f:
						self.models[model_name] = pickle.load(f)
					logger.info(f"Loaded model: {model_name}")
			logger.info(f"Total models loaded: {len(self.models)}")
			return bool(self.models)
		except Exception as e:
			logger.error(f"Error loading models: {e}")
			return False

	def predict(self, features: Dict[str, float]) -> Dict[str, float]:
		logger.info(f"Predicting quality with features: {features}")
		# Ensure we have a complete feature set; if predictor knows fewer than the model, we'll realign per-model below
		missing = [name for name in self.feature_names if name not in features]
		if missing:
			logger.warning(f"Adding missing features with default 0.0: {missing}")
			for m in missing:
				features[m] = 0.0

		# Prepare a base row with the predictor's feature order (may be < model's expectation; we'll adjust per model)
		row_base = {name: features.get(name, 0.0) for name in self.feature_names}

		preds: Dict[str, float] = {}
		for name, model in self.models.items():
			try:
				# If model exposes `feature_names_in_`, construct input using exactly those names
				model_feature_names = getattr(model, 'feature_names_in_', None)
				X_input = None
				try:
					import pandas as pd
					expected_cols = list(model_feature_names) if model_feature_names is not None else self.feature_names
					row = {fn: features.get(fn, row_base.get(fn, 0.0)) for fn in expected_cols}
					X_input = pd.DataFrame([row], columns=expected_cols)
				except Exception:
					# Fallback to numpy array if pandas isn't available
					expected_cols = list(model_feature_names) if model_feature_names is not None else self.feature_names
					X_input = np.array([row_base.get(fn, 0.0) for fn in expected_cols]).reshape(1, -1)

				preds[name] = float(model.predict(X_input)[0])
				logger.info(f"Prediction from {name}: {preds[name]}")
			except Exception as e:
				logger.error(f"Model {name} prediction failed: {e}")

		preds['ensemble_average'] = float(np.mean([v for k, v in preds.items() if k != 'ensemble_average'])) if preds else 0.0
		logger.info(f"Ensemble average: {preds['ensemble_average']}")
		return preds

	def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
		return [self.predict(f) for f in features_list]

	def train_models_from_scratch(self, n_samples: int = 1000) -> bool:
		"""Generate a small synthetic dataset and train baseline models.

		Used as a fallback when no pre-trained models exist (primarily for tests / demos).
		"""
		try:
			from ml.training.generate_dataset import SyntheticDatasetGenerator
			from ml.training.train_large_model import LargeScaleMLTrainer
		except ImportError as e:
			logger.error(f"Training modules unavailable: {e}")
			return False
		gen = SyntheticDatasetGenerator(random_state=42)
		df = gen.generate(n_samples=n_samples)
		trainer = LargeScaleMLTrainer()
		# Use in-memory DataFrame without saving CSV
		trainer.df = df
		trainer.prepare_data()
		trainer.train_models()
		trainer.evaluate_models()
		os.makedirs(self.model_dir, exist_ok=True)
		trainer.save_models(self.model_dir)
		return self.load_models_from_disk()

	def print_feature_guide(self):
		print("Feature guide:")
		for name in self.feature_names:
			print(f"{name}")

	def get_feature_ranges(self) -> Dict[str, tuple]:
		# Placeholder: should be loaded from FeatureExtractor
		return {name: (0, 1) for name in self.feature_names}
