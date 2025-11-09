
"""
Feature Extractor for ML models
Extracts and processes features from HLD documents.

Also provides a lightweight HLDFeatures dataclass and an `extract` wrapper
returning a structured object to satisfy test expectations.
"""


import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class HLDFeatures:
	# Minimal subset; dynamic attributes will be attached for missing features
	word_count: int = 0
	sentence_count: int = 0
	avg_sentence_length: float = 0.0


class FeatureExtractor:
	def __init__(self):
		# Example feature list (should be 37 in production)
		# Align with SyntheticDatasetGenerator.ALL_FEATURES for consistent ordering (37 features)
		self.feature_names = [
			'word_count', 'sentence_count', 'avg_sentence_length',
			'header_count', 'code_block_count', 'table_count', 'list_count',
			'security_mentions', 'scalability_mentions', 'api_mentions',
			'service_count', 'entity_count', 'api_endpoint_count',
			'readability_score', 'documentation_quality',
		] + [f'feature_{i}' for i in range(16, 38)]  # filler features to reach 37
		self.feature_ranges = {  # Example ranges
			'word_count': (500, 5000),
			'sentence_count': (20, 300),
			'avg_sentence_length': (10, 40),
			'header_count': (3, 20),
			'code_block_count': (0, 10),
			'table_count': (0, 10),
			'list_count': (0, 20),
			'security_mentions': (0, 10),
			'scalability_mentions': (0, 10),
			'api_mentions': (0, 10),
			'service_count': (1, 20),
			'entity_count': (1, 30),
			'api_endpoint_count': (1, 50),
			'readability_score': (0, 100),
			'documentation_quality': (0, 100),
			# ...
		}
		# Include filler feature ranges to support scenario sliders and robust inference
		for i in range(16, 38):
			self.feature_ranges[f'feature_{i}'] = (0, 100)

	def extract_features(self, hld_text: str) -> Dict[str, float]:
		logger.info("Extracting features from HLD text.")
		features = {}
		features['word_count'] = self.count_words(hld_text)
		features['sentence_count'] = self.count_sentences(hld_text)
		features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
		features['header_count'] = self.count_headers(hld_text)
		features['code_block_count'] = self.count_code_blocks(hld_text)
		features['table_count'] = self.count_tables(hld_text)
		features['list_count'] = self.count_lists(hld_text)
		features['security_mentions'] = self.count_mention(hld_text, ['security', 'encryption', 'auth'])
		features['scalability_mentions'] = self.count_mention(hld_text, ['scalable', 'scalability', 'horizontal'])
		features['api_mentions'] = self.count_mention(hld_text, ['api', 'endpoint', 'rest'])
		# ... more features ...
		# Fill with dummy values for missing features
		for name in self.feature_names:
			if name not in features:
				features[name] = 0.0
		# Ensure ordering consistency when later converted to arrays
		features = {k: features[k] for k in self.feature_names}
		logger.debug(f"Extracted features: {features}")
		return features

	def validate_features(self, features: Dict) -> bool:
		logger.info("Validating features.")
		for name in self.feature_names:
			if name not in features:
				logger.warning(f"Feature missing: {name}")
				return False
			rng = self.feature_ranges.get(name)
			if rng and not (rng[0] <= features[name] <= rng[1]):
				logger.warning(f"Feature {name} out of range: {features[name]}")
				return False
		logger.info("All features valid.")
		return True

	def extract(self, hld_text: str) -> HLDFeatures:
		"""Extract features and return a structured HLDFeatures instance.

		Tests expect an object with attributes like word_count, sentence_count, etc.
		We create the dataclass and attach any remaining features as attributes.
		"""
		m = self.extract_features(hld_text)
		obj = HLDFeatures(
			word_count=int(m.get('word_count', 0)),
			sentence_count=int(m.get('sentence_count', 0)),
			avg_sentence_length=float(m.get('avg_sentence_length', 0.0)),
		)
		# Attach all other features dynamically
		for k, v in m.items():
			if not hasattr(obj, k):
				setattr(obj, k, v)
		return obj

	def get_feature_names(self) -> List[str]:
		logger.info("Getting feature names.")
		return self.feature_names

	def get_feature_ranges(self) -> Dict[str, Tuple]:
		logger.info("Getting feature ranges.")
		return self.feature_ranges

	@staticmethod
	def count_words(text: str) -> int:
		return len(text.split())

	@staticmethod
	def count_sentences(text: str) -> int:
		return len(re.findall(r'[.!?]+', text))

	@staticmethod
	def count_headers(markdown: str) -> int:
		return len(re.findall(r'^#+ ', markdown, re.MULTILINE))

	@staticmethod
	def count_code_blocks(markdown: str) -> int:
		return len(re.findall(r'```', markdown)) // 2

	@staticmethod
	def count_tables(markdown: str) -> int:
		return len(re.findall(r'\|', markdown))

	@staticmethod
	def count_lists(markdown: str) -> int:
		return len(re.findall(r'^[-*+] ', markdown, re.MULTILINE))

	@staticmethod
	def count_mention(text: str, keywords: List[str]) -> int:
		return sum(text.lower().count(k) for k in keywords)

	@staticmethod
	def get_sections(markdown: str) -> Dict[str, bool]:
		sections = ['architecture', 'security', 'api', 'scalability']
		return {s: (s in markdown.lower()) for s in sections}

	# Add normalization, scaling, missing/null handling as needed
