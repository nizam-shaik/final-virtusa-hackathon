
"""
Rule-Based Quality Scorer for HLD documents
Provides rule-based quality assessment without ML
"""


from dataclasses import dataclass, field
from typing import List
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
	overall_score: float
	completeness: float
	clarity: float
	consistency: float
	security: float
	recommendations: List[str] = field(default_factory=list)
	missing_elements: List[str] = field(default_factory=list)

class RuleBasedQualityScorer:
	def __init__(self):
		self.weights = {
			'completeness': 0.3,
			'clarity': 0.25,
			'consistency': 0.2,
			'security': 0.25
		}
		self.required_sections = ['architecture', 'security', 'api', 'scalability']

	def score(self, hld_markdown: str) -> QualityScore:
		logger.info("Scoring HLD markdown.")
		completeness = self.check_section_completeness(hld_markdown)
		clarity = self.calculate_readability(hld_markdown)
		consistency = self.check_formatting_consistency(hld_markdown)
		security = self.check_security_coverage(hld_markdown)
		overall = (
			completeness * self.weights['completeness'] +
			clarity * self.weights['clarity'] +
			consistency * self.weights['consistency'] +
			security * self.weights['security']
		)
		missing = self.identify_missing_elements(hld_markdown)
		recs = self.generate_recommendations(overall, completeness, clarity, consistency, security, missing)
		logger.info(f"Score breakdown: completeness={completeness}, clarity={clarity}, consistency={consistency}, security={security}, overall={overall}")
		logger.debug(f"Recommendations: {recs}, Missing: {missing}")
		return QualityScore(
			overall_score=overall,
			completeness=completeness,
			clarity=clarity,
			consistency=consistency,
			security=security,
			recommendations=recs,
			missing_elements=missing
		)

	def check_section_completeness(self, markdown: str) -> float:
		found = sum(1 for s in self.required_sections if s in markdown.lower())
		return 100.0 * found / len(self.required_sections)

	def calculate_readability(self, text: str) -> float:
		words = len(text.split())
		sentences = len(re.findall(r'[.!?]+', text))
		avg_len = words / max(sentences, 1)
		# Simple heuristic: shorter sentences = more readable
		score = max(0, min(100, 100 - (avg_len - 15) * 3))
		return score

	def check_formatting_consistency(self, markdown: str) -> float:
		headers = re.findall(r'^#+ ', markdown, re.MULTILINE)
		bullets = re.findall(r'^[-*+] ', markdown, re.MULTILINE)
		if not headers or not bullets:
			return 50.0
		return 100.0

	def check_security_coverage(self, text: str) -> float:
		keywords = ['security', 'encryption', 'auth']
		count = sum(text.lower().count(k) for k in keywords)
		return min(100.0, count * 20.0)

	def generate_recommendations(self, overall, completeness, clarity, consistency, security, missing) -> List[str]:
		recs = []
		if completeness < 100:
			recs.append("Add missing sections: " + ", ".join(missing))
		if clarity < 70:
			recs.append("Improve sentence clarity and reduce sentence length.")
		if consistency < 80:
			recs.append("Standardize formatting (headers, bullets, etc.)")
		if security < 60:
			recs.append("Add more security-related content.")
		if overall > 90:
			recs.append("Excellent HLD quality!")
		return recs

	def identify_missing_elements(self, markdown: str) -> List[str]:
		return [s for s in self.required_sections if s not in markdown.lower()]
