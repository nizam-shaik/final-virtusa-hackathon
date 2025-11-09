from __future__ import annotations
from pathlib import Path
"""
Base Agent class for HLD generation agents
"""

# TODO: Import necessary modules for API communication, JSON parsing, and data modeling
# TODO: Implement BaseAgent class with abstract methods for system_prompt and process
# TODO: Implement call_llm() method to make API calls to Gemini with retry logic
# TODO: Implement parse_json_loose() method with multiple fallback strategies for JSON extraction
# TODO: Implement error handling and logging for API failures
# TODO: Add temperature and token parameters for LLM configuration
# TODO: Implement state normalization utilities (normalize_string, normalize_list, etc.)
# TODO: Handle both synchronous and potential asynchronous API calls
# TODO: Add API key management with environment variable loading
# TODO: Implement system prompt caching or dynamic loading strategy
# TODO: Add token counting and cost estimation features
# TODO: Implement rate limiting and backoff strategies for API calls
"""
Base Agent class for HLD generation agents
"""

import os
import json
import time
import logging
import random
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List

import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==========================================================
# Multi-Key API Key Manager
# ==========================================================
class APIKeyManager:
    """
    Manages multiple Gemini API keys with rotation and fallback.
    Thread-safe key rotation for concurrent agent usage.
    """
    _instance = None
    _lock = threading.Lock()
    _keys: List[str] = []
    _current_index: int = 0
    _key_errors: Dict[str, int] = {}  # Track errors per key
    _max_errors_per_key: int = 3  # Switch key after this many failures
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(APIKeyManager, cls).__new__(cls)
                    cls._instance._load_keys()
        return cls._instance
    
    def _load_keys(self):
        """Load all available Gemini API keys from environment."""
        keys = []
        # Primary key
        primary = os.getenv("GEMINI_API_KEY")
        if primary:
            keys.append(primary.strip())
        
        # Additional keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
        idx = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{idx}")
            if not key:
                break
            keys.append(key.strip())
            idx += 1
        
        if not keys:
            raise EnvironmentError("No GEMINI_API_KEY found in environment variables.")
        
        self._keys = keys
        self._key_errors = {key: 0 for key in keys}
        logging.info(f"[APIKeyManager] Loaded {len(keys)} Gemini API key(s)")
    
    def get_key(self) -> str:
        """Get the next API key in round-robin order for every call (parallel safe)."""
        if not self._keys:
            raise EnvironmentError("No API keys available.")
        with self._lock:
            key = self._keys[self._current_index % len(self._keys)]
            self._current_index = (self._current_index + 1) % len(self._keys)
            return key
    
    def rotate_key(self, failed_key: Optional[str] = None):
        """Rotate to the next available key."""
        with self._lock:
            if failed_key:
                # Increment error count for failed key
                if failed_key in self._key_errors:
                    self._key_errors[failed_key] += 1
                    logging.warning(f"[APIKeyManager] Key error count: {self._key_errors[failed_key]}/{self._max_errors_per_key}")
            
            # Find next healthy key
            original_index = self._current_index
            attempts = 0
            while attempts < len(self._keys):
                self._current_index = (self._current_index + 1) % len(self._keys)
                current_key = self._keys[self._current_index]
                
                # Use key if error count is below threshold
                if self._key_errors.get(current_key, 0) < self._max_errors_per_key:
                    if self._current_index != original_index:
                        logging.info(f"[APIKeyManager] Rotated to key index {self._current_index}")
                    return current_key
                
                attempts += 1
            
            # Reset error counts if all keys have errors (they may have recovered)
            if attempts >= len(self._keys):
                logging.warning("[APIKeyManager] All keys have errors, resetting error counts")
                self._key_errors = {key: 0 for key in self._keys}
                self._current_index = (self._current_index + 1) % len(self._keys)
            
            return self._keys[self._current_index]
    
    def mark_success(self, key: str):
        """Reset error count on successful API call."""
        if key in self._key_errors:
            self._key_errors[key] = 0
    
    def get_next_key(self) -> str:
        """Get next key in rotation (for load balancing)."""
        with self._lock:
            key = self.get_key()
            self._current_index = (self._current_index + 1) % len(self._keys)
            return key


# ==========================================================
# Utility Models and Constants
# ==========================================================
class LLMResponse(BaseModel):
    """Standardized response model for LLM output."""
    text: str = Field(default="")
    parsed_json: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    cost_estimate_usd: float = 0.0


# ==========================================================
# BaseAgent Definition
# ==========================================================
class BaseAgent(ABC):
    """
    Abstract base agent for all HLD generation components.
    Handles:
      - API communication
      - JSON parsing and normalization
      - Retry and rate limiting
      - Prompt management
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 8192,  # Increased for better output generation
        retries: int = 5,  # Increased retries for multi-key support
        backoff_factor: float = 2.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.key_manager = APIKeyManager()
        
        # Log agent initialization
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("=" * 80)
        logger.info(f"{self.__class__.__name__} INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max Output Tokens: {max_output_tokens}")
        logger.info(f"Retry Attempts: {retries}")
        logger.info(f"API Keys Available: {len(self.key_manager._keys)}")
        logger.info("=" * 80)
        
        # Configure Gemini API with first key (will rotate on errors)
        current_key = self.key_manager.get_key()
        genai.configure(api_key=current_key)

    # ======================================================
    # Abstract Methods
    # ======================================================
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Base system prompt defining the role and context of the agent."""
        ...

    @abstractmethod
    def process(self, state: Any) -> Dict[str, Any]:
        """Main method each derived agent implements to transform the workflow state."""
        ...

    # ======================================================
    # LLM Invocation
    # ======================================================
    def call_llm(self, prompt: str) -> LLMResponse:
        """
        Make a call to Gemini with retry, backoff, and multi-key rotation.
        Automatically rotates keys on rate limit or quota errors.
        Returns structured LLMResponse.
        """
        delay = 1.0
        last_error: Optional[Exception] = None
        current_key = self.key_manager.get_key()
        keys_tried = set()
        
        # Get logger for this agent class
        logger = logging.getLogger(self.__class__.__name__)

        for attempt in range(1, self.retries + 1):
            try:
                # Get current API key and configure
                current_key = self.key_manager.get_key()
                genai.configure(api_key=current_key)
                keys_tried.add(current_key)
                
                # Mask API key for logging (show only first/last 4 chars)
                masked_key = f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else "***"
                logger.info(f"Calling Gemini API (attempt {attempt}/{self.retries}, key: {masked_key}, key index: {self.key_manager._current_index})")
                
                call_start = time.time()
                model = genai.GenerativeModel(self.model_name)
                
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                    },
                )
                
                call_duration = time.time() - call_start

                text = response.text or ""
                parsed = self.parse_json_loose(text)
                tokens_used = len(text.split())
                cost_estimate = tokens_used * 0.000002  # rough cost estimation

                # Mark key as successful
                self.key_manager.mark_success(current_key)
                
                logger.info(f"✓ API call successful! Duration: {call_duration:.2f}s, Response: {len(text)} chars, {tokens_used} tokens, Cost: ${cost_estimate:.6f}")
                
                return LLMResponse(
                    text=text.strip(),
                    parsed_json=parsed,
                    tokens_used=tokens_used,
                    cost_estimate_usd=round(cost_estimate, 6),
                )

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a rate limit or quota error
                is_rate_limit = any(keyword in error_msg for keyword in [
                    "rate limit", "quota", "resource exhausted", 
                    "429", "too many requests", "quota exceeded",
                    "permission denied", "api key"
                ])
                
                if is_rate_limit or attempt > 1:
                    # Rotate to next key on rate limit or if retrying
                    logger.warning(f"✗ API error (rate limit? {is_rate_limit}): {e}")
                    logger.info(f"Rotating to next API key...")
                    self.key_manager.rotate_key(failed_key=current_key)
                    current_key = self.key_manager.get_key()
                    
                    # If we've tried all keys, wait longer
                    if len(keys_tried) >= len(self.key_manager._keys):
                        keys_tried.clear()  # Reset and try again
                        delay = max(delay, 5.0)  # Longer delay after trying all keys
                
                logger.warning(f"Gemini API call failed (attempt {attempt}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * (self.backoff_factor + random.random() * 0.5), 30.0)  # Cap delay at 30s

        logging.error(f"[BaseAgent] API failed after {self.retries} retries with all available keys: {last_error}")
        raise RuntimeError(f"LLM call failed after trying all keys: {last_error}")

    # ======================================================
    # JSON Parsing Utilities
    # ======================================================
    @staticmethod
    def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON robustly from a model response.
        Tries multiple fallback strategies to extract JSON objects.
        """
        if not text:
            return None

        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback 1: extract text between braces
        if "{" in text and "}" in text:
            candidate = text[text.find("{"): text.rfind("}") + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Fallback 2: manually fix common JSON issues
        cleaned = text.replace("'", '"').replace("\n", " ")
        try:
            return json.loads(cleaned)
        except Exception:
            logging.warning("[BaseAgent] Could not extract valid JSON from model response.")
            return None

    # ======================================================
    # Normalization Utilities
    # ======================================================
    @staticmethod
    def normalize_string(value: Optional[str]) -> str:
        if not value:
            return ""
        return " ".join(value.strip().split())

    @staticmethod
    def normalize_list(values: Optional[List[Any]]) -> List[Any]:
        if not values:
            return []
        if isinstance(values, str):
            return [values.strip()]
        return [v for v in values if v is not None and str(v).strip()]

    @staticmethod
    def normalize_dict(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not data:
            return {}
        return {str(k).strip(): v for k, v in data.items() if k is not None}

    # ======================================================
    # Logging & Caching Utilities
    # ======================================================
    def log_cost(self, response: LLMResponse):
        logging.info(
            f"[LLM] Tokens: {response.tokens_used}, Cost: ${response.cost_estimate_usd:.6f}"
        )

    def cached_prompt_path(self) -> Path:
        """Optional: path for storing system prompts for inspection."""
        base = Path("cache/prompts")
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.__class__.__name__}.txt"

    def save_system_prompt(self):
        """Save the system prompt to a cache file (optional for debugging)."""
        path = self.cached_prompt_path()
        path.write_text(self.system_prompt, encoding="utf-8")
        logging.info(f"Saved system prompt: {path}")
