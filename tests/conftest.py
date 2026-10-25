"""Shared pytest configuration for the CLDB test suite.

Forces HuggingFace/transformers into offline mode before any module imports
them, so tests never block on a network download. The embedder falls back to
its deterministic hashing backend when the model is not cached locally.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
