"""
CL Engine - Production-grade continual learning engine for index optimization.
Uses PyTorch with EWC and replay buffer for catastrophic forgetting prevention.
"""
from cldb.enhanced_cl_engine import CLEngine, EnhancedCLEngine, EMBEDDING_DIM, CONTEXT_DIM
from cldb.enhanced_cl_engine import ImprovedPolicyNetwork, ResidualPolicyNetwork, AttentionPolicyNetwork

__all__ = [
    "CLEngine",
    "EnhancedCLEngine",
    "ImprovedPolicyNetwork", 
    "ResidualPolicyNetwork",
    "AttentionPolicyNetwork",
    "EMBEDDING_DIM",
    "CONTEXT_DIM",
]