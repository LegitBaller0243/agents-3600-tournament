import numpy as np
import math

def softmax_sample(visit_counts):
    """
    Sample an action from visit counts using softmax.
    visit_counts: list of (visit_count, action) tuples
    Returns: (visit_count, action) tuple
    """
    if not visit_counts:
        return None, None
    
    counts, actions = zip(*visit_counts)
    counts = np.array(counts, dtype=np.float32)
    
    # Apply softmax
    exp_counts = np.exp(counts - np.max(counts))  # Numerical stability
    probs = exp_counts / np.sum(exp_counts)
    
    # Sample according to probabilities
    idx = np.random.choice(len(actions), p=probs)
    return visit_counts[idx]

