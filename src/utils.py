import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    if len(relevant) == 0:
        return 0.0
    return float(len(set(recommended_k) & set(relevant)))/k

def ndcg_at_k(recommended, relevant, k=10):
    # sklearn's ndcg_score expects relevance matrix
    y_true = np.array([[1 if i in relevant else 0 for i in recommended[:k]]])
    y_score = np.array([[1.0/(idx+1) for idx in range(len(recommended[:k]))]])
    try:
        return float(ndcg_score(y_true, y_score))
    except Exception:
        return 0.0
