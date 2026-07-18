"""Evaluate explanation quality: faithfulness and consistency.
Saves /results/explanation_metrics.json
"""
import os
import json
import random
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import shap
from tqdm import tqdm

def get_shap_contributors(u_idx, target_idx, user_hist_map, U, V, top_k=3):
    hist = list(map(int, user_hist_map.get(u_idx, [])))
    m = len(hist)
    if m == 0:
        return []

    # Vectorized prediction function for SHAP
    hist_embs = V[hist]
    item_vec = V[target_idx]
    mean_user_emb = np.mean(U, axis=0)

    def pred_fn(batch_X):
        row_sums = batch_X.sum(axis=1, keepdims=True)
        denom = np.where(row_sums == 0, 1.0, row_sums)
        user_embs = (batch_X @ hist_embs) / denom

        zero_rows = (row_sums.ravel() == 0)
        if np.any(zero_rows):
            user_embs[zero_rows] = mean_user_emb

        return user_embs @ item_vec

    background = np.zeros((1, m))
    explainer = shap.KernelExplainer(pred_fn, background, silent=True)
    instance = np.ones((1, m))

    # To keep it extremely fast, sample fewer points while keeping SHAP valid
    nsamples = min(100, 2 * m + 20)
    shap_values = explainer.shap_values(instance, nsamples=nsamples, silent=True)

    sv = np.array(shap_values).reshape(-1)[:m]

    # Zip history item indices with their SHAP values
    contributors = []
    for idx, val in zip(hist, sv):
        contributors.append((idx, val))

    # Sort by positive SHAP values descending
    pos_contributors = sorted([c for c in contributors if c[1] > 0], key=lambda x: -x[1])
    return [c[0] for c in pos_contributors[:top_k]]

def main():
    processed = './data/processed'
    models_dir = './models'
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(processed, 'train.csv'))
    test = pd.read_csv(os.path.join(processed, 'test.csv'))
    model = load(os.path.join(models_dir, 'recommender.joblib'))
    U = model['user_factors']
    V = model['item_factors']

    # build user histories
    user_hist_map = train.groupby('user_idx')['item_idx'].apply(list).to_dict()

    # Faithfulness
    # Filter test set to users with at least 3 history items to ensure we can sample
    valid_test = test[test['user_idx'].isin([u for u, hist in user_hist_map.items() if len(hist) >= 3])]
    
    if len(valid_test) == 0:
        print("Warning: no users with history length >= 3 in test set. Relaxing constraint to history length >= 1.")
        valid_test = test[test['user_idx'].isin([u for u, hist in user_hist_map.items() if len(hist) >= 1])]

    samples = valid_test.sample(n=min(100, len(valid_test)), random_state=42)
    faithful_count = 0
    total_faithful_evaluated = 0

    print("Evaluating faithfulness...")
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        u = int(row['user_idx'])
        target = int(row['item_idx'])
        hist = list(map(int, user_hist_map.get(u, [])))
        
        top3 = get_shap_contributors(u, target, user_hist_map, U, V, top_k=3)
        k = len(top3)
        if k == 0:
            continue

        # original score
        orig_emb = V[hist].mean(axis=0) if len(hist) > 0 else np.mean(U, axis=0)
        orig_score = float(np.dot(orig_emb, V[target]))

        # remove top3
        remaining = [h for h in hist if h not in top3]
        emb_removed = V[remaining].mean(axis=0) if len(remaining) > 0 else np.mean(U, axis=0)
        score_removed = float(np.dot(emb_removed, V[target]))
        delta_shap = orig_score - score_removed

        # remove k random
        rand_removed = random.sample(hist, k)
        rem2 = [h for h in hist if h not in rand_removed]
        emb_rand = V[rem2].mean(axis=0) if len(rem2) > 0 else np.mean(U, axis=0)
        score_rand = float(np.dot(emb_rand, V[target]))
        delta_rand = orig_score - score_rand

        if delta_shap > delta_rand:
            faithful_count += 1
        total_faithful_evaluated += 1

    faithfulness = float(faithful_count) / max(1, total_faithful_evaluated)

    # Consistency
    print("Evaluating consistency...")
    # build user-item binary matrix sample
    users = list(user_hist_map.keys())
    # compute pairwise similarities on a small sample
    sampled_users = users[:500]
    # build sparse interaction vectors
    num_items = V.shape[0]
    user_vecs = []
    for u in sampled_users:
        vec = np.zeros(num_items, dtype=int)
        for it in user_hist_map.get(u, []):
            vec[int(it)] = 1
        user_vecs.append(vec)
    user_vecs = np.array(user_vecs)
    sims = cosine_similarity(user_vecs)
    
    pairs = []
    n = len(sampled_users)
    for i in range(n):
        for j in range(i+1, n):
            if sims[i, j] > 0.5:
                pairs.append((sampled_users[i], sampled_users[j]))
            if len(pairs) >= 50:
                break
        if len(pairs) >= 50:
            break

    jaccards = []
    for (u1, u2) in tqdm(pairs, total=len(pairs)):
        # find a common recommended item
        hist1 = list(map(int, user_hist_map.get(u1, [])))
        hist2 = list(map(int, user_hist_map.get(u2, [])))
        if len(hist1) == 0 or len(hist2) == 0:
            continue
        
        # pick a target item from items not in their histories (a candidate)
        # generate candidate recommendations via score
        emb1 = V[hist1].mean(axis=0)
        scores1 = V.dot(emb1)
        top1 = list(np.argsort(-scores1)[:100])
        
        emb2 = V[hist2].mean(axis=0)
        scores2 = V.dot(emb2)
        top2 = list(np.argsort(-scores2)[:100])
        
        common = set(top1) & set(top2)
        if len(common) == 0:
            continue
        
        target = list(common)[0]
        top3_1 = set(get_shap_contributors(u1, target, user_hist_map, U, V, top_k=3))
        top3_2 = set(get_shap_contributors(u2, target, user_hist_map, U, V, top_k=3))
        if len(top3_1) == 0 and len(top3_2) == 0:
            continue
        jacc = float(len(top3_1 & top3_2) / max(1, len(top3_1 | top3_2)))
        jaccards.append(jacc)

    consistency = float(np.mean(jaccards)) if len(jaccards) > 0 else 0.0

    out = {"faithfulness": float(faithfulness), "consistency": float(consistency)}
    with open(os.path.join(results_dir, 'explanation_metrics.json'), 'w') as f:
        json.dump(out, f)
    print('Saved explanation metrics to', os.path.join(results_dir, 'explanation_metrics.json'))


if __name__ == '__main__':
    main()
