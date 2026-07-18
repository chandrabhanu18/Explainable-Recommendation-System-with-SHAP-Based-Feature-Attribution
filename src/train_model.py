"""Train baseline (popularity) and Matrix Factorization model.
Saves model artifact and metrics to disk.
"""
import os
import argparse
import json
import sys
import numpy as np
import pandas as pd
from joblib import dump
from tqdm import trange

# Ensure the repository root is on PYTHONPATH when running scripts directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import precision_at_k, ndcg_at_k


def train_funk_svd(train_df, num_users, num_items, factors=32, epochs=10, lr=0.01, reg=0.01):
    # initialize
    rng = np.random.RandomState(42)
    U = 0.1 * rng.randn(num_users, factors)
    V = 0.1 * rng.randn(num_items, factors)

    # training via SGD on explicit ratings
    interactions = train_df[['user_idx','item_idx','rating']].values
    for ep in range(epochs):
        perm = rng.permutation(len(interactions))
        for idx in perm:
            u,i,r = interactions[idx]
            pred = U[u].dot(V[i])
            err = (r - pred)
            U[u] += lr * (err * V[i] - reg * U[u])
            V[i] += lr * (err * U[u] - reg * V[i])
    return U, V


def recommend_scores(u_idx, U, V, train_user_items, top_k=10):
    scores = V.dot(U[u_idx])
    # exclude training items
    seen = train_user_items.get(u_idx, set())
    candidate_idxs = [i for i in np.argsort(-scores) if i not in seen]
    return candidate_idxs[:top_k]


def evaluate(recommendations, test_df):
    precisions = []
    ndcgs = []
    for u, recs in recommendations.items():
        true_items = test_df.loc[test_df['user_idx']==u, 'item_idx'].tolist()
        if len(true_items) == 0:
            continue
        precisions.append(precision_at_k(recs, true_items, k=10))
        ndcgs.append(ndcg_at_k(recs, true_items, k=10))
    precision_at_10 = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    ndcg_at_10 = float(np.mean(ndcgs)) if len(ndcgs) > 0 else 0.0
    return precision_at_10, ndcg_at_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/processed')
    parser.add_argument('--models-dir', default='./models')
    parser.add_argument('--results-dir', default='./results')
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir,'train.csv'))
    test = pd.read_csv(os.path.join(args.data_dir,'test.csv'))

    num_users = int(max(train['user_idx'].max(), test['user_idx'].max())+1)
    num_items = int(max(train['item_idx'].max(), test['item_idx'].max())+1)

    # baseline: popularity
    popular = train.groupby('item_idx').size().sort_values(ascending=False).index.tolist()
    # build train_user_items map
    train_user_items = train.groupby('user_idx')['item_idx'].apply(set).to_dict()

    # baseline recommendations
    baseline_recs = {u: [i for i in popular if i not in train_user_items.get(u, set())][:10] for u in train['user_idx'].unique()}
    p10_base, ndcg_base = evaluate(baseline_recs, test)
    baseline_metrics = {"precision_at_10": float(p10_base), "ndcg_at_10": float(ndcg_base)}
    with open(os.path.join(args.results_dir,'baseline_metrics.json'),'w') as f:
        json.dump(baseline_metrics, f)

    # train MF
    U, V = train_funk_svd(train, num_users, num_items, factors=args.factors, epochs=args.epochs, lr=args.lr)

    # generate recommendations for each user in test
    recommendations = {}
    user_list = test['user_idx'].unique()
    for u in user_list:
        recs = recommend_scores(int(u), U, V, train_user_items, top_k=10)
        recommendations[int(u)] = recs

    p10_model, ndcg_model = evaluate(recommendations, test)
    model_metrics = {"precision_at_10": float(p10_model), "ndcg_at_10": float(ndcg_model)}
    with open(os.path.join(args.results_dir,'model_metrics.json'),'w') as f:
        json.dump(model_metrics, f)

    # save model artifact
    model_obj = {
        'user_factors': U,
        'item_factors': V,
    }
    dump(model_obj, os.path.join(args.models_dir,'recommender.joblib'))
    print('Saved model and metrics')


if __name__ == '__main__':
    main()
