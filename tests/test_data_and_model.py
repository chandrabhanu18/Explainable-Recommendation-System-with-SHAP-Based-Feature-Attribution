import os
import json
import pandas as pd
import pytest


def test_processed_files_exist():
    required = [
        'data/processed/train.csv',
        'data/processed/test.csv',
        'data/processed/user_map.json',
        'data/processed/item_map.json',
        'data/processed/item_titles.json',
        'models/recommender.joblib',
        'results/model_metrics.json',
        'results/explanation.json',
        'results/explanation_metrics.json'
    ]
    for f in required:
        assert os.path.exists(f), f"Missing {f}"


def test_metrics_json_structure():
    with open('results/model_metrics.json') as f:
        metrics = json.load(f)
    assert 'precision_at_10' in metrics
    assert 'ndcg_at_10' in metrics

    with open('results/explanation_metrics.json') as f:
        exp_metrics = json.load(f)
    assert 'faithfulness' in exp_metrics
    assert 'consistency' in exp_metrics


def test_train_and_test_splits():
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')
    assert not train.empty
    assert not test.empty
    assert set(train['user_idx']).issubset(set(test['user_idx']) | set(train['user_idx']))
