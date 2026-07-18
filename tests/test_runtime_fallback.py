import os
import json
import shutil
import pandas as pd
import pytest

from src.generate_explanation import explain_user_item


def test_explain_creates_missing_artifacts(tmp_path):
    processed_dir = tmp_path / 'processed'
    models_dir = tmp_path / 'models'
    results_dir = tmp_path / 'results'

    result = explain_user_item(
        1,
        1,
        processed_dir=str(processed_dir),
        models_dir=str(models_dir),
        results_dir=str(results_dir),
        save=True,
    )

    assert result['user_id'] == 1
    assert result['item_id'] == 1
    assert 'recommendation_score' in result
    assert os.path.exists(results_dir / 'explanation.json')
    assert os.path.exists(processed_dir / 'train.csv')
    assert os.path.exists(models_dir / 'recommender.joblib')
