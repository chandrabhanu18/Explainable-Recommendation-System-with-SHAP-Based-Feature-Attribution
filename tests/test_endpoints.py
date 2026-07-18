import os
import json
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_explain_endpoint():
    resp = client.post('/explain', json={'user_id': 1, 'item_id': 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data['user_id'] == 1
    assert data['item_id'] == 1
    assert 'recommendation_score' in data
    assert 'top_contributors' in data
    assert 'human_readable_explanation' in data
