import pytest
import sys
import os
import json

# Add parent dir to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the /api/health endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
    assert "service" in data

def test_home_page(client):
    """Test that the home page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"AI Multimodal Search" in response.data

def test_search_no_query(client):
    """Test search with missing query."""
    response = client.post('/api/search', json={})
    assert response.status_code == 400 or response.status_code == 503
    # Note: might be 503 if index not loaded, or 400 if loaded but no query
