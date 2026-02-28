from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.rag.models import Citation, RAGResponse, ScoredChunk, Chunk


@pytest.fixture
def client():
    """Create a test client with a mocked pipeline."""
    with patch("src.rag.api.pipeline") as mock_pipe:
        mock_pipe.is_ready = True
        mock_pipe.chunk_count = 42

        from src.rag.api import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_pipe


def test_health(client):
    c, mock_pipe = client
    resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"


def test_query_success(client):
    c, mock_pipe = client
    mock_pipe.query.return_value = RAGResponse(
        answer="The answer is 42 [1].",
        citations=[Citation(ref_id=1, source="doc.md", title="Doc")],
        chunks_used=[
            ScoredChunk(
                chunk=Chunk(chunk_id="c1", text="context", source="doc.md"),
                score=0.9,
            )
        ],
        query="What is the answer?",
    )

    resp = c.post("/query", json={"query": "What is the answer?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "42" in data["answer"]
    assert len(data["citations"]) == 1


def test_query_not_ready(client):
    c, mock_pipe = client
    mock_pipe.is_ready = False

    resp = c.post("/query", json={"query": "test"})
    assert resp.status_code == 503


def test_query_empty_string(client):
    c, _ = client
    resp = c.post("/query", json={"query": ""})
    assert resp.status_code == 422  # Validation error
