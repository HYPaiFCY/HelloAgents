import pytest
from hello_agents.memory.rag.pipeline import AdvancedRAGPipeline


class FakeStore:
    def __init__(self):
        self.stats = {"vectors_count": 3}

    def search_similar(self, query_vector, limit=10, score_threshold=None, where=None):
        items = [
            {
                "id": "a1",
                "score": 0.95,
                "metadata": {
                    "memory_id": "a1",
                    "doc_id": "doc-a",
                    "source_path": "a.md",
                    "start": 0,
                    "end": 20,
                    "content": "Alpha first chunk",
                    "memory_type": "rag_chunk",
                },
            },
            {
                "id": "a2",
                "score": 0.90,
                "metadata": {
                    "memory_id": "a2",
                    "doc_id": "doc-a",
                    "source_path": "a.md",
                    "start": 30,
                    "end": 60,
                    "content": "Alpha nearby chunk",
                    "memory_type": "rag_chunk",
                },
            },
            {
                "id": "b1",
                "score": 0.80,
                "metadata": {
                    "memory_id": "b1",
                    "doc_id": "doc-b",
                    "source_path": "b.md",
                    "start": 0,
                    "end": 15,
                    "content": "Beta chunk",
                    "memory_type": "rag_chunk",
                },
            },
        ]
        return items[:limit]

    def get_collection_stats(self):
        return self.stats


def test_advanced_rag_pipeline_answer_context(monkeypatch):
    monkeypatch.setattr(
        "hello_agents.memory.rag.pipeline.embed_query",
        lambda query: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "hello_agents.memory.rag.pipeline.rerank_with_cross_encoder",
        lambda query, items, model_name, top_k: items[:top_k],
    )

    pipeline = AdvancedRAGPipeline(store=FakeStore(), rag_namespace="test")
    result = pipeline.answer_context(
        "alpha",
        top_k=2,
        max_chars=500,
        include_citations=True,
        rerank=True,
        enable_mqe=False,
        enable_hyde=False,
    )

    assert result["namespace"] == "test"
    assert result["items"]
    assert "Alpha" in result["context"]
    assert "References:" in result["context"]
    assert pipeline.get_stats() == {"vectors_count": 3}

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])