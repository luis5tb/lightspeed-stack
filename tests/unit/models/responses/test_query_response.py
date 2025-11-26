"""Unit tests for QueryResponse model."""

from models.responses import QueryResponse, ReferencedDocument
from utils.types import RAGChunk, ToolCallSummary, ToolResultSummary


class TestQueryResponse:
    """Test cases for the QueryResponse model."""

    def test_constructor(self) -> None:
        """Test the QueryResponse constructor."""
        qr = QueryResponse(
            conversation_id="123e4567-e89b-12d3-a456-426614174000",
            response="LLM answer",
        )
        assert qr.conversation_id == "123e4567-e89b-12d3-a456-426614174000"
        assert qr.response == "LLM answer"

    def test_optional_conversation_id(self) -> None:
        """Test the QueryResponse with default conversation ID."""
        qr = QueryResponse(response="LLM answer")
        assert qr.conversation_id is None
        assert qr.response == "LLM answer"

    def test_rag_chunks_empty_by_default(self) -> None:
        """Test that rag_chunks is empty by default."""
        qr = QueryResponse(response="LLM answer")
        assert not qr.rag_chunks

    def test_rag_chunks_with_data(self) -> None:
        """Test QueryResponse with RAG chunks."""
        rag_chunks = [
            RAGChunk(
                content="Kubernetes is an open-source container orchestration system",
                source="kubernetes-docs/overview.md",
                score=0.95,
            ),
            RAGChunk(
                content="Container orchestration automates deployment and management",
                source="kubernetes-docs/concepts.md",
                score=0.87,
            ),
        ]

        qr = QueryResponse(
            conversation_id="123e4567-e89b-12d3-a456-426614174000",
            response="LLM answer with RAG context",
            rag_chunks=rag_chunks,
        )

        assert len(qr.rag_chunks) == 2
        assert (
            qr.rag_chunks[0].content
            == "Kubernetes is an open-source container orchestration system"
        )
        assert qr.rag_chunks[0].source == "kubernetes-docs/overview.md"
        assert qr.rag_chunks[0].score == 0.95
        assert (
            qr.rag_chunks[1].content
            == "Container orchestration automates deployment and management"
        )
        assert qr.rag_chunks[1].source == "kubernetes-docs/concepts.md"
        assert qr.rag_chunks[1].score == 0.87

    def test_rag_chunks_with_optional_fields(self) -> None:
        """Test RAG chunks with optional source and score fields."""
        rag_chunks = [
            RAGChunk(content="Some content without source or score"),
            RAGChunk(content="Content with source only", source="docs/guide.md"),
            RAGChunk(content="Content with score only", score=0.75),
        ]

        qr = QueryResponse(response="LLM answer", rag_chunks=rag_chunks)

        assert len(qr.rag_chunks) == 3
        assert qr.rag_chunks[0].source is None
        assert qr.rag_chunks[0].score is None
        assert qr.rag_chunks[1].source == "docs/guide.md"
        assert qr.rag_chunks[1].score is None
        assert qr.rag_chunks[2].source is None
        assert qr.rag_chunks[2].score == 0.75

    def test_complete_query_response_with_all_fields(self) -> None:
        """Test QueryResponse with all fields including RAG chunks, tool calls, and docs."""
        rag_chunks = [
            RAGChunk(
                content="OLM is a component of the Operator Framework toolkit",
                source="kubernetes-docs/operators.md",
                score=0.95,
            )
        ]

        tool_calls = [
            ToolCallSummary(
                id="call-1",
                name="knowledge_search",
                args={"query": "operator lifecycle manager"},
                type="tool_call",
            )
        ]
        tool_results = [
            ToolResultSummary(
                id="call-1",
                status="success",
                content={"chunks_found": 5},
                type="tool_result",
                round=1,
            )
        ]

        referenced_documents = [
            ReferencedDocument(
                doc_url=(
                    "https://docs.openshift.com/container-platform/4.15/operators/olm/index.html"
                ),
                doc_title="Operator Lifecycle Manager (OLM)",
            )
        ]

        qr = QueryResponse(
            conversation_id="123e4567-e89b-12d3-a456-426614174000",
            response="Operator Lifecycle Manager (OLM) helps users install...",
            rag_chunks=rag_chunks,
            tool_calls=tool_calls,
            tool_results=tool_results,
            referenced_documents=referenced_documents,
        )

        assert qr.conversation_id == "123e4567-e89b-12d3-a456-426614174000"
        assert qr.response == "Operator Lifecycle Manager (OLM) helps users install..."
        assert len(qr.rag_chunks) == 1
        assert (
            qr.rag_chunks[0].content
            == "OLM is a component of the Operator Framework toolkit"
        )
        assert len(qr.tool_calls) == 1
        assert qr.tool_calls[0].tool_name == "knowledge_search"
        assert len(qr.referenced_documents) == 1
        assert (
            qr.referenced_documents[0].doc_title == "Operator Lifecycle Manager (OLM)"
        )
