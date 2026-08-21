"""RAGLean — a cost/latency optimization layer for production RAG pipelines.

Pipeline: semantic cache -> complexity router -> context compressor -> provider,
with a budget tracker and eval harness wrapped around it.

See docs/ARCHITECTURE.md for the full design.
"""

__version__ = "0.1.0"
