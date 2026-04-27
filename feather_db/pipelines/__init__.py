"""End-to-end ingestion pipelines built on the extractor primitives."""
from .ingest import IngestPipeline, IngestRecord, IngestStats

__all__ = ["IngestPipeline", "IngestRecord", "IngestStats"]
