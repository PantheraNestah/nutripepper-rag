# app/database/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, Integer, Float,
    Boolean, DateTime, ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


class EndpointType(enum.Enum):
    query = "query"
    learn = "learn"


class Interaction(Base):
    """
    Top-level record for every request hitting /query, /learn/structured, or /learn/text.
    One row per request regardless of endpoint type.
    """
    __tablename__ = "interactions"

    id                = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id        = Column(String(36), nullable=True, index=True)
    endpoint          = Column(SAEnum(EndpointType), nullable=False)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    latency_ms        = Column(Float, nullable=True)
    is_conversational = Column(Boolean, default=False, nullable=False)

    query_detail     = relationship("QueryDetail",  back_populates="interaction", uselist=False, cascade="all, delete-orphan")
    learn_detail     = relationship("LearnDetail",  back_populates="interaction", uselist=False, cascade="all, delete-orphan")
    retrieved_chunks = relationship("RetrievedChunk", back_populates="interaction", cascade="all, delete-orphan")


class QueryDetail(Base):
    """
    Payload specific to /query interactions.
    Includes Phase 1 instrumentation fields for grounding evaluation.
    """

    __tablename__ = "query_details"

    id                 = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id     = Column(String(36), ForeignKey("interactions.id"), nullable=False, unique=True)
    original_query     = Column(Text, nullable=False)
    processed_query    = Column(Text, nullable=False)
    generated_response = Column(Text, nullable=False)

    # ── Phase 1 instrumentation ───────────────────────────────────────────────
    citations_present  = Column(Boolean, nullable=True)   # True if response contains [1], [2] etc.
    citation_count     = Column(Integer, nullable=True)   # how many inline citations found
    grounding_score    = Column(Float,   nullable=True)   # 0.0–1.0 from verifier (Phase 2)
    grounding_label    = Column(String(20), nullable=True) # "grounded" | "partial" | "ungrounded"
    has_contradiction  = Column(Boolean, nullable=True)   # True if any claim contradicts sources
    claims_checked     = Column(Integer, nullable=True)   # total claims extracted and checked
    claims_supported   = Column(Integer, nullable=True)   # claims verified as entailed by context


    interaction = relationship("Interaction", back_populates="query_detail")


class LearnDetail(Base):
    """
    Stores lesson output for both /learn/structured and /learn/text.
    - lesson_mode distinguishes which endpoint generated the lesson.
    - lesson_json stores structured beat content (mode: structured).
    - lesson_text stores the full narrative paragraphs (mode: text).
    - quiz_hints stores comma-separated key terms for future quiz generation.
    """
    __tablename__ = "learn_details"
 
    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(36), ForeignKey("interactions.id"), nullable=False, unique=True)
    concept        = Column(Text, nullable=False)
    lesson_mode    = Column(String(20), nullable=True)    # "structured" | "text"
    sources_used   = Column(Text, nullable=True)
 
    # Structured lesson fields
    total_beats    = Column(Integer, nullable=True)
    beat_types     = Column(Text, nullable=True)
    lesson_json    = Column(Text, nullable=True)
 
    # Text lesson fields
    lesson_text    = Column(Text, nullable=True)
    quiz_hints     = Column(Text, nullable=True)
    description    = Column(Text, nullable=True)   # optional user-provided focus/audience guidance
 
    interaction = relationship("Interaction", back_populates="learn_detail")

class RetrievedChunk(Base):
    """
    Each document chunk retrieved from ChromaDB for an interaction.
    """
    __tablename__ = "retrieved_chunks"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(36), ForeignKey("interactions.id"), nullable=False, index=True)
    chunk_index    = Column(Integer, nullable=False)
    source         = Column(String(255), nullable=True)
    authority      = Column(String(255), nullable=True)
    domain         = Column(String(100), nullable=True)
    text_preview   = Column(Text, nullable=True)

    interaction = relationship("Interaction", back_populates="retrieved_chunks")