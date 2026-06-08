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
    Top-level record for every request hitting /query or /learn.
    One row per request regardless of endpoint type.
    """
    __tablename__ = "interactions"

    id                = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id        = Column(String(36), nullable=True, index=True)  # groups multi-turn convos
    endpoint          = Column(SAEnum(EndpointType), nullable=False)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    latency_ms        = Column(Float, nullable=True)                   # end-to-end response time
    is_conversational = Column(Boolean, default=False, nullable=False) # True = skipped retrieval

    # Relationships
    query_detail  = relationship("QueryDetail",  back_populates="interaction", uselist=False, cascade="all, delete-orphan")
    learn_detail  = relationship("LearnDetail",  back_populates="interaction", uselist=False, cascade="all, delete-orphan")
    retrieved_chunks = relationship("RetrievedChunk", back_populates="interaction", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Interaction id={self.id} endpoint={self.endpoint.value} at={self.created_at}>"


class QueryDetail(Base):
    """
    Payload specific to /query interactions.
    """
    __tablename__ = "query_details"

    id                 = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id     = Column(String(36), ForeignKey("interactions.id"), nullable=False, unique=True)
    original_query     = Column(Text, nullable=False)
    processed_query    = Column(Text, nullable=False)
    generated_response = Column(Text, nullable=False)

    interaction = relationship("Interaction", back_populates="query_detail")


class LearnDetail(Base):
    """
    Payload specific to /learn interactions.
    """
    __tablename__ = "learn_details"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(36), ForeignKey("interactions.id"), nullable=False, unique=True)
    concept        = Column(Text, nullable=False)
    total_beats    = Column(Integer, nullable=True)
    beat_types     = Column(Text, nullable=True)  # comma-separated e.g. "introduction,kenyan_context,quiz"
    sources_used   = Column(Text, nullable=True)  # comma-separated source filenames

    interaction = relationship("Interaction", back_populates="learn_detail")


class RetrievedChunk(Base):
    """
    Each document chunk retrieved from ChromaDB for an interaction.
    Stored as child rows so you can query retrieval patterns across interactions.
    """
    __tablename__ = "retrieved_chunks"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(36), ForeignKey("interactions.id"), nullable=False, index=True)
    chunk_index    = Column(Integer, nullable=False)   # position in retrieval results (0-based)
    source         = Column(String(255), nullable=True)
    authority      = Column(String(255), nullable=True)
    domain         = Column(String(100), nullable=True)
    text_preview   = Column(Text, nullable=True)       # first 500 chars only — saves space

    interaction = relationship("Interaction", back_populates="retrieved_chunks")