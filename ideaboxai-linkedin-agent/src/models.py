from datetime import datetime

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class PersonProfile(Base):
    __tablename__ = "person_profiles"
    linkedin_urn = Column(String(255), primary_key=True)
    name = Column(String(255))
    company = Column(String(255), nullable=True)
    title = Column(String(255), nullable=True)
    is_vip = Column(Boolean, default=False)
    vip_tier = Column(String(50), nullable=True)
    last_profile_fetch = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float, default=0.0)


class EngagementEvent(Base):
    __tablename__ = "engagement_events"
    id = Column(Integer, primary_key=True)
    linkedin_urn = Column(String(255))
    linkedin_comment_urn = Column(String(255), unique=True)
    engagement_type = Column(String(50))
    engagement_text = Column(Text)
    engagement_timestamp = Column(DateTime)
    received_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    escalated = Column(Boolean, default=False)


class ReplyLog(Base):
    __tablename__ = "reply_log"
    id = Column(Integer, primary_key=True)
    engagement_id = Column(Integer)
    linkedin_urn = Column(String(255))
    persona_tier = Column(String(50))
    generated_reply = Column(Text)
    posted_reply = Column(Text, nullable=True)
    posted_at = Column(DateTime, nullable=True)
    model_used = Column(String(100))
    tokens_used = Column(Integer, nullable=True)
    cost_estimate = Column(Float, nullable=True)
    success = Column(Boolean, default=False)


class VIPRegistry(Base):
    __tablename__ = "vip_registry"
    linkedin_urn = Column(String(255), primary_key=True)
    name = Column(String(255))
    company = Column(String(255))
    role = Column(String(255))
    relationship_to_us = Column(String(100))
    voice_note = Column(Text, nullable=True)
    reply_tier = Column(String(50))
    added_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Create all tables
Base.metadata.create_all(bind=engine)
