from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/basketball_analysis")

# Create engine with connection pooling
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending, processing, completed, error
    error_message = Column(Text, nullable=True)
    duration = Column(Float, nullable=True)
    video_metadata = Column(JSON, nullable=True)
    analysis = relationship("Analysis", back_populates="video", uselist=False)

    def __repr__(self):
        return f"<Video(id={self.id}, filename='{self.filename}', status='{self.status}')>"

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    result_json = Column(JSON)
    processing_time = Column(Float, nullable=True)  # in seconds
    video = relationship("Video", back_populates="analysis")

    def __repr__(self):
        return f"<Analysis(id={self.id}, video_id={self.video_id})>"

def init_db():
    """Initialize the database, creating all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

def get_db():
    """Get database session with error handling"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

# Initialize database on import
init_db() 