from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class Video(Base):
    __tablename__ = "videos"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False, index=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="pending")  # pending, processing, completed, error
    processed_at = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Float, nullable=True)
    video_metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    analyses = relationship("Analysis", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Video(id={self.id}, filename='{self.filename}', status='{self.status}')>" 