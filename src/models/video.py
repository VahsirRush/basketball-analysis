from sqlalchemy import Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="pending")
    processed_at = Column(DateTime(timezone=True))
    duration = Column(Float)
    video_metadata = Column(JSON)
    error_message = Column(String)

    analyses = relationship("Analysis", back_populates="video") 