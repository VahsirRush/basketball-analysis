from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class Analysis(Base):
    __tablename__ = "plays"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    play_type = Column(String)
    confidence = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="analyses") 