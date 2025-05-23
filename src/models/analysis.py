from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    play_type = Column(String)
    confidence = Column(Float)
    analysis_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time = Column(Float, nullable=True)  # in seconds

    video = relationship("Video", back_populates="analyses")

    def __repr__(self):
        return f"<Analysis(id={self.id}, video_id={self.video_id}, play_type='{self.play_type}')>" 