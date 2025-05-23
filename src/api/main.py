from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import shutil
import os
from sqlalchemy.orm import Session
from models.db import SessionLocal, init_db
from models.video import Video, VideoStatus
from models.analysis import Analysis
from fastapi.staticfiles import StaticFiles
from models.video_processor import VideoProcessor
import cv2
import asyncio
from pathlib import Path
import json
import psutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Basketball Play Analysis API",
    description="API for analyzing basketball plays using computer vision and machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Use absolute path for uploads
UPLOAD_DIR = Path(os.path.abspath("uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files for video playback/download
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Initialize the video processor once
video_processor = VideoProcessor()

# Store processing status
processing_status = {}

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

async def process_video_async(video_id: int, file_path: str, db: Session):
    """Process video asynchronously and update database"""
    try:
        logger.info(f"Starting video processing for video_id: {video_id}")
        
        # Update video status
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video {video_id} not found")
            return
        
        video.status = VideoStatus.PROCESSING
        db.commit()
        logger.info(f"Updated video {video_id} status to processing")

        # Process video
        try:
            result = video_processor.process_video(file_path)
            logger.info(f"Video processing completed for video_id: {video_id}")
        except Exception as e:
            logger.error(f"Error in video processing for video_id {video_id}: {str(e)}")
            raise
        
        # Save analysis results
        try:
            analysis = Analysis(
                video_id=video_id,
                start_frame=result.get("start_frame", 0),
                end_frame=result.get("end_frame", 0),
                play_type=result.get("play_type"),
                confidence=result.get("confidence"),
                analysis_metadata=result.get("metadata", {})
            )
            db.add(analysis)
            db.commit()
            logger.info(f"Analysis saved for video_id: {video_id}")
        except Exception as e:
            logger.error(f"Error saving analysis for video_id {video_id}: {str(e)}")
            raise
        
        # Update video status
        video.status = VideoStatus.COMPLETED
        video.processed_at = datetime.utcnow()
        db.commit()
        logger.info(f"Video {video_id} processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        # Update video status to error
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = VideoStatus.ERROR
                video.error_message = str(e)
                db.commit()
                logger.info(f"Updated video {video_id} status to error")
        except Exception as db_error:
            logger.error(f"Error updating video status to error: {str(db_error)}")
        
        # Clean up file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")

def validate_video(file_path: str) -> Dict[str, Any]:
    """Validate video file and extract metadata"""
    try:
        logger.info(f"Validating video file: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else None
        
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid video properties")
            
        cap.release()
        
        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }
        logger.info(f"Video validation successful: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Video validation failed: {str(e)}")
        raise ValueError(f"Video validation failed: {str(e)}")

@app.post("/upload/video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a video file"""
    file_path = None
    try:
        logger.info(f"Received video upload request for file: {file.filename}")
        
        # Validate file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.error(f"Invalid file format: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: MP4, AVI, MOV, MKV"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error saving file")
            
        # Validate video
        try:
            video_metadata = validate_video(str(file_path))
        except ValueError as e:
            logger.error(f"Video validation failed: {str(e)}")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(e))
            
        # Create video record
        video = Video(
            filename=file.filename,
            status=VideoStatus.PENDING,
            video_metadata=video_metadata
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        
        # Start background processing
        background_tasks.add_task(process_video_async, video.id, str(file_path), db)
        
        return {
            "id": video.id,
            "filename": video.filename,
            "status": video.status,
            "message": "Video upload successful. Processing started."
        }
    except Exception as e:
        # Clean up file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Error in upload_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos")
def list_videos(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List all videos with optional status filter"""
    try:
        query = db.query(Video)
        if status:
            query = query.filter(Video.status == status)
        videos = query.order_by(Video.upload_date.desc()).offset(offset).limit(limit).all()
        return videos
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving videos")

@app.get("/videos/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    """Get video details by ID"""
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving video")

@app.get("/analysis/{video_id}")
def get_analysis(video_id: int, db: Session = Depends(get_db)):
    """Get analysis results for a video"""
    try:
        analysis = db.query(Analysis).filter(Analysis.video_id == video_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Return structured response
        return {
            "id": analysis.id,
            "video_id": analysis.video_id,
            "play_type": analysis.play_type,
            "confidence": analysis.confidence,
            "start_frame": analysis.start_frame,
            "end_frame": analysis.end_frame,
            "processing_time": analysis.processing_time,
            "analysis_metadata": analysis.analysis_metadata,
            "created_at": analysis.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving analysis")

@app.get("/health")
async def health_check():
    """Health check endpoint with system information"""
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "disk_usage": f"{disk.percent}%",
                "memory_available": f"{memory.available / (1024 * 1024 * 1024):.2f} GB",
                "disk_free": f"{disk.free / (1024 * 1024 * 1024):.2f} GB"
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/progress/{video_id}")
async def get_progress(video_id: int):
    """Get processing progress for a video"""
    try:
        if video_id not in processing_status:
            raise HTTPException(status_code=404, detail="No processing status found for this video")
        return processing_status[video_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving progress")

@app.on_event("startup")
async def startup_event():
    init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 