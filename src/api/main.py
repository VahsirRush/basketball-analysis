from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import shutil
import os
from sqlalchemy.orm import Session
from models.db import SessionLocal
from models.video import Video
from models.analysis import Analysis
from fastapi.staticfiles import StaticFiles
from models.video_processor import VideoProcessor
import cv2
import asyncio
from pathlib import Path
import json

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Basketball Play Analysis API",
    description="API for analyzing basketball plays using computer vision and machine learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for video playback/download
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Initialize the video processor once
video_processor = VideoProcessor()

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
        # Update video status
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video {video_id} not found")
            return
        
        video.status = "processing"
        db.commit()

        # Process video
        result = video_processor.process_video(file_path)
        
        # Save analysis results
        analysis = Analysis(video_id=video_id, result_json=result)
        db.add(analysis)
        
        # Update video status
        video.status = "completed"
        db.commit()
        
        logger.info(f"Video {video_id} processing completed")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        # Update video status to error
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = "error"
            video.error_message = str(e)
            db.commit()
        
        # Clean up file
        try:
            os.remove(file_path)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")

def validate_video(file_path: str) -> Dict[str, Any]:
    """Validate video file and extract metadata"""
    try:
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
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }
    except Exception as e:
        raise ValueError(f"Video validation failed: {str(e)}")

@app.post("/upload/video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = next(get_db())
):
    """Upload and process a video file"""
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: MP4, AVI, MOV, MKV"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Validate video
        try:
            video_metadata = validate_video(str(file_path))
        except ValueError as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(e))
            
        # Create DB record
        video = Video(
            filename=file.filename,
            status="pending",
            duration=video_metadata["duration"],
            video_metadata=video_metadata
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        
        # Start async processing
        background_tasks.add_task(
            process_video_async,
            video.id,
            str(file_path),
            SessionLocal()
        )
        
        return {
            "message": "Video uploaded successfully",
            "video_id": video.id,
            "filename": video.filename,
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        # Clean up file if it exists
        if 'file_path' in locals():
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos")
def list_videos(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = next(get_db())
):
    """List uploaded videos with optional filtering"""
    query = db.query(Video)
    
    if status:
        query = query.filter(Video.status == status)
        
    total = query.count()
    videos = query.order_by(Video.upload_time.desc()).offset(offset).limit(limit).all()
    
    result = []
    for v in videos:
        analysis = db.query(Analysis).filter(Analysis.video_id == v.id).first()
        result.append({
            "id": v.id,
            "filename": v.filename,
            "upload_time": v.upload_time,
            "status": v.status,
            "error_message": v.error_message if v.status == "error" else None,
            "analysis_id": analysis.id if analysis else None,
            "duration": v.duration,
            "video_metadata": v.video_metadata
        })
        
    return {
        "videos": result,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/analysis/{video_id}")
def get_analysis(video_id: int, db: Session = next(get_db())):
    """Get analysis results for a video"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    if video.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Video processing not completed. Current status: {video.status}"
        )
        
    analysis = db.query(Analysis).filter(Analysis.video_id == video_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    return analysis.result_json

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_initialized": video_processor.models_initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 