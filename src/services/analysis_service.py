import os
import uuid
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(
        self,
        upload_dir: str = "uploads",
        output_dir: str = "outputs",
        max_workers: int = 4
    ):
        """Initialize the analysis service"""
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video processor
        self.processor = VideoProcessor()
        
        # Store analysis tasks
        self.tasks: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized AnalysisService")

    async def process_video(
        self,
        video_path: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Process a video file asynchronously
        
        Args:
            video_path: Path to the uploaded video file
            user_id: ID of the user who uploaded the video
            
        Returns:
            Dictionary containing task information
        """
        try:
            # Generate unique analysis ID
            analysis_id = str(uuid.uuid4())
            
            # Create task entry
            task = {
                "id": analysis_id,
                "user_id": user_id,
                "video_path": video_path,
                "status": "processing",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Store task
            self.tasks[analysis_id] = task
            
            # Start processing in background
            asyncio.create_task(self._process_video_task(analysis_id))
            
            return task
            
        except Exception as e:
            logger.error(f"Error starting video processing: {str(e)}")
            raise

    async def _process_video_task(self, analysis_id: str):
        """Background task for processing video"""
        try:
            task = self.tasks[analysis_id]
            video_path = task["video_path"]
            
            # Generate output paths
            output_video = self.output_dir / f"{analysis_id}_processed.mp4"
            output_json = self.output_dir / f"{analysis_id}_results.json"
            
            # Process video in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self.processor.process_video,
                video_path,
                str(output_video)
            )
            
            # Save results
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Update task status
            task.update({
                "status": "completed",
                "output_video": str(output_video),
                "output_json": str(output_json),
                "updated_at": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Completed processing video {analysis_id}")
            
        except Exception as e:
            logger.error(f"Error processing video {analysis_id}: {str(e)}")
            self.tasks[analysis_id].update({
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.utcnow().isoformat()
            })

    def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an analysis task"""
        return self.tasks.get(analysis_id)

    def get_analysis_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a completed analysis"""
        task = self.tasks.get(analysis_id)
        if not task or task["status"] != "completed":
            return None
            
        try:
            with open(task["output_json"], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            return None

    def list_analyses(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List analysis tasks with filtering and pagination"""
        # Filter tasks
        filtered_tasks = self.tasks.values()
        if user_id:
            filtered_tasks = [t for t in filtered_tasks if t["user_id"] == user_id]
        if status:
            filtered_tasks = [t for t in filtered_tasks if t["status"] == status]
            
        # Sort by creation date
        filtered_tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        total = len(filtered_tasks)
        tasks = filtered_tasks[offset:offset + limit]
        
        return {
            "tasks": tasks,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    def __del__(self):
        """Cleanup when the service is destroyed"""
        self.executor.shutdown(wait=True) 