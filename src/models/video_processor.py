import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import torch
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.structures import ActionDataSample
import os
import time
import hashlib
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the video processor with ML models
        
        Args:
            cache_dir: Directory to store processed video results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.models_initialized = False
        self.initialize_models()

    def initialize_models(self):
        """Initialize all ML models with proper error handling"""
        try:
            logger.info("Initializing ML models...")
            
            # Check CUDA availability
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Initialize YOLOv8 for player detection
            try:
                self.detector = YOLO('yolov8x.pt')
                logger.info("YOLOv8 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {str(e)}")
                raise
            
            # Initialize DeepSORT for player tracking
            try:
                self.tracker = DeepSort(max_age=30)
                logger.info("DeepSORT tracker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DeepSORT: {str(e)}")
                raise
            
            # Initialize MediaPipe for pose estimation
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe Pose initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe: {str(e)}")
                raise
            
            # Initialize MMAction2 for action recognition
            try:
                config_path = 'configs/recognition/tsn/tsn_r50_1x1x8_100e_kinetics400_rgb.py'
                checkpoint_path = 'checkpoints/tsn_r50_1x1x8_100e_kinetics400_rgb_20200618-2692d16c.pth'
                
                if not os.path.exists(config_path):
                    logger.warning(f"MMAction2 config not found at {config_path}")
                    self.action_recognizer = None
                elif not os.path.exists(checkpoint_path):
                    logger.warning(f"MMAction2 checkpoint not found at {checkpoint_path}")
                    self.action_recognizer = None
                else:
                    self.action_recognizer = init_recognizer(
                        config_path,
                        checkpoint_path,
                        device=self.device
                    )
                    logger.info("MMAction2 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MMAction2: {str(e)}")
                self.action_recognizer = None
            
            self.models_initialized = True
            logger.info("All ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            self.models_initialized = False
            raise

    def test_models(self, test_frame: Optional[np.ndarray] = None) -> Dict[str, bool]:
        """
        Test all ML models with a sample frame
        
        Args:
            test_frame: Optional test frame. If None, generates a random frame
            
        Returns:
            Dictionary containing test results for each model
        """
        results = {
            'yolov8': False,
            'deepsort': False,
            'mediapipe': False,
            'mmaction2': False
        }
        
        try:
            if test_frame is None:
                # Generate a random test frame
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Test YOLOv8
            try:
                detections = self.detect_players(test_frame)
                results['yolov8'] = True
                logger.info(f"YOLOv8 test successful. Detected {len(detections)} players")
            except Exception as e:
                logger.error(f"YOLOv8 test failed: {str(e)}")
            
            # Test DeepSORT
            try:
                if results['yolov8']:
                    tracked_players = self.track_players(test_frame, detections)
                    results['deepsort'] = True
                    logger.info(f"DeepSORT test successful. Tracked {len(tracked_players)} players")
            except Exception as e:
                logger.error(f"DeepSORT test failed: {str(e)}")
            
            # Test MediaPipe
            try:
                if results['yolov8']:
                    for detection in detections:
                        pose = self.estimate_pose(test_frame, detection['bbox'])
                        if pose is not None:
                            results['mediapipe'] = True
                            logger.info("MediaPipe test successful")
                            break
            except Exception as e:
                logger.error(f"MediaPipe test failed: {str(e)}")
            
            # Test MMAction2
            try:
                if self.action_recognizer is not None and results['yolov8']:
                    for detection in detections:
                        action = self.recognize_action(test_frame, detection['bbox'])
                        if action != "unknown":
                            results['mmaction2'] = True
                            logger.info(f"MMAction2 test successful. Detected action: {action}")
                            break
            except Exception as e:
                logger.error(f"MMAction2 test failed: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model testing: {str(e)}")
            return results

    def validate_video(self, video_path: str) -> Dict[str, Any]:
        """
        Validate video file and extract basic metadata
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata and validation status
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "valid": False,
                    "error": f"Could not open video file: {video_path}"
                }

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps else 0

            # Validate video properties
            if fps <= 0:
                return {
                    "valid": False,
                    "error": "Invalid FPS value"
                }
            
            if frame_count <= 0:
                return {
                    "valid": False,
                    "error": "Invalid frame count"
                }

            if width <= 0 or height <= 0:
                return {
                    "valid": False,
                    "error": "Invalid video dimensions"
                }

            cap.release()

            return {
                "valid": True,
                "metadata": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "file_size": os.path.getsize(video_path),
                    "format": Path(video_path).suffix[1:].upper()
                }
            }

        except Exception as e:
            logger.error(f"Error validating video: {str(e)}")
            return {
                "valid": False,
                "error": str(e)
            }

    def detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players in a frame using YOLOv8
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected players with bounding boxes and confidence scores
        """
        results = self.detector(frame, classes=[0])  # class 0 is person
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                if conf > 0.5:  # Confidence threshold
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        return detections

    def track_players(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Track players across frames using DeepSORT
        
        Args:
            frame: Input frame
            detections: List of player detections
            
        Returns:
            List of tracked players with IDs and positions
        """
        # Convert detections to DeepSORT format
        bboxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Update tracker
        tracks = self.tracker.update_tracks(bboxes, scores=scores, frame=frame)
        
        # Convert tracks to our format
        tracked_players = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_tlbr()
            
            tracked_players.append({
                'id': track_id,
                'bbox': bbox.tolist(),
                'position': [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]  # Center point
            })
            
        return tracked_players

    def estimate_pose(self, frame: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
        """
        Estimate player pose using MediaPipe
        
        Args:
            frame: Input frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary containing pose keypoints
        """
        x1, y1, x2, y2 = map(int, bbox)
        player_roi = frame[y1:y2, x1:x2]
        
        if player_roi.size == 0:
            return None
            
        results = self.pose.process(cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None
            
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
            
        return {'keypoints': keypoints}

    def recognize_action(self, frame: np.ndarray, bbox: List[float]) -> str:
        """
        Recognize player action using MMAction2
        
        Args:
            frame: Input frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            Recognized action label
        """
        x1, y1, x2, y2 = map(int, bbox)
        player_roi = frame[y1:y2, x1:x2]
        
        if player_roi.size == 0:
            return "unknown"
            
        # Prepare input for MMAction2
        data = {
            'imgs': player_roi,
            'label': -1,
            'modality': 'RGB',
            'num_clips': 1,
            'clip_len': 1
        }
        
        result = inference_recognizer(self.action_recognizer, data)
        return result.pred_label.item()

    def get_video_hash(self, video_path: str) -> str:
        """
        Generate a unique hash for the video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            MD5 hash of the video file
        """
        hash_md5 = hashlib.md5()
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_cache_path(self, video_path: str) -> Path:
        """
        Get the cache file path for a video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the cache file
        """
        video_hash = self.get_video_hash(video_path)
        return self.cache_dir / f"{video_hash}.pkl"

    def load_from_cache(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Load processed results from cache
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Cached results if available, None otherwise
        """
        cache_path = self.get_cache_path(video_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded results from cache: {cache_path}")
                return cached_data
            except Exception as e:
                logger.error(f"Error loading from cache: {str(e)}")
        return None

    def save_to_cache(self, video_path: str, results: Dict[str, Any]):
        """
        Save processed results to cache
        
        Args:
            video_path: Path to the video file
            results: Processing results to cache
        """
        cache_path = self.get_cache_path(video_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved results to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")

    def analyze_play_type(self, play: Dict[str, Any]) -> str:
        """
        Analyze and categorize a play based on player movements and actions
        
        Args:
            play: Play data to analyze
            
        Returns:
            Categorized play type
        """
        if not play.get('players'):
            return "unknown"

        # Extract player movements
        movements = []
        for player in play['players']:
            if player.get('movements'):
                movements.extend(player['movements'])

        if not movements:
            return "unknown"

        # Calculate average movement speed
        speeds = []
        for i in range(1, len(movements)):
            prev_pos = np.array(movements[i-1]['position'])
            curr_pos = np.array(movements[i]['position'])
            speed = np.linalg.norm(curr_pos - prev_pos)
            speeds.append(speed)

        avg_speed = np.mean(speeds) if speeds else 0

        # Count actions
        action_counts = {}
        for movement in movements:
            action = movement.get('action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1

        # Determine play type based on characteristics
        if avg_speed > 100:  # High movement speed
            return "fast_break"
        elif action_counts.get('shooting', 0) > 0:
            return "shooting_play"
        elif action_counts.get('passing', 0) > 2:
            return "passing_play"
        elif avg_speed < 20:  # Low movement speed
            return "set_play"
        else:
            return "half_court"

    def get_play_breakdown(self, plays: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a breakdown of plays by type
        
        Args:
            plays: List of plays to analyze
            
        Returns:
            Dictionary containing play type statistics
        """
        play_types = {}
        total_duration = 0
        
        for play in plays:
            play_type = self.analyze_play_type(play)
            duration = play['end_time'] - play['start_time']
            
            if play_type not in play_types:
                play_types[play_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'plays': []
                }
            
            play_types[play_type]['count'] += 1
            play_types[play_type]['total_duration'] += duration
            play_types[play_type]['plays'].append(play)
            total_duration += duration

        # Calculate percentages and sort by count
        for play_type in play_types:
            play_types[play_type]['percentage'] = (
                play_types[play_type]['count'] / len(plays) * 100
            )
            play_types[play_type]['avg_duration'] = (
                play_types[play_type]['total_duration'] / play_types[play_type]['count']
            )

        return {
            'play_types': play_types,
            'total_plays': len(plays),
            'total_duration': total_duration,
            'timestamp': datetime.now().isoformat()
        }

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process a basketball video to detect and track players
        
        Args:
            video_path: Path to the input video file
            output_path: Optional path to save the processed video
            confidence_threshold: Minimum confidence score for detections
            force_reprocess: Force reprocessing even if cached results exist
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.models_initialized:
            raise RuntimeError("ML models not initialized. Call initialize_models() first.")

        try:
            # Check cache first
            if not force_reprocess:
                cached_results = self.load_from_cache(video_path)
                if cached_results:
                    logger.info("Using cached results")
                    return cached_results

            # First validate the video
            validation = self.validate_video(video_path)
            if not validation["valid"]:
                raise ValueError(validation["error"])

            video_info = validation["metadata"]
            
            # Test models with first frame
            cap = cv2.VideoCapture(video_path)
            ret, test_frame = cap.read()
            if ret:
                test_results = self.test_models(test_frame)
                logger.info(f"Model test results: {test_results}")
                if not all(test_results.values()):
                    logger.warning("Some models failed testing. Results may be incomplete.")
            cap.release()
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Initialize play detection variables
            plays = []
            current_play = None
            frame_idx = 0
            last_play_end = 0
            min_play_duration = int(video_info["fps"] * 3)  # Minimum 3 seconds per play
            max_play_duration = int(video_info["fps"] * 24)  # Maximum 24 seconds per play
            play_boundary_threshold = int(video_info["fps"] * 1)  # 1 second of inactivity to detect play boundary

            # Track game state
            score = {"home": 0, "away": 0}
            possession = "home"
            last_score_change = 0
            player_tracks = {}  # Track player IDs across frames

            logger.info(f"Starting video processing: {video_info}")
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect players in current frame
                detections = self.detect_players(frame)
                
                # Track players
                tracked_players = self.track_players(frame, detections)
                
                # Update player tracks
                for player in tracked_players:
                    player_id = player['id']
                    if player_id not in player_tracks:
                        player_tracks[player_id] = {
                            'id': player_id,
                            'role': 'player',  # Could be determined by position/behavior
                            'movements': []
                        }
                    
                    # Get pose and action
                    pose = self.estimate_pose(frame, player['bbox'])
                    action = self.recognize_action(frame, player['bbox'])
                    
                    # Add movement
                    player_tracks[player_id]['movements'].append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / video_info["fps"],
                        'position': player['position'],
                        'action': action,
                        'pose': pose if pose else {'keypoints': []}
                    })

                # Detect play boundaries based on player movements
                is_play_boundary = False
                if current_play:
                    # Check if players have moved significantly
                    player_movement = False
                    for player in tracked_players:
                        if len(player_tracks[player['id']]['movements']) > 1:
                            last_pos = player_tracks[player['id']]['movements'][-2]['position']
                            current_pos = player_tracks[player['id']]['movements'][-1]['position']
                            movement = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
                            if movement > 50:  # Threshold for significant movement
                                player_movement = True
                                break
                    
                    # End play if no significant movement for threshold duration
                    if not player_movement and (frame_idx - last_play_end) >= play_boundary_threshold:
                        is_play_boundary = True

                if is_play_boundary and current_play:
                    # End current play
                    current_play["end_frame"] = frame_idx
                    current_play["end_time"] = frame_idx / video_info["fps"]
                    current_play["players"] = list(player_tracks.values())
                    plays.append(current_play)
                    last_play_end = frame_idx
                    current_play = None
                    player_tracks = {}  # Reset tracks for new play

                # Start new play if needed
                if not current_play and (frame_idx - last_play_end) >= min_play_duration:
                    current_play = {
                        "id": len(plays) + 1,
                        "start_frame": frame_idx,
                        "start_time": frame_idx / video_info["fps"],
                        "type": "play",  # Will be determined by analyze_play_type
                        "team_possession": possession,
                        "score": score.copy()
                    }

                frame_idx += 1
                if frame_idx % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_idx / elapsed_time
                    logger.info(f"Processed {frame_idx}/{video_info['frame_count']} frames ({fps:.2f} fps)")

            # Add the last play if exists
            if current_play:
                current_play["end_frame"] = frame_idx
                current_play["end_time"] = frame_idx / video_info["fps"]
                current_play["players"] = list(player_tracks.values())
                plays.append(current_play)

            # Clean up
            cap.release()

            # Analyze play types
            play_breakdown = self.get_play_breakdown(plays)
            
            # Update play types
            for play in plays:
                play['type'] = self.analyze_play_type(play)

            # Log processing summary
            total_time = time.time() - start_time
            logger.info(f"Processing completed in {total_time:.2f} seconds")
            logger.info(f"Detected {len(plays)} plays")
            logger.info(f"Average processing speed: {frame_idx/total_time:.2f} fps")
            
            # Print play type breakdown
            logger.info("\nPlay Type Breakdown:")
            for play_type, stats in play_breakdown['play_types'].items():
                logger.info(f"{play_type}: {stats['count']} plays ({stats['percentage']:.1f}%)")

            results = {
                "plays": plays,
                "video_info": video_info,
                "processing_stats": {
                    "total_time": total_time,
                    "total_frames": frame_idx,
                    "average_fps": frame_idx/total_time,
                    "play_count": len(plays)
                },
                "play_breakdown": play_breakdown
            }

            # Cache the results
            self.save_to_cache(video_path, results)

            return results

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'pose'):
            self.pose.close() 