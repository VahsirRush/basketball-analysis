import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def detect_players(frame, model):
    try:
        # Get detections from model
        results = model(frame)
        
        # Initialize list to store player boxes
        player_boxes = []
        
        # Process results
        if hasattr(results, 'boxes'):
            # YOLOv8 format
            boxes = results.boxes
            if len(boxes) > 0:
                # Get xyxy coordinates for each detection
                for box in boxes:
                    try:
                        # Convert box coordinates to numpy array
                        bbox = box.xyxy[0].cpu().numpy()
                        # Only include detections of class 'person' (class 0 in COCO dataset)
                        if box.cls[0].item() == 0:  # 0 is the class ID for 'person' in COCO
                            player_boxes.append(bbox)
                    except Exception as e:
                        logger.warning(f"Error processing box: {str(e)}")
                        continue
        
        return player_boxes
        
    except Exception as e:
        logger.error(f"Error in detect_players: {str(e)}")
        return []

# Example usage:
# player_boxes = detect_players(frame, model) 