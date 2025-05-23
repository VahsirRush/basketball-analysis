from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mock analysis results
MOCK_RESULTS = {
    "plays": [
        {
            "id": 1,
            "type": "Pick & Roll",
            "duration": 24,
            "playerCount": 10,
            "thumbnail": "assets/court-diagram.svg",
            "confidence": 0.98,
            "players": [
                {"number": 23, "position": {"x": 30, "y": 30}, "team": "offense", "hasBall": True},
                {"number": 7, "position": {"x": 40, "y": 40}, "team": "offense", "hasBall": False},
                {"number": 3, "position": {"x": 50, "y": 50}, "team": "defense", "hasBall": False}
            ],
            "movements": [
                {
                    "playerNumber": 23,
                    "type": "cut",
                    "start": {"x": 30, "y": 30},
                    "end": {"x": 40, "y": 40},
                    "confidence": 0.95
                },
                {
                    "playerNumber": 7,
                    "type": "screen",
                    "start": {"x": 40, "y": 40},
                    "end": {"x": 45, "y": 45},
                    "confidence": 0.92
                }
            ]
        }
    ]
}

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    logger.debug("Received analyze request")
    try:
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            logger.error("Empty filename")
            return jsonify({"error": "No video file selected"}), 400
        
        logger.debug(f"Processing video file: {video_file.filename}")
        
        # Save the video file
        video_path = os.path.join(UPLOAD_DIR, video_file.filename)
        video_file.save(video_path)
        logger.debug(f"Video saved to: {video_path}")
        
        # Simulate processing time
        time.sleep(2)
        
        # For now, return mock results
        logger.debug("Returning mock results")
        return jsonify(MOCK_RESULTS)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 