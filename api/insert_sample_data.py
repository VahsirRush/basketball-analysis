from datetime import datetime
from sqlalchemy.orm import Session
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .database import SessionLocal, engine
from .models import Base, Game, Play, PlayerBiometrics

# Create tables
Base.metadata.create_all(bind=engine)

# Sample data
sample_games = [
    {
        "home_team": "Lakers",
        "away_team": "Celtics",
        "game_date": datetime(2024, 2, 15),
        "season": "2023-24",
        "game_type": "Regular Season",
        "game_number": 1,
        "venue": "Crypto.com Arena",
        "home_score": 120,
        "away_score": 115
    },
    {
        "home_team": "Warriors",
        "away_team": "Bucks",
        "game_date": datetime(2024, 2, 16),
        "season": "2023-24",
        "game_type": "Regular Season",
        "game_number": 2,
        "venue": "Chase Center",
        "home_score": 118,
        "away_score": 112
    }
]

sample_plays = [
    {
        "game_id": 1,
        "play_type": "pick_and_roll",
        "player_role": "ball_handler",
        "defensive_scheme": "switch",
        "success": True
    },
    {
        "game_id": 1,
        "play_type": "isolation",
        "player_role": "ball_handler",
        "defensive_scheme": "man_to_man",
        "success": False
    },
    {
        "game_id": 2,
        "play_type": "post_up",
        "player_role": "post_player",
        "defensive_scheme": "double_team",
        "success": True
    }
]

sample_biometrics = [
    {
        "player_id": 1,
        "game_id": 1,
        "height": 6.6,  # in feet
        "weight": 220,  # in pounds
        "age": 25
    },
    {
        "player_id": 2,
        "game_id": 1,
        "height": 6.8,
        "weight": 240,
        "age": 28
    },
    {
        "player_id": 3,
        "game_id": 2,
        "height": 6.4,
        "weight": 210,
        "age": 23
    }
]

def insert_sample_data():
    db = SessionLocal()
    try:
        # Insert games
        for game_data in sample_games:
            game = Game(**game_data)
            db.add(game)
        db.commit()

        # Insert plays
        for play_data in sample_plays:
            play = Play(**play_data)
            db.add(play)
        db.commit()

        # Insert biometrics
        for bio_data in sample_biometrics:
            bio = PlayerBiometrics(**bio_data)
            db.add(bio)
        db.commit()

        print("Sample data inserted successfully!")
    except Exception as e:
        print(f"Error inserting sample data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    insert_sample_data() 