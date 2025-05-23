from fastapi import FastAPI, Depends, Query, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func, Integer, asc, desc
from .database import get_db, init_db, engine
from .models import Base, Game, Play, PlayerBiometrics, PlayType, PlayCategory
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
from . import models, schemas
import os
import shutil
from pathlib import Path
import requests
import json
from play_scheme_processor import process_game_video
from nba_api.stats.endpoints import (
    scoreboard,
    boxscoretraditionalv2,
    playbyplayv2,
    leaguegamefinder,
    commonplayerinfo
)
from nba_api.stats.static import teams, players
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

app = FastAPI(title="NBA Play Analysis API")

# Configuration
UPLOAD_DIR = "game_vods"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# NBA API Configuration
NBA_API_BASE = "https://api.nba.com/stats/v1"
NBA_API_KEY = os.getenv("NBA_API_KEY")
NBA_API_HEADERS = {
    'Authorization': f'Bearer {NBA_API_KEY}',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

# Second Spectrum Configuration (if using)
SECOND_SPECTRUM_API_KEY = os.getenv("SECOND_SPECTRUM_API_KEY")
SECOND_SPECTRUM_API_BASE = "https://api.secondspectrum.com/v1"

class GameVOD(BaseModel):
    game_id: int
    video_path: str
    upload_date: datetime
    duration: float
    resolution: str
    file_size: int
    status: str  # "uploading", "processing", "ready", "error"

@app.on_event("startup")
async def startup_event():
    init_db()

# Pydantic models for request/response validation
class GameBase(BaseModel):
    home_team: str
    away_team: str
    date: datetime
    season: Optional[str] = None
    game_type: Optional[str] = None
    game_number: Optional[int] = None
    venue: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None

class GameCreate(GameBase):
    pass

class GameResponse(GameBase):
    id: int

    class Config:
        from_attributes = True

class PlayBase(BaseModel):
    game_id: int
    play_type: str
    play_variation: str
    play_name: str
    play_code: str
    play_description: str
    play_intent: str
    expected_outcome: str
    player_role: str
    defensive_scheme: str
    success: bool
    success_percentage: float
    avg_points_per_attempt: float
    shot_priority: str
    lebron_score: float
    pressure_adjusted_shooting: float
    score_differential: int
    time_remaining: int
    possession: int
    offensive_player_id: int
    defensive_player_id: int
    play_efficiency: float
    defensive_rating: float
    play_sequence: int
    counter_adjustment: str

class PlayCreate(PlayBase):
    pass

class PlayResponse(PlayBase):
    id: int

    class Config:
        from_attributes = True

class PlayerBiometricsBase(BaseModel):
    player_id: int
    game_id: int
    play_id: Optional[int] = None
    height: float
    weight: float
    age: int
    heart_rate: int
    sweat_rate: float
    fatigue_level: float
    movement_speed: float
    jump_height: float
    reaction_time: float

class PlayerBiometricsCreate(PlayerBiometricsBase):
    pass

class PlayerBiometricsResponse(PlayerBiometricsBase):
    id: int

    class Config:
        from_attributes = True

@app.post("/games", response_model=GameResponse)
def create_game(game: GameCreate, db: Session = Depends(get_db)):
    db_game = Game(**game.dict())
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return db_game

@app.post("/plays", response_model=PlayResponse)
def create_play(play: PlayCreate, db: Session = Depends(get_db)):
    # Verify game exists
    game = db.query(Game).filter(Game.id == play.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    db_play = Play(**play.dict())
    db.add(db_play)
    db.commit()
    db.refresh(db_play)
    return db_play

@app.post("/player-biometrics", response_model=PlayerBiometricsResponse)
def create_player_biometrics(bio: PlayerBiometricsCreate, db: Session = Depends(get_db)):
    # Verify game exists
    game = db.query(Game).filter(Game.id == bio.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    db_bio = PlayerBiometrics(**bio.dict())
    db.add(db_bio)
    db.commit()
    db.refresh(db_bio)
    return db_bio

@app.get("/debug/db-path")
def debug_db_path():
    return {"db_path": str(engine.url)}

@app.get("/games/recent")
def get_recent_games(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    games = db.query(Game).order_by(Game.date.desc()).limit(limit).all()
    return [
        {
            "game_id": g.id,
            "date": g.date.strftime("%Y-%m-%d") if g.date else None,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "season": g.season,
            "game_type": g.game_type
        }
        for g in games
    ]

@app.get("/games")
def get_games(db: Session = Depends(get_db)):
    return db.query(Game).all()

@app.get("/player-biometrics/{game_id}")
def get_player_biometrics(game_id: int, db: Session = Depends(get_db)):
    biometrics = db.query(PlayerBiometrics).filter(PlayerBiometrics.game_id == game_id).all()
    return biometrics

@app.get("/plays/analysis")
def analyze_plays(db: Session = Depends(get_db)):
    # Get play type distribution
    play_types = db.query(
        Play.play_type,
        func.count(Play.id).label('count'),
        func.avg(Play.success.cast(Integer)).label('success_rate')
    ).group_by(Play.play_type).all()
    
    # Get defensive scheme effectiveness
    defensive_schemes = db.query(
        Play.defensive_scheme,
        func.count(Play.id).label('count'),
        func.avg(Play.success.cast(Integer)).label('success_rate')
    ).group_by(Play.defensive_scheme).all()
    
    return {
        "play_type_analysis": [
            {
                "play_type": pt.play_type,
                "count": pt.count,
                "success_rate": float(pt.success_rate) if pt.success_rate is not None else 0.0
            }
            for pt in play_types
        ],
        "defensive_scheme_analysis": [
            {
                "scheme": ds.defensive_scheme,
                "count": ds.count,
                "success_rate": float(ds.success_rate) if ds.success_rate is not None else 0.0
            }
            for ds in defensive_schemes
        ]
    }

@app.get("/games/{game_id}/plays")
def get_game_plays(game_id: int, db: Session = Depends(get_db)):
    plays = db.query(Play).filter(Play.game_id == game_id).all()
    return plays

@app.get("/players/stats")
def get_player_stats(db: Session = Depends(get_db)):
    # Get average biometrics by player
    player_stats = db.query(
        PlayerBiometrics.player_id,
        func.avg(PlayerBiometrics.height).label('avg_height'),
        func.avg(PlayerBiometrics.weight).label('avg_weight'),
        func.avg(PlayerBiometrics.age).label('avg_age')
    ).group_by(PlayerBiometrics.player_id).all()
    
    return [
        {
            "player_id": ps.player_id,
            "average_height": float(ps.avg_height),
            "average_weight": float(ps.avg_weight),
            "average_age": float(ps.avg_age)
        }
        for ps in player_stats
    ]

@app.get("/plays/advanced-search")
def advanced_search_plays(
    play_type: Optional[str] = None,
    player_role: Optional[str] = None,
    defensive_scheme: Optional[str] = None,
    min_success_percentage: Optional[float] = None,
    max_success_percentage: Optional[float] = None,
    min_lebron_score: Optional[float] = None,
    max_lebron_score: Optional[float] = None,
    sort_by: Optional[str] = Query(None, description="Field to sort by (e.g., lebron_score, shot_priority, success_percentage)"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Play)
    if play_type:
        query = query.filter(Play.play_type == play_type)
    if player_role:
        query = query.filter(Play.player_role == player_role)
    if defensive_scheme:
        query = query.filter(Play.defensive_scheme == defensive_scheme)
    if min_success_percentage is not None:
        query = query.filter(Play.success_percentage >= min_success_percentage)
    if max_success_percentage is not None:
        query = query.filter(Play.success_percentage <= max_success_percentage)
    if min_lebron_score is not None:
        query = query.filter(Play.lebron_score >= min_lebron_score)
    if max_lebron_score is not None:
        query = query.filter(Play.lebron_score <= max_lebron_score)
    # Sorting
    if sort_by and hasattr(Play, sort_by):
        sort_column = getattr(Play, sort_by)
        if sort_order == "asc":
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))
    # Pagination
    query = query.offset(offset).limit(limit)
    plays = query.all()
    # Return all fields for each play
    return [
        {
            "id": p.id,
            "game_id": p.game_id,
            "play_type": p.play_type,
            "player_role": p.player_role,
            "defensive_scheme": p.defensive_scheme,
            "success": p.success,
            "success_percentage": p.success_percentage,
            "avg_points_per_attempt": p.avg_points_per_attempt,
            "shot_priority": p.shot_priority,
            "lebron_score": p.lebron_score,
            "pressure_adjusted_shooting": p.pressure_adjusted_shooting
        }
        for p in plays
    ]

@app.get("/plays/coach-analysis")
def coach_analysis_plays(
    score_differential: Optional[int] = None,
    time_remaining: Optional[int] = None,
    possession: Optional[str] = None,
    offensive_player_id: Optional[int] = None,
    defensive_player_id: Optional[int] = None,
    min_play_efficiency: Optional[float] = None,
    max_play_efficiency: Optional[float] = None,
    min_defensive_rating: Optional[float] = None,
    max_defensive_rating: Optional[float] = None,
    play_sequence: Optional[str] = None,
    counter_adjustment: Optional[str] = None,
    sort_by: Optional[str] = Query(None, description="Field to sort by (e.g., play_efficiency, defensive_rating)"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Play)
    if score_differential is not None:
        query = query.filter(Play.score_differential == score_differential)
    if time_remaining is not None:
        query = query.filter(Play.time_remaining == time_remaining)
    if possession:
        query = query.filter(Play.possession == possession)
    if offensive_player_id is not None:
        query = query.filter(Play.offensive_player_id == offensive_player_id)
    if defensive_player_id is not None:
        query = query.filter(Play.defensive_player_id == defensive_player_id)
    if min_play_efficiency is not None:
        query = query.filter(Play.play_efficiency >= min_play_efficiency)
    if max_play_efficiency is not None:
        query = query.filter(Play.play_efficiency <= max_play_efficiency)
    if min_defensive_rating is not None:
        query = query.filter(Play.defensive_rating >= min_defensive_rating)
    if max_defensive_rating is not None:
        query = query.filter(Play.defensive_rating <= max_defensive_rating)
    if play_sequence:
        query = query.filter(Play.play_sequence == play_sequence)
    if counter_adjustment:
        query = query.filter(Play.counter_adjustment == counter_adjustment)
    # Sorting
    if sort_by and hasattr(Play, sort_by):
        sort_column = getattr(Play, sort_by)
        if sort_order == "asc":
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))
    # Pagination
    query = query.offset(offset).limit(limit)
    plays = query.all()
    # Return all fields for each play
    return [
        {
            "id": p.id,
            "game_id": p.game_id,
            "play_type": p.play_type,
            "player_role": p.player_role,
            "defensive_scheme": p.defensive_scheme,
            "success": p.success,
            "success_percentage": p.success_percentage,
            "avg_points_per_attempt": p.avg_points_per_attempt,
            "shot_priority": p.shot_priority,
            "lebron_score": p.lebron_score,
            "pressure_adjusted_shooting": p.pressure_adjusted_shooting,
            "score_differential": p.score_differential,
            "time_remaining": p.time_remaining,
            "possession": p.possession,
            "offensive_player_id": p.offensive_player_id,
            "defensive_player_id": p.defensive_player_id,
            "play_efficiency": p.play_efficiency,
            "defensive_rating": p.defensive_rating,
            "play_sequence": p.play_sequence,
            "counter_adjustment": p.counter_adjustment
        }
        for p in plays
    ]

@app.get("/plays/variations")
def play_variations(
    play_type: Optional[str] = None,
    play_variation: Optional[str] = None,
    play_name: Optional[str] = None,
    play_code: Optional[str] = None,
    play_intent: Optional[str] = None,
    expected_outcome: Optional[str] = None,
    sort_by: Optional[str] = Query(None, description="Field to sort by (e.g., play_efficiency, success_percentage)"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Play)
    if play_type:
        query = query.filter(Play.play_type == play_type)
    if play_variation:
        query = query.filter(Play.play_variation == play_variation)
    if play_name:
        query = query.filter(Play.play_name == play_name)
    if play_code:
        query = query.filter(Play.play_code == play_code)
    if play_intent:
        query = query.filter(Play.play_intent == play_intent)
    if expected_outcome:
        query = query.filter(Play.expected_outcome == expected_outcome)
    # Sorting
    if sort_by and hasattr(Play, sort_by):
        sort_column = getattr(Play, sort_by)
        if sort_order == "asc":
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))
    # Pagination
    query = query.offset(offset).limit(limit)
    plays = query.all()
    # Return all fields for each play
    return [
        {
            "id": p.id,
            "game_id": p.game_id,
            "play_type": p.play_type,
            "play_variation": p.play_variation,
            "play_name": p.play_name,
            "play_code": p.play_code,
            "play_description": p.play_description,
            "play_intent": p.play_intent,
            "expected_outcome": p.expected_outcome,
            "player_role": p.player_role,
            "defensive_scheme": p.defensive_scheme,
            "success": p.success,
            "success_percentage": p.success_percentage,
            "avg_points_per_attempt": p.avg_points_per_attempt,
            "shot_priority": p.shot_priority,
            "lebron_score": p.lebron_score,
            "pressure_adjusted_shooting": p.pressure_adjusted_shooting,
            "score_differential": p.score_differential,
            "time_remaining": p.time_remaining,
            "possession": p.possession,
            "offensive_player_id": p.offensive_player_id,
            "defensive_player_id": p.defensive_player_id,
            "play_efficiency": p.play_efficiency,
            "defensive_rating": p.defensive_rating,
            "play_sequence": p.play_sequence,
            "counter_adjustment": p.counter_adjustment
        }
        for p in plays
    ]

# Play Collection Endpoints
@app.post("/play-collections/", response_model=schemas.PlayCollection)
def create_play_collection(collection: schemas.PlayCollectionCreate, db: Session = Depends(get_db)):
    db_collection = models.PlayCollection(
        name=collection.name,
        description=collection.description,
        play_type=collection.play_type,
        play_category=collection.play_category,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    db.add(db_collection)
    db.commit()
    db.refresh(db_collection)
    return db_collection

@app.get("/play-collections/", response_model=List[schemas.PlayCollection])
def get_play_collections(
    play_type: Optional[PlayType] = None,
    play_category: Optional[PlayCategory] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.PlayCollection)
    if play_type:
        query = query.filter(models.PlayCollection.play_type == play_type)
    if play_category:
        query = query.filter(models.PlayCollection.play_category == play_category)
    return query.all()

@app.get("/play-collections/{collection_id}", response_model=schemas.PlayCollectionDetail)
def get_play_collection(collection_id: int, db: Session = Depends(get_db)):
    collection = db.query(models.PlayCollection).filter(models.PlayCollection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Play collection not found")
    return collection

@app.post("/play-collections/{collection_id}/plays/{play_id}")
def add_play_to_collection(
    collection_id: int,
    play_id: int,
    order: Optional[int] = None,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    collection = db.query(models.PlayCollection).filter(models.PlayCollection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Play collection not found")
    
    play = db.query(models.Play).filter(models.Play.id == play_id).first()
    if not play:
        raise HTTPException(status_code=404, detail="Play not found")
    
    # Get the next order number if not specified
    if order is None:
        last_item = db.query(models.PlayCollectionItem)\
            .filter(models.PlayCollectionItem.collection_id == collection_id)\
            .order_by(models.PlayCollectionItem.order.desc())\
            .first()
        order = (last_item.order + 1) if last_item else 1
    
    collection_item = models.PlayCollectionItem(
        collection_id=collection_id,
        play_id=play_id,
        order=order,
        notes=notes,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    db.add(collection_item)
    db.commit()
    return {"message": "Play added to collection successfully"}

# Organized Play Access Endpoints
@app.get("/plays/organized/", response_model=List[schemas.Play])
def get_organized_plays(
    play_type: Optional[PlayType] = None,
    play_category: Optional[PlayCategory] = None,
    min_success_rate: Optional[float] = None,
    min_lebron_score: Optional[float] = None,
    sort_by: str = "success_percentage",
    sort_order: str = "desc",
    db: Session = Depends(get_db)
):
    query = db.query(models.Play)
    
    # Apply filters
    if play_type:
        query = query.filter(models.Play.play_type == play_type)
    if play_category:
        query = query.filter(models.Play.play_category == play_category)
    if min_success_rate:
        query = query.filter(models.Play.success_percentage >= min_success_rate)
    if min_lebron_score:
        query = query.filter(models.Play.lebron_score >= min_lebron_score)
    
    # Apply sorting
    if sort_by == "success_percentage":
        query = query.order_by(models.Play.success_percentage.desc() if sort_order == "desc" else models.Play.success_percentage.asc())
    elif sort_by == "lebron_score":
        query = query.order_by(models.Play.lebron_score.desc() if sort_order == "desc" else models.Play.lebron_score.asc())
    elif sort_by == "play_efficiency":
        query = query.order_by(models.Play.play_efficiency.desc() if sort_order == "desc" else models.Play.play_efficiency.asc())
    
    return query.all()

@app.get("/plays/statistics/", response_model=schemas.PlayStatistics)
def get_play_statistics(
    play_type: Optional[PlayType] = None,
    play_category: Optional[PlayCategory] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Play)
    
    if play_type:
        query = query.filter(models.Play.play_type == play_type)
    if play_category:
        query = query.filter(models.Play.play_category == play_category)
    
    plays = query.all()
    
    if not plays:
        raise HTTPException(status_code=404, detail="No plays found matching the criteria")
    
    total_plays = len(plays)
    successful_plays = sum(1 for play in plays if play.success)
    avg_success_rate = sum(play.success_percentage for play in plays) / total_plays
    avg_lebron_score = sum(play.lebron_score for play in plays) / total_plays
    avg_play_efficiency = sum(play.play_efficiency for play in plays) / total_plays
    
    return {
        "total_plays": total_plays,
        "successful_plays": successful_plays,
        "success_rate": successful_plays / total_plays,
        "avg_success_percentage": avg_success_rate,
        "avg_lebron_score": avg_lebron_score,
        "avg_play_efficiency": avg_play_efficiency
    }

# Game VOD Endpoints
@app.post("/games/{game_id}/vod")
async def upload_game_vod(
    game_id: int,
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Verify game exists
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Create game-specific directory
    game_vod_dir = os.path.join(UPLOAD_DIR, f"game_{game_id}")
    os.makedirs(game_vod_dir, exist_ok=True)
    
    # Save video file
    video_path = os.path.join(game_vod_dir, f"game_{game_id}_vod.mp4")
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")
    
    # Get video metadata
    import cv2
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create VOD record
    vod_info = GameVOD(
        game_id=game_id,
        video_path=video_path,
        upload_date=datetime.now(),
        duration=duration,
        resolution=f"{width}x{height}",
        file_size=os.path.getsize(video_path),
        status="ready"
    )
    
    return {
        "message": "Game VOD uploaded successfully",
        "vod_info": vod_info
    }

@app.get("/games/{game_id}/vod")
async def get_game_vod(game_id: int, db: Session = Depends(get_db)):
    # Verify game exists
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Check if VOD exists
    video_path = os.path.join(UPLOAD_DIR, f"game_{game_id}", f"game_{game_id}_vod.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Game VOD not found")
    
    # Get video metadata
    import cv2
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    vod_info = GameVOD(
        game_id=game_id,
        video_path=video_path,
        upload_date=datetime.fromtimestamp(os.path.getctime(video_path)),
        duration=duration,
        resolution=f"{width}x{height}",
        file_size=os.path.getsize(video_path),
        status="ready"
    )
    
    return vod_info

@app.get("/games/{game_id}/vod/stream")
async def stream_game_vod(game_id: int, db: Session = Depends(get_db)):
    # Verify game exists
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Check if VOD exists
    video_path = os.path.join(UPLOAD_DIR, f"game_{game_id}", f"game_{game_id}_vod.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Game VOD not found")
    
    # Stream video file
    from fastapi.responses import StreamingResponse
    def iterfile():
        with open(video_path, "rb") as file:
            while chunk := file.read(1024*1024):  # 1MB chunks
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="game_{game_id}_vod.mp4"'
        }
    )

@app.get("/games/{game_id}/vod/plays")
async def get_game_vod_plays(
    game_id: int,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    play_type: Optional[PlayType] = None,
    db: Session = Depends(get_db)
):
    # Verify game exists
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get plays for the game
    query = db.query(Play).filter(Play.game_id == game_id)
    
    # Apply filters
    if start_time is not None:
        query = query.filter(Play.timestamp >= start_time)
    if end_time is not None:
        query = query.filter(Play.timestamp <= end_time)
    if play_type is not None:
        query = query.filter(Play.play_type == play_type)
    
    plays = query.order_by(Play.timestamp).all()
    
    return [
        {
            "play_id": play.id,
            "play_type": play.play_type,
            "timestamp": play.timestamp,
            "success": play.success,
            "diagram_path": play.diagram_path,
            "clip_path": play.clip_path
        }
        for play in plays
    ]

@app.get("/games/search")
def search_games(query: str = Query(..., description="Team name or date (YYYY-MM-DD)"), db: Session = Depends(get_db)):
    # Try to match by team name or date
    games = db.query(Game)
    if query:
        # Try to parse as date
        from datetime import datetime
        try:
            date = datetime.strptime(query, "%Y-%m-%d").date()
            games = games.filter(Game.date == date)
        except Exception:
            # Not a date, search by team name (case-insensitive)
            games = games.filter((Game.home_team.ilike(f"%{query}%")) | (Game.away_team.ilike(f"%{query}%")))
    games = games.order_by(Game.date.desc()).all()
    return [
        {
            "id": g.id,
            "date": g.date.strftime("%Y-%m-%d") if g.date else None,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "season": g.season,
            "game_type": g.game_type,
            "venue": g.venue
        }
        for g in games
    ]

@app.get("/nba/fetch-game")
def fetch_and_create_game(
    game_date: str = Query(..., description="Game date in YYYY-MM-DD format"),
    home_team: str = Query(..., description="Home team name"),
    away_team: str = Query(..., description="Away team name"),
    db: Session = Depends(get_db)
):
    """Fetch real NBA game data and create game in database"""
    try:
        # Fetch game data from NBA API
        game_data = fetch_nba_game(game_date, home_team, away_team)
        
        # Extract game details from boxscore
        boxscore = game_data['boxscore']
        game_info = boxscore['game']
        
        # Create game in database
        game = Game(
            home_team=home_team,
            away_team=away_team,
            date=datetime.strptime(game_date, "%Y-%m-%d"),
            season="2024-25",
            game_type="regular",
            venue=game_info['arena']['name'],
            home_score=game_info['hTeam']['score'],
            away_score=game_info['vTeam']['score']
        )
        db.add(game)
        db.commit()
        db.refresh(game)
        
        # Process play-by-play data
        pbp = game_data['play_by_play']
        plays_data = pbp['plays']
        
        for play in plays_data:
            # Create play in database
            db_play = Play(
                game_id=game.id,
                play_type=play.get('actionType', 'Unknown'),
                play_description=play.get('description', ''),
                timestamp=play.get('clock', ''),
                success=play.get('shotResult', '') == 'Made',
                score_differential=play.get('scoreDiff', 0),
                time_remaining=play.get('timeActual', '')
            )
            db.add(db_play)
        
        db.commit()
        
        return {
            "message": "Game and plays created successfully",
            "game_id": game.id
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/{game_id}")
def get_game(game_id: int, db: Session = Depends(get_db)):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return {
        "id": game.id,
        "date": game.date.strftime("%Y-%m-%d") if game.date else None,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "season": game.season,
        "game_type": game.game_type,
        "venue": game.venue
    }

def fetch_nba_game(game_date: str, home_team: str, away_team: str):
    """Fetch game data from NBA API"""
    try:
        # Get scoreboard for the date
        games = scoreboard.Scoreboard(
            day_offset=0,
            game_date=game_date,
            league_id="00"
        )
        
        # Find the specific game
        game_id = None
        for game in games.get_dict()['resultSets'][0]['rowSet']:
            if (game[6] == home_team and game[7] == away_team) or \
               (game[6] == away_team and game[7] == home_team):
                game_id = game[2]
                break
        
        if not game_id:
            raise ValueError("Game not found")
            
        # Get detailed game data
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
            game_id=game_id
        )
        
        # Get play-by-play data
        pbp = playbyplayv2.PlayByPlayV2(
            game_id=game_id,
            start_period=1,
            end_period=4
        )
        
        return {
            'game_id': game_id,
            'boxscore': boxscore.get_dict(),
            'play_by_play': pbp.get_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching NBA data: {str(e)}")

@app.post("/games/{game_id}/process-plays")
async def process_game_plays(
    game_id: int,
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Process a game video to extract play schemes"""
    # Verify game exists
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Save video file
    video_path = f"game_vods/game_{game_id}/game_video.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    with open(video_path, "wb") as f:
        content = await video_file.read()
        f.write(content)
    
    # Process video
    try:
        result = process_game_video(video_path, f"play_analysis/game_{game_id}")
        
        # Update game metadata
        game.processed_plays = len(result["plays"])
        game.last_processed = datetime.now()
        db.commit()
        
        return {
            "message": "Game video processed successfully",
            "total_plays": len(result["plays"]),
            "play_types": list(set(p["play_type"] for p in result["plays"])),
            "analysis_path": f"play_analysis/game_{game_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/{game_id}/play-schemes")
def get_game_play_schemes(
    game_id: int,
    play_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get play schemes for a specific game"""
    # Verify game exists
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Load play analysis
    analysis_path = f"play_analysis/game_{game_id}/play_analysis.json"
    if not os.path.exists(analysis_path):
        raise HTTPException(status_code=404, detail="Play analysis not found")
    
    with open(analysis_path, "r") as f:
        analysis = json.load(f)
    
    # Filter by play type if specified
    if play_type:
        analysis["plays"] = [p for p in analysis["plays"] if p["play_type"] == play_type]
    
    return analysis

@app.get("/play-schemes/types")
def get_play_scheme_types(db: Session = Depends(get_db)):
    """Get all available play scheme types"""
    # Load all play analyses
    play_types = set()
    for game_dir in os.listdir("play_analysis"):
        analysis_path = f"play_analysis/{game_dir}/play_analysis.json"
        if os.path.exists(analysis_path):
            with open(analysis_path, "r") as f:
                analysis = json.load(f)
                play_types.update(p["play_type"] for p in analysis["plays"])
    
    return {
        "play_types": sorted(list(play_types)),
        "total_types": len(play_types)
    }

@app.get("/play-schemes/analysis")
def analyze_play_schemes(
    play_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Analyze play schemes across all games"""
    # Load all play analyses
    all_plays = []
    for game_dir in os.listdir("play_analysis"):
        analysis_path = f"play_analysis/{game_dir}/play_analysis.json"
        if os.path.exists(analysis_path):
            with open(analysis_path, "r") as f:
                analysis = json.load(f)
                all_plays.extend(analysis["plays"])
    
    # Filter by play type if specified
    if play_type:
        all_plays = [p for p in all_plays if p["play_type"] == play_type]
    
    # Calculate statistics
    play_types = {}
    for play in all_plays:
        if play["play_type"] not in play_types:
            play_types[play["play_type"]] = {
                "count": 0,
                "total_frames": 0,
                "avg_duration": 0
            }
        
        play_types[play["play_type"]]["count"] += 1
        play_types[play["play_type"]]["total_frames"] += (
            play["frame_end"] - play["frame_start"]
        )
    
    # Calculate averages
    for play_type in play_types:
        play_types[play_type]["avg_duration"] = (
            play_types[play_type]["total_frames"] / play_types[play_type]["count"]
        )
    
    return {
        "total_plays": len(all_plays),
        "play_types": play_types
    }

@app.get("/nba/teams")
def get_nba_teams():
    """Get all NBA teams"""
    return teams.get_teams()

@app.get("/nba/players")
def get_nba_players():
    """Get all NBA players"""
    return players.get_players()

@app.get("/nba/live-games")
def get_live_games():
    """Get currently live NBA games"""
    games = live_scoreboard.ScoreBoard()
    return games.get_dict()

@app.get("/nba/player/{player_id}/career")
def get_player_career(player_id: str):
    """Get player career statistics"""
    from nba_api.stats.endpoints import playercareerstats
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    return career.get_dict()

@app.get("/nba/games/{team_id}")
def get_team_games(
    team_id: str,
    season: str = "2023-24",
    season_type: str = "Regular Season"
):
    """Get all games for a specific team"""
    games = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable=season_type
    )
    return games.get_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 