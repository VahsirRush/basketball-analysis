from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from .models import PlayType, PlayCategory

# ... existing schemas ...

class PlayBase(BaseModel):
    game_id: int
    play_type: PlayType
    play_category: PlayCategory
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
    diagram_path: Optional[str] = None
    clip_path: Optional[str] = None
    timestamp: Optional[float] = None

class PlayCreate(PlayBase):
    pass

class Play(PlayBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PlayCollectionBase(BaseModel):
    name: str
    description: str
    play_type: Optional[PlayType] = None
    play_category: Optional[PlayCategory] = None

class PlayCollectionCreate(PlayCollectionBase):
    pass

class PlayCollection(PlayCollectionBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PlayCollectionItemBase(BaseModel):
    order: int
    notes: Optional[str] = None

class PlayCollectionItem(PlayCollectionItemBase):
    id: int
    collection_id: int
    play_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PlayCollectionDetail(PlayCollection):
    plays: List[Play]

class PlayStatistics(BaseModel):
    total_plays: int
    successful_plays: int
    success_rate: float
    avg_success_percentage: float
    avg_lebron_score: float
    avg_play_efficiency: float

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
    series_id: Optional[int] = None
    round: Optional[str] = None
    series_game_number: Optional[int] = None
    is_highlight: Optional[bool] = False
    highlight_type: Optional[str] = None

class GameCreate(GameBase):
    pass

class GameResponse(GameBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 