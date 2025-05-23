from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from .database import Base
import enum
from datetime import datetime

class Game(Base):
    __tablename__ = 'games'
    id = Column(Integer, primary_key=True, index=True)
    home_team = Column(String, index=True)
    away_team = Column(String, index=True)
    date = Column(DateTime)
    season = Column(String)
    game_type = Column(String)  # regular, playoff, finals
    game_number = Column(Integer)  # For playoff series
    venue = Column(String)
    home_score = Column(Integer)
    away_score = Column(Integer)
    series_id = Column(Integer, nullable=True)  # For playoff series
    round = Column(String, nullable=True)  # first_round, conference_semifinals, conference_finals, finals
    series_game_number = Column(Integer, nullable=True)  # Game number in the series (1-7)
    is_highlight = Column(Boolean, default=False)  # Whether this is a highlight reel
    highlight_type = Column(String, nullable=True)  # game_highlights, series_highlights, season_highlights
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    plays = relationship("Play", back_populates="game")

class PlayType(str, enum.Enum):
    PICK_AND_ROLL = "pick_and_roll"
    ISOLATION = "isolation"
    POST_UP = "post_up"
    SPOT_UP = "spot_up"
    HANDOFF = "handoff"

class PlayCategory(str, enum.Enum):
    SCORING = "scoring"
    CREATION = "creation"
    RESET = "reset"
    TRANSITION = "transition"
    SPECIAL = "special"

class Play(Base):
    __tablename__ = "plays"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    play_type = Column(Enum(PlayType))
    play_category = Column(Enum(PlayCategory))
    play_variation = Column(String)
    play_name = Column(String)
    play_code = Column(String, unique=True, index=True)
    play_description = Column(String)
    play_intent = Column(String)
    expected_outcome = Column(String)
    player_role = Column(String)
    defensive_scheme = Column(String)
    success = Column(Boolean)
    success_percentage = Column(Float)
    avg_points_per_attempt = Column(Float)
    shot_priority = Column(String)
    lebron_score = Column(Float)
    pressure_adjusted_shooting = Column(Float)
    score_differential = Column(Integer)
    time_remaining = Column(Integer)
    possession = Column(Integer)
    offensive_player_id = Column(Integer, ForeignKey("players.id"))
    defensive_player_id = Column(Integer, ForeignKey("players.id"))
    play_efficiency = Column(Float)
    defensive_rating = Column(Float)
    play_sequence = Column(Integer)
    counter_adjustment = Column(String)
    diagram_path = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    timestamp = Column(Float, nullable=True)

    # Relationships
    game = relationship("Game", back_populates="plays")
    offensive_player = relationship("Player", foreign_keys=[offensive_player_id])
    defensive_player = relationship("Player", foreign_keys=[defensive_player_id])
    biometrics = relationship("PlayerBiometrics", back_populates="play")

class PlayCollection(Base):
    __tablename__ = "play_collections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    play_type = Column(Enum(PlayType))
    play_category = Column(Enum(PlayCategory))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    # Relationships
    plays = relationship("Play", secondary="play_collection_items")
    items = relationship("PlayCollectionItem", back_populates="collection")

class PlayCollectionItem(Base):
    __tablename__ = "play_collection_items"

    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("play_collections.id"))
    play_id = Column(Integer, ForeignKey("plays.id"))
    order = Column(Integer)
    notes = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    # Relationships
    collection = relationship("PlayCollection", back_populates="items")
    play = relationship("Play")

class PlayerBiometrics(Base):
    __tablename__ = 'player_biometrics'
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer)
    heart_rate = Column(Integer)
    sweat_rate = Column(Float)
    fatigue_level = Column(Float)
    movement_speed = Column(Float)
    jump_height = Column(Float)
    reaction_time = Column(Float)
    game_id = Column(Integer, ForeignKey('games.id'))
    play_id = Column(Integer, ForeignKey('plays.id'))
    height = Column(Float)
    weight = Column(Float)
    age = Column(Integer)
    play = relationship("Play", back_populates="biometrics")

class Player(Base):
    __tablename__ = 'players'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=True)
    team = Column(String, nullable=True)
    number = Column(Integer, nullable=True)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    # Add any other fields as needed 