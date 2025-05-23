-- Create tables for basketball analysis
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    processed_at TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS plays (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    play_type VARCHAR(50),
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    play_id INTEGER REFERENCES plays(id),
    player_number INTEGER,
    team VARCHAR(50),
    position VARCHAR(50),
    tracking_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_plays_video_id ON plays(video_id);
CREATE INDEX IF NOT EXISTS idx_players_play_id ON players(play_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres; 