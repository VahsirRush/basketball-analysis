import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from datetime import datetime, timedelta

# Fetch all games for the 2022-23 season
games = leaguegamelog.LeagueGameLog(season='2022-23', season_type_all_star='Regular Season')
df = games.get_data_frames()[0]

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Set the start and end dates for the last 3 months of the regular season
start_date = pd.to_datetime('2023-01-09')
end_date = pd.to_datetime('2023-04-09')

# Filter games from the last 3 months of the regular season
recent_games = df[(df['GAME_DATE'] >= start_date) & (df['GAME_DATE'] <= end_date)]

# Save the filtered schedule to a CSV file
recent_games.to_csv('nba_schedule_last_3_months.csv', index=False)

print(f"Fetched {len(recent_games)} games from the last 3 months. Schedule saved to 'nba_schedule_last_3_months.csv'.") 