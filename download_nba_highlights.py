import pandas as pd
import os
import subprocess

# Parameters
CSV_PATH = 'nba_schedule_last_3_months.csv'
VIDEOS_DIR = 'videos'
SAMPLE_SIZE = 5

os.makedirs(VIDEOS_DIR, exist_ok=True)

def sanitize_filename(s):
    return ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in s)

def main():
    df = pd.read_csv(CSV_PATH)
    sample = df.head(SAMPLE_SIZE)
    for idx, row in sample.iterrows():
        date = pd.to_datetime(row['GAME_DATE']).strftime('%Y-%m-%d')
        away = row['MATCHUP'].split(' at ')[0].strip()
        home = row['MATCHUP'].split(' at ')[1].strip() if ' at ' in row['MATCHUP'] else 'Unknown'
        query = f"{away} vs {home} {date} NBA highlights"
        filename = sanitize_filename(f"{date}_{away}_vs_{home}.mp4")
        out_path = os.path.join(VIDEOS_DIR, filename)
        print(f"Searching and downloading: {query}")
        cmd = [
            'yt-dlp',
            '--quiet',
            '--no-warnings',
            '--output', out_path,
            f"ytsearch1:{query}"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Downloaded to {out_path}")
        except subprocess.CalledProcessError:
            print(f"Failed to download for: {query}")

if __name__ == '__main__':
    main() 