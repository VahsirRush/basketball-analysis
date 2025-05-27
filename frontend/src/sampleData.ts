import { Play } from './types';

export const samplePlays: Play[] = [
  {
    id: 1,
    start_frame: 0,
    end_frame: 120,
    start_time: 0,
    end_time: 4,
    type: "Pick and Roll",
    players: [
      {
        id: 1,
        role: "Point Guard",
        movements: [
          { frame: 0, timestamp: 0, position: [10, 20], action: "Dribble", pose: { keypoints: [] } },
          { frame: 60, timestamp: 2, position: [15, 25], action: "Pass", pose: { keypoints: [] } }
        ]
      },
      {
        id: 2,
        role: "Center",
        movements: [
          { frame: 0, timestamp: 0, position: [20, 30], action: "Set Screen", pose: { keypoints: [] } },
          { frame: 60, timestamp: 2, position: [25, 35], action: "Roll", pose: { keypoints: [] } }
        ]
      }
    ],
    team_possession: "home",
    score: { home: 85, away: 80 },
    success: true,
    confidence: 0.95
  },
  {
    id: 2,
    start_frame: 120,
    end_frame: 240,
    start_time: 4,
    end_time: 8,
    type: "Post Up",
    players: [
      {
        id: 3,
        role: "Power Forward",
        movements: [
          { frame: 120, timestamp: 4, position: [30, 40], action: "Post Up", pose: { keypoints: [] } },
          { frame: 180, timestamp: 6, position: [32, 42], action: "Shoot", pose: { keypoints: [] } }
        ]
      }
    ],
    team_possession: "away",
    score: { home: 85, away: 82 },
    success: false,
    confidence: 0.88
  },
  {
    id: 3,
    start_frame: 240,
    end_frame: 360,
    start_time: 8,
    end_time: 12,
    type: "Pick and Roll",
    players: [
      {
        id: 1,
        role: "Point Guard",
        movements: [
          { frame: 240, timestamp: 8, position: [15, 25], action: "Dribble", pose: { keypoints: [] } },
          { frame: 300, timestamp: 10, position: [20, 30], action: "Pass", pose: { keypoints: [] } }
        ]
      },
      {
        id: 2,
        role: "Center",
        movements: [
          { frame: 240, timestamp: 8, position: [25, 35], action: "Set Screen", pose: { keypoints: [] } },
          { frame: 300, timestamp: 10, position: [30, 40], action: "Roll", pose: { keypoints: [] } }
        ]
      }
    ],
    team_possession: "home",
    score: { home: 87, away: 82 },
    success: true,
    confidence: 0.92
  },
  {
    id: 4,
    start_frame: 360,
    end_frame: 480,
    start_time: 12,
    end_time: 16,
    type: "Off Ball Screen",
    players: [
      {
        id: 4,
        role: "Shooting Guard",
        movements: [
          { frame: 360, timestamp: 12, position: [40, 20], action: "Run", pose: { keypoints: [] } },
          { frame: 420, timestamp: 14, position: [45, 25], action: "Shoot", pose: { keypoints: [] } }
        ]
      },
      {
        id: 5,
        role: "Small Forward",
        movements: [
          { frame: 360, timestamp: 12, position: [35, 25], action: "Set Screen", pose: { keypoints: [] } },
          { frame: 420, timestamp: 14, position: [40, 30], action: "Roll", pose: { keypoints: [] } }
        ]
      }
    ],
    team_possession: "away",
    score: { home: 87, away: 85 },
    success: true,
    confidence: 0.85
  }
]; 