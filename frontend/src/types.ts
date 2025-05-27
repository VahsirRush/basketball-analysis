export interface Keypoint {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface Pose {
  keypoints: Keypoint[];
}

export interface Movement {
  frame: number;
  timestamp: number;
  position: [number, number];
  action: string | null;
  pose: Pose;
}

export interface Player {
  id: number;
  role: string;
  movements: Movement[];
}

export interface Play {
  id: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  type: string;
  players: Player[];
  team_possession: string;
  score: {
    home: number;
    away: number;
  };
  success?: boolean;
  confidence?: number;
}

export interface PlayDiagram {
  id: string;
  path: string;
  category: string;
  subcategory: string;
  description: string;
  key_players: string[];
  movements: string[];
  variations: string[];
}

export interface Analysis {
  id: string;
  videoId: string;
  status: 'processing' | 'completed' | 'failed';
  plays?: Play[];
  playDiagrams?: PlayDiagram[];
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Video {
  id: number;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  error_message?: string;
  progress?: number;
  created_at: string;
  processed_at?: string;
}

export interface PlayTypeDistribution {
  type: string;
  count: number;
  percentage: number;
}

export interface SuccessRate {
  type: string;
  successRate: number;
}

export interface MovementPattern {
  action: string;
  count: number;
}

export interface TeamMetrics {
  totalPlays: number;
  successfulPlays: number;
  totalPoints: number;
  avgPlayDuration: number;
} 