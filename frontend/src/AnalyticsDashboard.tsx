import React, { useMemo } from 'react';
import type { 
  Play, 
  PlayTypeDistribution, 
  SuccessRate, 
  MovementPattern, 
  TeamMetrics,
  Player,
  Movement
} from './types';

interface AnalyticsDashboardProps {
  plays: Play[];
  isLoading?: boolean;
  error?: string | null;
}

interface PlayTypeStats {
  total: number;
  successful: number;
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ 
  plays, 
  isLoading = false, 
  error = null 
}) => {
  // Calculate play type distribution
  const playTypeDistribution = useMemo<PlayTypeDistribution[]>(() => {
    if (!plays.length) return [];
    
    const distribution = plays.reduce((acc, play) => {
      acc[play.type] = (acc[play.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(distribution).map(([type, count]) => ({
      type,
      count: count as number,
      percentage: ((count as number) / plays.length) * 100
    }));
  }, [plays]);

  // Calculate success rates by play type
  const successRates = useMemo<SuccessRate[]>(() => {
    if (!plays.length) return [];
    
    const rates = plays.reduce((acc, play) => {
      if (!acc[play.type]) {
        acc[play.type] = { total: 0, successful: 0 };
      }
      acc[play.type].total++;
      if (play.success) {
        acc[play.type].successful++;
      }
      return acc;
    }, {} as Record<string, PlayTypeStats>);

    return Object.entries(rates).map(([type, stats]) => {
      const typedStats = stats as PlayTypeStats;
      return {
        type,
        successRate: (typedStats.successful / typedStats.total) * 100
      };
    });
  }, [plays]);

  // Calculate player movement patterns
  const movementPatterns = useMemo<MovementPattern[]>(() => {
    if (!plays.length) return [];
    
    const patterns = plays.reduce((acc, play) => {
      play.players.forEach((player: Player) => {
        player.movements.forEach((movement: Movement) => {
          if (movement.action) {
            if (!acc[movement.action]) {
              acc[movement.action] = 0;
            }
            acc[movement.action]++;
          }
        });
      });
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(patterns)
      .map(([action, count]) => ({ action, count: count as number }))
      .sort((a, b) => (b.count as number) - (a.count as number));
  }, [plays]);

  // Calculate team performance metrics
  const teamMetrics = useMemo<Record<string, TeamMetrics>>(() => {
    if (!plays.length) return {};
    
    const metrics = plays.reduce((acc, play) => {
      const team = play.team_possession;
      if (!acc[team]) {
        acc[team] = {
          totalPlays: 0,
          successfulPlays: 0,
          totalPoints: 0,
          avgPlayDuration: 0
        };
      }
      acc[team].totalPlays++;
      if (play.success) {
        acc[team].successfulPlays++;
      }
      // Safely access team score
      const teamScore = play.score[team as keyof typeof play.score] || 0;
      acc[team].totalPoints += teamScore;
      acc[team].avgPlayDuration += (play.end_time - play.start_time);
      return acc;
    }, {} as Record<string, TeamMetrics>);

    // Calculate averages
    Object.keys(metrics).forEach(team => {
      if (metrics[team].totalPlays > 0) {
        metrics[team].avgPlayDuration /= metrics[team].totalPlays;
      }
    });

    return metrics;
  }, [plays]);

  if (isLoading) {
    return (
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-8"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-64"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium">Error Loading Analytics</h3>
          <p className="text-red-600 mt-2">{error}</p>
        </div>
      </div>
    );
  }

  if (!plays.length) {
    return (
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h3 className="text-gray-800 font-medium">No Data Available</h3>
          <p className="text-gray-600 mt-2">Upload a video to see analytics.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Play Type Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Play Type Distribution</h3>
          <div className="space-y-4">
            {playTypeDistribution.map(({ type, count, percentage }) => (
              <div key={type} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{type}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-600">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Success Rates */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Success Rates by Play Type</h3>
          <div className="space-y-4">
            {successRates.map(({ type, successRate }) => (
              <div key={type} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{type}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${successRate}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-600">{successRate.toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Movement Patterns */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Common Movement Patterns</h3>
          <div className="space-y-4">
            {movementPatterns.slice(0, 5).map(({ action, count }) => (
              <div key={action} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{action}</span>
                <span className="text-sm text-gray-600">{count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Team Performance */}
        <div className="bg-white rounded-lg shadow p-6 md:col-span-2 lg:col-span-3">
          <h3 className="text-lg font-semibold mb-4">Team Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(teamMetrics).map(([team, metrics]) => (
              <div key={team} className="space-y-4">
                <h4 className="font-medium text-gray-900">{team.toUpperCase()}</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Plays</span>
                    <span className="text-sm font-medium">{metrics.totalPlays}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Success Rate</span>
                    <span className="text-sm font-medium">
                      {((metrics.successfulPlays / metrics.totalPlays) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Points</span>
                    <span className="text-sm font-medium">{metrics.totalPoints}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Play Duration</span>
                    <span className="text-sm font-medium">
                      {metrics.avgPlayDuration.toFixed(1)}s
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard; 