import React, { useState, useCallback, useMemo, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, PlayDiagram, Player, Movement, Pose, Keypoint } from './types';

interface PlayVisualizerProps {
  plays: Play[];
  onPlaySelect?: (play: Play) => void;
  playDiagrams?: PlayDiagram[];
}

// Court dimensions (NBA half-court, feet, for scaling)
const COURT_WIDTH = 50; // feet
const COURT_HEIGHT = 47; // feet
const SVG_WIDTH = 800;
const SVG_HEIGHT = 752;

function scaleX(x: number) {
  return (x / COURT_WIDTH) * SVG_WIDTH;
}

function scaleY(y: number) {
  return (y / COURT_HEIGHT) * SVG_HEIGHT;
}

type TeamPossession = 'home' | 'away' | 'unknown';

const teamColor: Record<TeamPossession, string> = {
  home: '#2563eb', // blue
  away: '#dc2626', // red
  unknown: '#6b7280', // gray
};

// Memoize the player movement calculations
const PlayerMovement = memo(({ player, currentFrame, showKeypoints, showActions, teamColor }: {
  player: Player & { currentPosition: [number, number], currentAction: string | null, currentPose: Pose },
  currentFrame: number,
  showKeypoints: boolean,
  showActions: boolean,
  teamColor: string
}) => (
  <g>
    {/* Player movements (trails) */}
    {player.movements.slice(0, currentFrame + 1).map((movement, idx) => (
      <motion.circle
        key={idx}
        cx={scaleX(movement.position[0])}
        cy={scaleY(movement.position[1])}
        r={2}
        fill={teamColor}
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.3 }}
        transition={{ duration: 0.2 }}
      />
    ))}
    
    {/* Player circle */}
    <motion.circle
      cx={scaleX(player.currentPosition[0])}
      cy={scaleY(player.currentPosition[1])}
      r={14}
      fill={teamColor}
      stroke="#fff"
      strokeWidth={2}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: "spring", stiffness: 500, damping: 30 }}
    />
    
    {/* Player ID */}
    <text
      x={scaleX(player.currentPosition[0])}
      y={scaleY(player.currentPosition[1]) + 5}
      textAnchor="middle"
      fontSize={14}
      fill="#fff"
      fontWeight="bold"
    >
      {player.id}
    </text>

    {/* Keypoints */}
    {showKeypoints && player.currentPose.keypoints.map((keypoint, idx) => (
      keypoint.visibility > 0.5 && (
        <motion.circle
          key={idx}
          cx={scaleX(player.currentPosition[0] + keypoint.x)}
          cy={scaleY(player.currentPosition[1] + keypoint.y)}
          r={3}
          fill="#f59e0b"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: idx * 0.02 }}
        />
      )
    ))}

    {/* Action label */}
    {showActions && player.currentAction && (
      <motion.text
        x={scaleX(player.currentPosition[0])}
        y={scaleY(player.currentPosition[1]) - 20}
        textAnchor="middle"
        fontSize={12}
        fill="#374151"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {player.currentAction}
      </motion.text>
    )}
  </g>
));

const PlayVisualizer: React.FC<PlayVisualizerProps> = ({ plays, onPlaySelect, playDiagrams = [] }) => {
  const [selectedPlayId, setSelectedPlayId] = useState<number>(plays[0]?.id || 0);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [showActions, setShowActions] = useState(true);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showDiagram, setShowDiagram] = useState(true);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparisonPlayId, setComparisonPlayId] = useState<number | null>(null);

  const selectedPlay = useMemo(() => 
    plays.find(play => play.id === selectedPlayId),
    [plays, selectedPlayId]
  );

  const currentPlayers = useMemo(() => 
    selectedPlay ? selectedPlay.players.map(player => {
    const movement = player.movements.find(m => m.frame === currentFrame) || player.movements[0];
    return {
      ...player,
      currentPosition: movement.position,
      currentAction: movement.action,
      currentPose: movement.pose,
    };
    }) : [],
    [selectedPlay, currentFrame]
  );

  const selectedPlayDiagram = useMemo(() => {
    if (!playDiagrams.length) return null;
    return playDiagrams.find(diagram => 
      diagram.category.toLowerCase() === selectedPlay?.type.toLowerCase()
    );
  }, [playDiagrams, selectedPlay]);

  const comparisonPlay = useMemo(() => 
    plays.find(play => play.id === comparisonPlayId),
    [plays, comparisonPlayId]
  );

  const handlePlaySelect = useCallback((playId: number) => {
    setSelectedPlayId(playId);
    setCurrentFrame(0);
    setIsPlaying(false);
    onPlaySelect?.(plays.find(play => play.id === playId) as Play);
  }, [onPlaySelect, plays]);

  const handleFrameChange = useCallback((frame: number) => {
    setCurrentFrame(frame);
  }, []);

  const togglePlayback = useCallback(() => {
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const handleComparisonSelect = useCallback((playId: number | null) => {
    setComparisonPlayId(playId);
  }, []);

  if (!selectedPlay) {
    return <div>No plays to display.</div>;
  }

  return (
    <div className="w-full flex flex-col items-center space-y-4">
      {/* Enhanced Controls */}
      <div className="w-full max-w-4xl flex flex-col space-y-4 p-4 bg-white rounded-lg shadow">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <select
              value={selectedPlayId}
              onChange={(e) => handlePlaySelect(Number(e.target.value))}
              className="px-3 py-2 border rounded-md"
            >
              {plays.map(play => (
                <option key={play.id} value={play.id}>
                  Play #{play.id} - {play.type}
                </option>
              ))}
            </select>
            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showKeypoints}
                  onChange={(e) => setShowKeypoints(e.target.checked)}
                  className="form-checkbox"
                />
                <span>Show Keypoints</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showActions}
                  onChange={(e) => setShowActions(e.target.checked)}
                  className="form-checkbox"
                />
                <span>Show Actions</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showDiagram}
                  onChange={(e) => setShowDiagram(e.target.checked)}
                  className="form-checkbox"
                />
                <span>Show Diagram</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showAnnotations}
                  onChange={(e) => setShowAnnotations(e.target.checked)}
                  className="form-checkbox"
                />
                <span>Show Annotations</span>
              </label>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={togglePlayback}
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <input
              type="range"
              min={selectedPlay.start_frame}
              max={selectedPlay.end_frame}
              value={currentFrame}
              onChange={(e) => handleFrameChange(Number(e.target.value))}
              className="w-64"
            />
            <span className="text-sm text-gray-500">
              Frame: {currentFrame} / {selectedPlay.end_frame}
            </span>
          </div>
        </div>

        {/* Comparison Controls */}
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={comparisonMode}
              onChange={(e) => setComparisonMode(e.target.checked)}
              className="form-checkbox"
            />
            <span>Compare Plays</span>
          </label>
          {comparisonMode && (
            <select
              value={comparisonPlayId || ''}
              onChange={(e) => handleComparisonSelect(e.target.value ? Number(e.target.value) : null)}
              className="px-3 py-2 border rounded-md"
            >
              <option value="">Select play to compare</option>
              {plays.filter(play => play.id !== selectedPlayId).map(play => (
                <option key={play.id} value={play.id}>
                  Play #{play.id} - {play.type}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Main Visualization Area */}
      <div className="flex space-x-4">
        {/* Court Visualization */}
        <div className="relative">
          <svg
            viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
            width={SVG_WIDTH}
            height={SVG_HEIGHT}
            className="bg-green-50 border border-gray-300 rounded shadow"
          >
            {/* Draw court outline */}
            <rect x={0} y={0} width={SVG_WIDTH} height={SVG_HEIGHT} fill="#f3f4f6" stroke="#bbb" strokeWidth={4} />
            
            {/* Center circle */}
            <circle cx={SVG_WIDTH / 2} cy={SVG_HEIGHT / 2} r={60} fill="none" stroke="#bbb" strokeWidth={2} />
            
            {/* Paint area */}
            <rect
              x={scaleX(17)}
              y={scaleY(0)}
              width={scaleX(16)}
              height={scaleY(19)}
              fill="#e5e7eb"
              stroke="#bbb"
              strokeWidth={2}
            />

            {/* Players */}
            {currentPlayers.map(player => (
              <PlayerMovement
                key={player.id}
                player={player}
                currentFrame={currentFrame}
                showKeypoints={showKeypoints}
                showActions={showActions}
                teamColor={teamColor[selectedPlay.team_possession as TeamPossession]}
              />
            ))}

            {/* Comparison Players */}
            {comparisonMode && comparisonPlay && (
              comparisonPlay.players.map(player => {
                const movement = player.movements.find(m => m.frame === currentFrame) || player.movements[0];
                return (
                  <PlayerMovement
                    key={`comp-${player.id}`}
                    player={{
                      ...player,
                      currentPosition: movement.position,
                      currentAction: movement.action,
                      currentPose: movement.pose,
                    }}
                    currentFrame={currentFrame}
                    showKeypoints={showKeypoints}
                    showActions={showActions}
                    teamColor={teamColor[comparisonPlay.team_possession as TeamPossession]}
                  />
                );
              })
            )}

            {/* Play boundary indicators */}
            <motion.rect
              x={0}
              y={0}
              width={SVG_WIDTH}
              height={SVG_HEIGHT}
              fill="none"
              stroke="#374151"
              strokeWidth={4}
              strokeDasharray="10,10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.3 }}
              transition={{ duration: 0.5 }}
            />
          </svg>

          {/* Play info overlay */}
          <div className="absolute top-4 left-4 bg-white/90 p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold">Play #{selectedPlay.id}</h3>
            <p className="text-sm text-gray-600">Type: {selectedPlay.type}</p>
            <p className="text-sm text-gray-600">
              Time: {selectedPlay.start_time.toFixed(1)}s - {selectedPlay.end_time.toFixed(1)}s
            </p>
            <p className="text-sm text-gray-600">
              Score: {selectedPlay.score.home} - {selectedPlay.score.away}
            </p>
          </div>
        </div>

        {/* Play Diagram and Annotations */}
        {showDiagram && selectedPlayDiagram && (
          <div className="w-64 bg-white rounded-lg shadow p-4">
            <img
              src={selectedPlayDiagram.path}
              alt={`${selectedPlayDiagram.category} diagram`}
              className="w-full h-auto mb-4"
            />
            {showAnnotations && (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold">Description</h4>
                  <p className="text-sm text-gray-600">{selectedPlayDiagram.description}</p>
                </div>
                <div>
                  <h4 className="font-semibold">Key Players</h4>
                  <ul className="text-sm text-gray-600 list-disc list-inside">
                    {selectedPlayDiagram.key_players.map((player, idx) => (
                      <li key={idx}>{player}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold">Movements</h4>
                  <ul className="text-sm text-gray-600 list-disc list-inside">
                    {selectedPlayDiagram.movements.map((movement, idx) => (
                      <li key={idx}>{movement}</li>
                    ))}
                  </ul>
                </div>
                {selectedPlayDiagram.variations.length > 0 && (
                  <div>
                    <h4 className="font-semibold">Variations</h4>
                    <ul className="text-sm text-gray-600 list-disc list-inside">
                      {selectedPlayDiagram.variations.map((variation, idx) => (
                        <li key={idx}>{variation}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(PlayVisualizer); 