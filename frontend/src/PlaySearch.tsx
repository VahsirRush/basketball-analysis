import React, { useState, useMemo } from 'react';
import { Play } from './types';

interface PlaySearchProps {
  plays: Play[];
  onFilterChange: (filteredPlays: Play[]) => void;
}

const PlaySearch: React.FC<PlaySearchProps> = ({ plays, onFilterChange }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPlayType, setSelectedPlayType] = useState<string>('');
  const [selectedPlayerRole, setSelectedPlayerRole] = useState<string>('');
  const [sortBy, setSortBy] = useState<'confidence' | 'time' | 'type'>('confidence');
  const [groupByType, setGroupByType] = useState(false);

  // Get unique play types and player roles
  const playTypes = useMemo(() => 
    Array.from(new Set(plays.map(play => play.type))),
    [plays]
  );

  const playerRoles = useMemo(() => 
    Array.from(new Set(plays.flatMap(play => 
      play.players.map(player => player.role)
    ))),
    [plays]
  );

  // Filter and sort plays
  const filteredPlays = useMemo(() => {
    let result = plays;

    // Apply search term filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      result = result.filter(play => 
        play.type.toLowerCase().includes(term) ||
        play.players.some(player => 
          player.role.toLowerCase().includes(term) ||
          player.movements.some(movement => 
            movement.action?.toLowerCase().includes(term)
          )
        )
      );
    }

    // Apply play type filter
    if (selectedPlayType) {
      result = result.filter(play => play.type === selectedPlayType);
    }

    // Apply player role filter
    if (selectedPlayerRole) {
      result = result.filter(play => 
        play.players.some(player => player.role === selectedPlayerRole)
      );
    }

    // Sort plays
    result = [...result].sort((a, b) => {
      switch (sortBy) {
        case 'confidence':
          return (b.confidence || 0) - (a.confidence || 0);
        case 'time':
          return a.start_time - b.start_time;
        case 'type':
          return a.type.localeCompare(b.type);
        default:
          return 0;
      }
    });

    // Group by type if enabled
    if (groupByType) {
      const grouped = result.reduce((acc, play) => {
        if (!acc[play.type]) {
          acc[play.type] = [];
        }
        acc[play.type].push(play);
        return acc;
      }, {} as Record<string, Play[]>);

      result = Object.values(grouped).flat();
    }

    return result;
  }, [plays, searchTerm, selectedPlayType, selectedPlayerRole, sortBy, groupByType]);

  // Update parent component when filters change
  React.useEffect(() => {
    onFilterChange(filteredPlays);
  }, [filteredPlays, onFilterChange]);

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Search Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Search
          </label>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search plays..."
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>

        {/* Play Type Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Play Type
          </label>
          <select
            value={selectedPlayType}
            onChange={(e) => setSelectedPlayType(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          >
            <option value="">All Types</option>
            {playTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        {/* Player Role Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Player Role
          </label>
          <select
            value={selectedPlayerRole}
            onChange={(e) => setSelectedPlayerRole(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          >
            <option value="">All Roles</option>
            {playerRoles.map(role => (
              <option key={role} value={role}>{role}</option>
            ))}
          </select>
        </div>

        {/* Sort By */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Sort By
          </label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="w-full px-3 py-2 border rounded-md"
          >
            <option value="confidence">Confidence</option>
            <option value="time">Time</option>
            <option value="type">Play Type</option>
          </select>
        </div>
      </div>

      {/* Group By Toggle */}
      <div className="mt-4">
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={groupByType}
            onChange={(e) => setGroupByType(e.target.checked)}
            className="form-checkbox"
          />
          <span className="text-sm text-gray-700">Group by Play Type</span>
        </label>
      </div>

      {/* Results Count */}
      <div className="mt-4 text-sm text-gray-600">
        Showing {filteredPlays.length} of {plays.length} plays
      </div>
    </div>
  );
};

export default PlaySearch; 