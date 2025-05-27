import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import PlayVisualizer from './PlayVisualizer';
import AnalyticsDashboard from './AnalyticsDashboard';
import PlaySearch from './PlaySearch';
import { motion } from 'framer-motion';
import { Play, PlayDiagram, Player, Movement, Pose, Keypoint, Analysis, Video } from './types';

// API configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes in milliseconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// Cache for video data with TTL
class VideoCache {
  private cache: Map<string, { data: any; timestamp: number }> = new Map();

  set(key: string, value: any) {
    this.cache.set(key, { data: value, timestamp: Date.now() });
  }

  get(key: string): any | null {
    const item = this.cache.get(key);
    if (!item) return null;
    
    // Check if cache entry has expired
    if (Date.now() - item.timestamp > CACHE_TTL) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data;
  }

  clear() {
    this.cache.clear();
  }
}

const videoCache = new VideoCache();

// Utility function for API calls with retry
async function fetchWithRetry(url: string, options: RequestInit = {}, retries = MAX_RETRIES): Promise<Response> {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  } catch (error) {
    if (retries > 0) {
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
      return fetchWithRetry(url, options, retries - 1);
    }
    throw error;
  }
}

function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<Analysis | null>(null);
  const [filteredPlays, setFilteredPlays] = useState<Play[]>([]);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  // Memoize fetchVideos to prevent unnecessary recreations
  const fetchVideos = useCallback(async () => {
    try {
      console.log('Fetching videos from:', `${API_URL}/videos`);
      const res = await fetch(`${API_URL}/videos`);
      console.log('Response status:', res.status);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to fetch videos: ${res.statusText}`);
      }
      const data = await res.json();
      console.log('Received data:', data);
      setVideos(Array.isArray(data) ? data : []);
      setError(null);
    } catch (err: any) {
      console.error('Error fetching videos:', err);
      setError(err.message || 'Failed to fetch videos. Please ensure the backend server is running at ' + API_URL);
    }
  }, []);

  useEffect(() => {
    console.log('Initial API URL:', API_URL);
    console.log('Starting video fetch...');
    fetchVideos();
    // Set up polling for video status updates
    const pollInterval = setInterval(fetchVideos, 5000);
    return () => clearInterval(pollInterval);
  }, [fetchVideos]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      // Validate file size (100MB limit)
      if (selectedFile.size > 100 * 1024 * 1024) {
        setError('File size must be less than 100MB');
        return;
      }
      // Validate file type
      if (!selectedFile.type.startsWith('video/')) {
        setError('Please select a valid video file');
        return;
      }
      setFile(selectedFile);
      setError(null);
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setSelectedAnalysis(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/upload/video`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Upload failed');
      }
      
      const data = await response.json();
      setFile(null);
      
      // Start polling for progress
      const videoId = data.id;
      const progressInterval = setInterval(async () => {
        try {
          const progressRes = await fetch(`${API_URL}/progress/${videoId}`);
          if (progressRes.ok) {
            const progressData = await progressRes.json();
            // Update the video list with the latest progress
            setVideos(prevVideos => 
              prevVideos.map(v => 
                v.id === videoId 
                  ? { ...v, status: progressData.status, progress: progressData.progress, error_message: progressData.error_message }
                  : v
              )
            );
            
            // Only clear interval if processing is complete or there's an error
            if (progressData.status === 'completed' || progressData.status === 'error') {
              clearInterval(progressInterval);
              // Fetch final video status
              await fetchVideos();
              // If there was an error, show it
              if (progressData.status === 'error' && progressData.error_message) {
                setError(progressData.error_message);
              }
            }
          } else {
            // If progress check fails, don't clear the interval
            console.error('Failed to get progress:', progressRes.statusText);
          }
        } catch (err) {
          console.error('Error checking progress:', err);
          // Don't clear interval on error, keep trying
        }
      }, 1000);
      
      await fetchVideos();
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(err.message || 'Failed to upload video');
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  }, [file, fetchVideos]);

  const handleViewAnalysis = useCallback(async (video: Video) => {
    if (!video.id) return;
    
    // Check cache first
    const cachedAnalysis = videoCache.get(video.id.toString());
    if (cachedAnalysis) {
      setSelectedAnalysis(cachedAnalysis as Analysis);
      return;
    }

    setLoadingAnalysis(true);
    setSelectedAnalysis(null);
    try {
      const res = await fetch(`${API_URL}/analysis/${video.id}`);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to fetch analysis: ${res.statusText}`);
      }
      const data = await res.json();
      
      // Extract plays from analysis metadata if not directly in the response
      if (!data.plays && data.analysis_metadata?.plays) {
        data.plays = data.analysis_metadata.plays;
      }
      
      videoCache.set(video.id.toString(), data);
      setSelectedAnalysis(data as Analysis);
      setError(null);
    } catch (err: any) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to fetch analysis');
    } finally {
      setLoadingAnalysis(false);
    }
  }, []);

  // Memoize the videos table to prevent unnecessary re-renders
  const videosTable = useMemo(() => (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Filename</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Video</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {videos.map((v) => (
            <tr key={v.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{v.filename}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                  ${v.status === 'completed' ? 'bg-green-100 text-green-800' : 
                    v.status === 'error' ? 'bg-red-100 text-red-800' : 
                    v.status === 'processing' ? 'bg-yellow-100 text-yellow-800' : 
                    'bg-gray-100 text-gray-800'}`}>
                  {v.status}
                </span>
                {v.error_message && (
                  <span className="ml-2 text-xs text-red-600">{v.error_message}</span>
                )}
                {v.status === 'processing' && (
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                        style={{ width: `${v.progress || 0}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {v.progress ? `${v.progress.toFixed(1)}% complete` : 'Processing...'}
                    </div>
                  </div>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {v.status === 'completed' ? (
                  <button
                    onClick={() => handleViewAnalysis(v)}
                    className="text-blue-600 hover:text-blue-900"
                    disabled={loadingAnalysis}
                  >
                    {loadingAnalysis ? 'Loading...' : 'View Analysis'}
                  </button>
                ) : (
                  <span className="text-gray-400">
                    {v.status === 'processing' ? 'Processing...' : 'Pending'}
                  </span>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {v.status === 'completed' && (
                  <>
                    <a
                      href={`${API_URL}/uploads/${encodeURIComponent(v.filename)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-green-600 hover:text-green-900 mr-4"
                    >
                      Play
                    </a>
                    <a
                      href={`${API_URL}/uploads/${encodeURIComponent(v.filename)}`}
                      download
                      className="text-gray-600 hover:text-gray-900"
                    >
                      Download
                    </a>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ), [videos, handleViewAnalysis, loadingAnalysis]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {error && (
        <div className="fixed top-0 left-0 right-0 bg-red-100 border-b border-red-400 text-red-700 px-4 py-3" role="alert">
          <div className="max-w-7xl mx-auto">
            <p className="font-bold">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative overflow-hidden"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
              <span className="block">Basketball Play</span>
              <span className="block text-blue-600">Analysis Platform</span>
            </h1>
            <p className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
              Upload your basketball game footage and get AI-powered analysis of plays, player movements, and tactical insights.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Upload Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
      >
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="px-6 py-8">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900">Upload Video</h2>
              <p className="mt-2 text-sm text-gray-500">
                Upload your basketball game footage for analysis
              </p>
            </div>
            <div className="mt-6">
              <div className="flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg">
                <div className="space-y-1 text-center">
                  <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div className="flex text-sm text-gray-600">
                    <label className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                      <span>Upload a file</span>
                      <input type="file" className="sr-only" accept="video/*" onChange={handleFileChange} />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">MP4, MOV, or AVI up to 100MB</p>
                </div>
              </div>
              {file && (
                <div className="mt-4 text-center">
                  <p className="text-sm text-gray-500">Selected: {file.name}</p>
                  {loading && (
                    <div className="mt-4">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                          style={{ width: `${uploadProgress}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-gray-500 mt-2">
                        Uploading: {uploadProgress.toFixed(1)}%
                      </p>
                    </div>
                  )}
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                  >
                    {loading ? 'Uploading...' : 'Start Analysis'}
                  </button>
                </div>
              )}
              {error && (
                <div className="mt-4 text-center text-sm text-red-600">
                  {error}
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Videos Dashboard */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
      >
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="px-6 py-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Videos</h2>
            {videosTable}
          </div>
        </div>
      </motion.div>

      {/* Analysis Results */}
      {loadingAnalysis && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
            <p className="mt-2 text-gray-600">Loading analysis...</p>
          </div>
        </div>
      )}
      
      {selectedAnalysis && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
        >
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div className="px-6 py-8">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
                <button
                  onClick={() => setShowAnalytics(!showAnalytics)}
                  className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                >
                  {showAnalytics ? 'Hide Analytics' : 'Show Analytics'}
                </button>
              </div>

              {showAnalytics && (
                <div className="mb-8">
                  <AnalyticsDashboard 
                    plays={selectedAnalysis.plays || []} 
                    isLoading={loadingAnalysis}
                    error={selectedAnalysis.error}
                  />
                </div>
              )}

              <div className="mb-8">
                <PlaySearch
                  plays={selectedAnalysis.plays || []}
                  onFilterChange={setFilteredPlays}
                />
              </div>

              {selectedAnalysis.plays ? (
                <PlayVisualizer
                  plays={filteredPlays.length > 0 ? filteredPlays : selectedAnalysis.plays}
                  playDiagrams={selectedAnalysis.playDiagrams}
                />
              ) : (
                <pre className="text-xs mt-2 overflow-x-auto bg-gray-50 p-4 rounded-lg">
                  {JSON.stringify(selectedAnalysis, null, 2)}
                </pre>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-gray-50 to-white">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-gray-900 mb-4">404</h1>
        <p className="text-xl text-gray-600 mb-8">Page not found</p>
        <a href="/" className="text-blue-600 hover:text-blue-800">
          Return to Home
        </a>
      </div>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

export default App;
