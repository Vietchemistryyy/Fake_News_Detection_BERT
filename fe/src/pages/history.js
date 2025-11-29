import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function History() {
  const [queries, setQueries] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [user, setUser] = useState(null);
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');

    if (!token) {
      router.push('/login');
    } else {
      setUser(JSON.parse(userData));
      fetchHistory(token);
      fetchStats(token);
    }
  }, [router]);

  const fetchHistory = async (token) => {
    try {
      const response = await axios.get(`${API_URL}/history?limit=50`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setQueries(response.data.queries);
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        router.push('/login');
      } else {
        setError('Failed to load history');
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async (token) => {
    try {
      const response = await axios.get(`${API_URL}/history/stats`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setStats(response.data);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    router.push('/');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Query History</h1>
            <p className="text-gray-600">View your past predictions</p>
          </div>
          <div className="flex gap-4">
            {/* Navigation handled by global header */}
          </div>
        </div>

        {/* Statistics */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Queries</h3>
              <p className="text-3xl font-bold text-blue-600">{stats.total_queries}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-gray-600 text-sm font-semibold mb-2">By Language</h3>
              <div className="space-y-1">
                <p className="text-sm">Eng English: {stats.by_language?.en || 0}</p>
                <p className="text-sm">🇻🇳 Vietnamese: {stats.by_language?.vi || 0}</p>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-gray-600 text-sm font-semibold mb-2">By Prediction</h3>
              <div className="space-y-1">
                <p className="text-sm text-green-600">✓ Real: {stats.by_prediction?.real || 0}</p>
                <p className="text-sm text-red-600">✗ Fake: {stats.by_prediction?.fake || 0}</p>
              </div>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Query List */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <h2 className="text-xl font-bold text-gray-900">Recent Queries</h2>
          </div>

          {queries.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <p>No queries yet. Start by analyzing some news!</p>
              <a
                href="/detector"
                className="inline-block mt-4 text-blue-600 hover:text-blue-700 font-semibold"
              >
                Go to Detector →
              </a>
            </div>
          ) : (
            <div className="divide-y">
              {queries.map((query, index) => (
                <div key={query._id || index} className="p-6 hover:bg-gray-50">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`px-3 py-1 rounded-full text-sm font-semibold ${(query.prediction.combined_result?.verdict || query.prediction.label) === 'fake'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-green-100 text-green-700'
                          }`}>
                          {(query.prediction.combined_result?.verdict || query.prediction.label).toUpperCase()}
                        </span>
                        {query.prediction.combined_result && (
                          <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded">
                            🎯 Voting Result
                          </span>
                        )}
                        <span className="text-sm text-gray-500">
                          {query.language === 'en' ? 'Eng English' : '🇻🇳 Vietnamese'}
                        </span>
                        <span className="text-sm text-gray-500">
                          {new Date(query.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <p className="text-gray-700 line-clamp-2">{query.text}</p>
                    </div>
                    <div className="ml-4 text-right">
                      <p className="text-2xl font-bold text-gray-900">
                        {((query.prediction.combined_result?.confidence || query.prediction.confidence) * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500">confidence</p>
                    </div>
                  </div>

                  {/* Probabilities */}
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600">Real</span>
                        <span className="font-semibold">{(query.prediction.probabilities.real * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500"
                          style={{ width: `${query.prediction.probabilities.real * 100}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600">Fake</span>
                        <span className="font-semibold">{(query.prediction.probabilities.fake * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-red-500"
                          style={{ width: `${query.prediction.probabilities.fake * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
