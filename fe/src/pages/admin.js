import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function AdminDashboard() {
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [queries, setQueries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('stats');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [userToDelete, setUserToDelete] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [userToEdit, setUserToEdit] = useState(null);
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (!token) {
      router.push('/login');
      return;
    }
    
    const user = JSON.parse(userData);
    if (user.role !== 'admin') {
      router.push('/detector');
      return;
    }
    
    fetchData(token);
  }, [router, activeTab]);

  const fetchData = async (token) => {
    setLoading(true);
    setError('');
    
    try {
      const headers = { Authorization: `Bearer ${token}` };
      
      if (activeTab === 'stats') {
        const response = await axios.get(`${API_URL}/admin/stats`, { headers });
        setStats(response.data);
      } else if (activeTab === 'users') {
        const response = await axios.get(`${API_URL}/admin/users?limit=100`, { headers });
        setUsers(response.data.users);
      } else if (activeTab === 'queries') {
        const response = await axios.get(`${API_URL}/admin/queries?limit=50`, { headers });
        setQueries(response.data.queries);
      }
    } catch (err) {
      if (err.response?.status === 401 || err.response?.status === 403) {
        localStorage.removeItem('token');
        router.push('/login');
      } else {
        setError('Failed to load data');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    router.push('/');
  };

  const handleDeleteUser = async (userId) => {
    const token = localStorage.getItem('token');
    try {
      await axios.delete(`${API_URL}/admin/users/${userId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setShowDeleteModal(false);
      setUserToDelete(null);
      fetchData(token); // Refresh data
    } catch (err) {
      setError('Failed to delete user');
    }
  };

  const handleUpdateUserRole = async (userId, newRole) => {
    const token = localStorage.getItem('token');
    try {
      await axios.put(
        `${API_URL}/admin/users/${userId}`,
        { role: newRole },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      fetchData(token); // Refresh data
    } catch (err) {
      setError('Failed to update user role');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">🛡️ Admin Dashboard</h1>
              <p className="text-sm text-gray-600">System Management & Monitoring</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-semibold"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('stats')}
              className={`px-6 py-3 font-semibold ${
                activeTab === 'stats'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              📊 Statistics
            </button>
            <button
              onClick={() => setActiveTab('users')}
              className={`px-6 py-3 font-semibold ${
                activeTab === 'users'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              👥 Users
            </button>
            <button
              onClick={() => setActiveTab('queries')}
              className={`px-6 py-3 font-semibold ${
                activeTab === 'queries'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              📝 Queries
            </button>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="text-center py-12">
            <div className="text-xl">Loading...</div>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {error}
          </div>
        ) : (
          <>
            {/* Statistics Tab */}
            {activeTab === 'stats' && stats && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Users</h3>
                    <p className="text-3xl font-bold text-blue-600">{stats.total_users}</p>
                  </div>
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Queries</h3>
                    <p className="text-3xl font-bold text-green-600">{stats.total_queries}</p>
                  </div>
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-gray-600 text-sm font-semibold mb-2">Real News</h3>
                    <p className="text-3xl font-bold text-green-600">{stats.by_prediction?.real || 0}</p>
                  </div>
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-gray-600 text-sm font-semibold mb-2">Fake News</h3>
                    <p className="text-3xl font-bold text-red-600">{stats.by_prediction?.fake || 0}</p>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-bold mb-4">Recent Queries</h3>
                  <div className="space-y-3">
                    {stats.recent_queries?.map((query, index) => (
                      <div key={index} className="border-b pb-3">
                        <div className="flex justify-between items-start mb-2">
                          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                            query.prediction.label === 'fake'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-green-100 text-green-700'
                          }`}>
                            {query.prediction.label.toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-500">
                            {new Date(query.timestamp).toLocaleString()}
                          </span>
                        </div>
                        <p className="text-gray-700 text-sm line-clamp-2">{query.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Users Tab */}
            {activeTab === 'users' && (
              <div className="bg-white rounded-lg shadow overflow-hidden">
                <div className="p-4 border-b bg-gray-50">
                  <h3 className="text-lg font-semibold text-gray-900">User Management</h3>
                  <p className="text-sm text-gray-600">Total users: {users.length}</p>
                </div>
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Username</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Login</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {users.map((user) => (
                      <tr key={user._id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {user.username}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {user.email}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <select
                            value={user.role || 'user'}
                            onChange={(e) => handleUpdateUserRole(user._id, e.target.value)}
                            className={`px-2 py-1 text-xs font-semibold rounded-full border-0 ${
                              user.role === 'admin'
                                ? 'bg-purple-100 text-purple-800'
                                : 'bg-gray-100 text-gray-800'
                            }`}
                            disabled={user.username === 'admin'}
                          >
                            <option value="user">User</option>
                            <option value="admin">Admin</option>
                          </select>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <button
                            onClick={() => {
                              setUserToDelete(user);
                              setShowDeleteModal(true);
                            }}
                            disabled={user.username === 'admin'}
                            className={`text-red-600 hover:text-red-900 font-semibold ${
                              user.username === 'admin' ? 'opacity-50 cursor-not-allowed' : ''
                            }`}
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Delete Confirmation Modal */}
            {showDeleteModal && userToDelete && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                  <h3 className="text-lg font-bold text-gray-900 mb-4">Confirm Delete</h3>
                  <p className="text-gray-600 mb-6">
                    Are you sure you want to delete user <strong>{userToDelete.username}</strong>? 
                    This will also delete all their queries. This action cannot be undone.
                  </p>
                  <div className="flex gap-3 justify-end">
                    <button
                      onClick={() => {
                        setShowDeleteModal(false);
                        setUserToDelete(null);
                      }}
                      className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => handleDeleteUser(userToDelete._id)}
                      className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Queries Tab */}
            {activeTab === 'queries' && (
              <div className="bg-white rounded-lg shadow">
                <div className="divide-y">
                  {queries.map((query) => (
                    <div key={query._id} className="p-6">
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-3">
                          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                            query.prediction.label === 'fake'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-green-100 text-green-700'
                          }`}>
                            {query.prediction.label.toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-500">
                            {query.language === 'en' ? 'Eng English' : '🇻🇳 Vietnamese'}
                          </span>
                          <span className="text-sm text-gray-500">
                            Confidence: {(query.prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <span className="text-sm text-gray-500">
                          {new Date(query.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <p className="text-gray-700 line-clamp-2">{query.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
