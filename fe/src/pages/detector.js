import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Detector() {
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [useAI, setUseAI] = useState(false);
  const [mcDropout, setMcDropout] = useState(false);
  const [user, setUser] = useState(null);
  const router = useRouter();

  // Check authentication
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (!token) {
      router.push('/login');
    } else {
      setUser(JSON.parse(userData));
    }
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    router.push('/');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    if (text.trim().length < 10) {
      setError('Text must be at least 10 characters');
      return;
    }

    if (text.length > 5000) {
      setError('Text must not exceed 5000 characters');
      return;
    }

    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API_URL}/predict`,
        {
          text: text.trim(),
          language: language,
          verify_with_ai: useAI,
          mc_dropout: mcDropout,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      console.log('API Response:', response.data);
      setResult(response.data);
    } catch (err) {
      console.error('API Error:', err);
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        router.push('/login');
      } else {
        const message = err.response?.data?.detail || err.message || 'Failed to analyze text';
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center">
      <div className="text-xl">Loading...</div>
    </div>;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 via-blue-600 to-purple-700 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header with User Info */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Fake News Detector</h1>
            <p className="text-blue-100">Welcome, {user?.username}! {user?.role === 'admin' && '(Admin)'}</p>
          </div>
          <div className="flex gap-4">
            {user?.role === 'admin' && (
              <a
                href="/admin"
                className="bg-purple-500 text-white px-4 py-2 rounded-lg hover:bg-purple-600 font-semibold"
              >
                🛡️ Admin
              </a>
            )}
            <a
              href="/history"
              className="bg-white text-blue-600 px-4 py-2 rounded-lg hover:bg-blue-50 font-semibold"
            >
              History
            </a>
            <button
              onClick={handleLogout}
              className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 font-semibold"
            >
              Logout
            </button>
          </div>
        </div>

        {/* Form Card */}
        <div className="bg-white rounded-lg shadow-2xl p-8 mb-6">
          <form onSubmit={handleSubmit}>
            {/* Language Selector */}
            <div className="mb-6">
              <label className="block text-gray-700 font-bold mb-2">Language / Ngôn ngữ</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="en"
                    checked={language === 'en'}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-5 h-5 text-blue-600"
                  />
                  <span className="text-gray-700 font-semibold">En English</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="vi"
                    checked={language === 'vi'}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-5 h-5 text-blue-600"
                  />
                  <span className="text-gray-700 font-semibold">🇻🇳 Tiếng Việt</span>
                </label>
              </div>
              <p className="text-sm text-gray-500 mt-2">
                {language === 'en' 
                  ? 'Using RoBERTa model (your trained model)'
                  : 'Using PhoBERT model (Vietnamese)'}
              </p>
            </div>

            {/* Text Input */}
            <div className="mb-6">
              <label className="block text-gray-700 font-bold mb-2">
                {language === 'en' ? 'News Text' : 'Nội dung tin tức'}
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={language === 'en' 
                  ? 'Paste your news article here...'
                  : 'Dán nội dung tin tức vào đây...'}
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 resize-none"
                rows="6"
              />
              <p className="text-sm text-gray-500 mt-2">
                {text.length}/5000 characters
              </p>
            </div>

            {/* Options */}
            <div className="flex gap-6 mb-6">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useAI}
                  onChange={(e) => setUseAI(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded"
                />
                <span className="text-gray-700">AI Verification (Gemini/Groq)</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={mcDropout}
                  onChange={(e) => setMcDropout(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded"
                />
                <span className="text-gray-700">MC Dropout (Uncertainty)</span>
              </label>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-6 rounded-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="loading-spinner" />
                  {language === 'en' ? 'Analyzing...' : 'Đang phân tích...'}
                </span>
              ) : (
                language === 'en' ? 'Analyze News' : 'Phân tích tin tức'
              )}
            </button>
          </form>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {/* Main Result */}
            <div className={`bg-white rounded-lg shadow-lg p-6 border-l-4 ${
              result.label === 'fake' ? 'border-l-red-500' : 'border-l-green-500'
            }`}>
              <h2 className="text-xl font-bold text-gray-800 mb-4">
                {language === 'en' ? 'Prediction Result' : 'Kết quả dự đoán'}
              </h2>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">
                    {language === 'en' ? 'Verdict' : 'Kết luận'}
                  </p>
                  <p className="text-2xl font-bold uppercase">
                    <span className={`px-3 py-1 rounded-full ${
                      result.label === 'fake'
                        ? 'bg-red-100 text-red-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {result.label === 'fake' 
                        ? (language === 'en' ? 'FAKE' : 'GIẢ')
                        : (language === 'en' ? 'REAL' : 'THẬT')}
                    </span>
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-600 mb-1">
                    {language === 'en' ? 'Confidence' : 'Độ tin cậy'}
                  </p>
                  <p className="text-2xl font-bold">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Probabilities */}
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-2">
                  {language === 'en' ? 'Score Breakdown' : 'Chi tiết điểm số'}
                </p>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>{language === 'en' ? 'Real' : 'Thật'}</span>
                      <span className="font-semibold">{(result.probabilities.real * 100).toFixed(1)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div
                        className="confidence-bar-fill bg-gradient-to-r from-green-500 to-green-600"
                        style={{ width: `${result.probabilities.real * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>{language === 'en' ? 'Fake' : 'Giả'}</span>
                      <span className="font-semibold">{(result.probabilities.fake * 100).toFixed(1)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div
                        className="confidence-bar-fill bg-gradient-to-r from-red-500 to-red-600"
                        style={{ width: `${result.probabilities.fake * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Language Info */}
              <div className="mt-4 pt-4 border-t">
                <p className="text-sm text-gray-500">
                  Model: {result.language === 'en' ? 'RoBERTa (English)' : 'PhoBERT (Vietnamese)'}
                </p>
              </div>
            </div>

            {/* AI Verification Result */}
            {result.openai_result && result.openai_result.is_available && (
              <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-l-yellow-500">
                <h2 className="text-xl font-bold text-gray-800 mb-4">
                  {language === 'en' ? 'AI Verification' : 'Xác minh AI'}
                </h2>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">
                      {language === 'en' ? 'Verdict' : 'Kết luận'}
                    </p>
                    <p className="text-2xl font-bold uppercase">
                      <span className={`px-3 py-1 rounded-full ${
                        result.openai_result.verdict === 'fake'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {result.openai_result.verdict === 'fake'
                          ? (language === 'en' ? 'FAKE' : 'GIẢ')
                          : (language === 'en' ? 'REAL' : 'THẬT')}
                      </span>
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">
                      {language === 'en' ? 'Confidence' : 'Độ tin cậy'}
                    </p>
                    <p className="text-2xl font-bold">
                      {(result.openai_result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="mb-4">
                  <p className="text-sm font-semibold text-gray-700 mb-2">
                    {language === 'en' ? 'Reasoning' : 'Lý do'}
                  </p>
                  <p className="text-gray-600">{result.openai_result.reasoning}</p>
                </div>

                {result.openai_result.concerns && result.openai_result.concerns.length > 0 && (
                  <div>
                    <p className="text-sm font-semibold text-gray-700 mb-2">
                      {language === 'en' ? 'Concerns' : 'Vấn đề'}
                    </p>
                    <ul className="list-disc list-inside space-y-1">
                      {result.openai_result.concerns.map((concern, idx) => (
                        <li key={idx} className="text-gray-600 text-sm">{concern}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Combined Result */}
            {result.combined_result && (
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg shadow-lg p-6 border-2 border-purple-200">
                <h2 className="text-xl font-bold text-gray-800 mb-4">
                  🎯 {language === 'en' ? 'Combined Verdict' : 'Kết luận tổng hợp'}
                </h2>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">
                      {language === 'en' ? 'Final Verdict' : 'Kết luận cuối cùng'}
                    </p>
                    <p className="text-3xl font-bold uppercase">
                      <span className={`px-3 py-1 rounded-full ${
                        result.combined_result.verdict === 'fake'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {result.combined_result.verdict === 'fake'
                          ? (language === 'en' ? 'FAKE' : 'GIẢ')
                          : (language === 'en' ? 'REAL' : 'THẬT')}
                      </span>
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">
                      {language === 'en' ? 'Confidence' : 'Độ tin cậy'}
                    </p>
                    <p className="text-3xl font-bold">
                      {(result.combined_result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-2">
                    {language === 'en' ? 'Model Weights' : 'Trọng số mô hình'}
                  </p>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-700">BERT (60%)</span>
                      <p className="font-semibold text-gray-900">{result.combined_result.bert_verdict}</p>
                    </div>
                    <div>
                      <span className="text-gray-700">AI (40%)</span>
                      <p className="font-semibold text-gray-900">{result.combined_result.openai_verdict}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* New Analysis Button */}
            <button
              onClick={() => {
                setText('');
                setResult(null);
                setError('');
              }}
              className="w-full bg-gray-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-gray-700 transition-all"
            >
              {language === 'en' ? 'New Analysis' : 'Phân tích mới'}
            </button>
          </div>
        )}
      </div>

      <style jsx>{`
        .confidence-bar {
          width: 100%;
          height: 8px;
          background-color: #e5e7eb;
          border-radius: 4px;
          overflow: hidden;
        }
        .confidence-bar-fill {
          height: 100%;
          transition: width 0.3s ease;
        }
        .loading-spinner {
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
