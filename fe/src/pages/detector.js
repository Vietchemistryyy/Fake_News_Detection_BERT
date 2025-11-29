import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Detector() {
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [inputMode, setInputMode] = useState('text'); // 'text' or 'url'
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [useAI, setUseAI] = useState(false);
  const [mcDropout, setMcDropout] = useState(false);
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  // Check authentication
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

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

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    // Basic URL validation
    try {
      new URL(url);
    } catch {
      setError('Please enter a valid URL');
      return;
    }

    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API_URL}/predict-url`,
        null,
        {
          params: {
            url: url.trim(),
            verify_with_ai: useAI,
            mc_dropout: mcDropout,
          },
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      console.log('URL API Response:', response.data);
      setResult(response.data);
    } catch (err) {
      console.error('URL API Error:', err);
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        router.push('/login');
      } else if (err.response?.status === 422) {
        // URL cannot be accessed - suggest manual paste
        const message = err.response?.data?.detail || 'Cannot access this URL';
        setError(`${message}\n\nPlease try switching to "Paste Text" mode and copy the article content manually.`);
      } else {
        const message = err.response?.data?.detail || err.message || 'Failed to analyze URL';
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  if (authLoading || !user) {
    return <div className="min-h-screen flex items-center justify-center">
      <div className="text-xl">Loading...</div>
    </div>;
  }

  return (
    <div>
      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content - Left Side (2/3 width) */}
          <div className="lg:col-span-2">
            {/* Form Card */}
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
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
                      <span className="text-gray-700 font-semibold">Vn Tiếng Việt</span>
                    </label>
                  </div>
                  <p className="text-sm text-gray-500 mt-2">
                    {language === 'en'
                      ? 'Using RoBERTa model'
                      : 'Using PhoBERT model'}
                  </p>
                </div>

                {/* Input Mode Tabs */}
                <div className="mb-6">
                  <div className="flex border-b border-gray-200">
                    <button
                      type="button"
                      onClick={() => { setInputMode('text'); setError(''); }}
                      className={`px-6 py-3 font-semibold transition-colors ${inputMode === 'text'
                        ? 'border-b-2 border-blue-600 text-blue-600'
                        : 'text-gray-600 hover:text-gray-900'
                        }`}
                    >
                      📝 Paste Text
                    </button>
                    <button
                      type="button"
                      onClick={() => { setInputMode('url'); setError(''); }}
                      className={`px-6 py-3 font-semibold transition-colors ${inputMode === 'url'
                        ? 'border-b-2 border-blue-600 text-blue-600'
                        : 'text-gray-600 hover:text-gray-900'
                        }`}
                    >
                      🔗 Enter URL
                    </button>
                  </div>
                </div>

                {/* Text Input Mode */}
                {inputMode === 'text' && (
                  <>
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
                  </>
                )}

                {/* URL Input Mode */}
                {inputMode === 'url' && (
                  <>
                    <div className="mb-6">
                      <label className="block text-gray-700 font-bold mb-2">
                        {language === 'en' ? 'Article URL' : 'URL bài viết'}
                      </label>
                      <input
                        type="url"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder={language === 'en'
                          ? 'https://example.com/news-article'
                          : 'https://vnexpress.net/bai-viet-tin-tuc'}
                        className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                      />
                      <p className="text-sm text-gray-500 mt-2">
                        {language === 'en'
                          ? 'Enter the full URL of the news article'
                          : 'Nhập URL đầy đủ của bài viết tin tức'}
                      </p>

                    </div>
                  </>
                )}

                {/* Text Input */}
                <div className="mb-6" style={{ display: 'none' }}>
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
                    <span className="text-gray-700">Cross-verify with Groq AI</span>
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
                  onClick={inputMode === 'url' ? handleUrlSubmit : handleSubmit}
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-6 rounded-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <div className="loading-spinner" />
                      {inputMode === 'url'
                        ? (language === 'en' ? 'Scraping & Analyzing...' : 'Đang tải & phân tích...')
                        : (language === 'en' ? 'Analyzing...' : 'Đang phân tích...')}
                    </span>
                  ) : (
                    inputMode === 'url'
                      ? (language === 'en' ? '🔗 Analyze URL' : '🔗 Phân tích URL')
                      : (language === 'en' ? '📝 Analyze Text' : '📝 Phân tích văn bản')
                  )}
                </button>
              </form>
            </div>

            {/* Results */}
            {result && (
              <div className="space-y-4">
                {/* Main Result */}
                <div className={`bg-white rounded-lg shadow-lg p-6 border-l-4 ${result.label === 'fake' ? 'border-l-red-500' : 'border-l-green-500'
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
                        <span className={`px-3 py-1 rounded-full ${result.label === 'fake'
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

                {/* Groq AI Cross-Verification */}
                {result.groq_result && result.groq_result.is_available && (
                  <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-lg shadow-lg p-8 border-2 border-orange-200">
                    <h2 className="text-3xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                      ⚡ {language === 'en' ? 'Groq AI Cross-Verification' : 'Xác minh chéo Groq AI'}
                    </h2>

                    <div className="bg-white rounded-lg p-6 border-2 border-orange-300 shadow-md">
                      <div className="flex items-center justify-between mb-6">
                        <div>
                          <span className={`px-6 py-3 rounded-full text-xl font-bold ${result.groq_result.verdict === 'fake'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-green-100 text-green-700'
                            }`}>
                            {result.groq_result.verdict === 'fake'
                              ? (language === 'en' ? 'FAKE' : 'GIẢ')
                              : (language === 'en' ? 'REAL' : 'THẬT')}
                          </span>
                        </div>
                        <div className="text-right">
                          <p className="text-base text-gray-600 mb-1">
                            {language === 'en' ? 'Confidence' : 'Độ tin cậy'}
                          </p>
                          <p className="text-4xl font-bold text-orange-600">
                            {(result.groq_result.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>

                      <div className="border-t-2 pt-5">
                        <p className="text-base font-semibold text-gray-700 mb-3">
                          {language === 'en' ? 'Analysis:' : 'Phân tích:'}
                        </p>
                        <p className="text-base text-gray-700 leading-relaxed">
                          {result.groq_result.reasoning}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Combined Result with Voting */}
                {result.combined_result && (
                  <div className="bg-gradient-to-r from-yellow-50 via-orange-50 to-red-50 rounded-lg shadow-2xl p-6 border-4 border-yellow-300">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                      🎯 {language === 'en' ? 'Final Verdict (Majority Voting)' : 'Kết luận cuối cùng (Bỏ phiếu đa số)'}
                    </h2>

                    {/* Final Verdict and Confidence - 2 columns */}
                    <div className="grid grid-cols-2 gap-6 mb-6">
                      <div className="bg-white rounded-lg p-6 text-center shadow-md">
                        <p className="text-base text-gray-600 mb-3 font-medium">
                          {language === 'en' ? 'Final Verdict' : 'Kết luận'}
                        </p>
                        <p className="text-3xl font-bold uppercase">
                          <span className={`px-6 py-3 rounded-full ${result.combined_result.verdict === 'fake'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-green-100 text-green-700'
                            }`}>
                            {result.combined_result.verdict === 'fake'
                              ? (language === 'en' ? 'FAKE' : 'GIẢ')
                              : (language === 'en' ? 'REAL' : 'THẬT')}
                          </span>
                        </p>
                      </div>

                      <div className="bg-white rounded-lg p-6 text-center shadow-md">
                        <p className="text-base text-gray-600 mb-3 font-medium">
                          {language === 'en' ? 'Confidence' : 'Độ tin cậy'}
                        </p>
                        <p className="text-5xl font-bold text-blue-600">
                          {(result.combined_result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {/* Voting Breakdown */}
                    <div className="bg-white rounded-lg p-5 mb-5 shadow-md">
                      <p className="text-base font-semibold text-gray-700 mb-4">
                        📊 {language === 'en' ? 'Voting Results' : 'Kết quả bỏ phiếu'}
                      </p>
                      <div className="flex items-center gap-6">
                        <div className="flex-1">
                          <div className="flex justify-between text-base mb-2">
                            <span className="text-green-700 font-semibold">REAL</span>
                            <span className="font-bold text-lg">{result.combined_result.real_votes}/{result.combined_result.total_votes}</span>
                          </div>
                          <div className="confidence-bar h-4">
                            <div
                              className="confidence-bar-fill bg-gradient-to-r from-green-500 to-green-600 h-4"
                              style={{ width: `${(result.combined_result.real_votes / result.combined_result.total_votes) * 100}%` }}
                            />
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="flex justify-between text-base mb-2">
                            <span className="text-red-700 font-semibold">FAKE</span>
                            <span className="font-bold text-lg">{result.combined_result.fake_votes}/{result.combined_result.total_votes}</span>
                          </div>
                          <div className="confidence-bar h-4">
                            <div
                              className="confidence-bar-fill bg-gradient-to-r from-red-500 to-red-600 h-4"
                              style={{ width: `${(result.combined_result.fake_votes / result.combined_result.total_votes) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Individual Verdicts - Larger and centered */}
                    {result.combined_result.breakdown && (
                      <div className="bg-white rounded-lg p-5 shadow-md">
                        <p className="text-base font-semibold text-gray-700 mb-4">
                          {language === 'en' ? 'Individual Verdicts' : 'Kết luận từng mô hình'}
                        </p>
                        <div className="flex justify-center gap-8">
                          {result.combined_result.breakdown.bert && (
                            <div className="text-center px-8 py-4 bg-gray-50 rounded-lg">
                              <span className="text-gray-600 text-base block mb-2">🤖 BERT</span>
                              <p className={`font-bold text-2xl ${result.combined_result.breakdown.bert.verdict === 'fake' ? 'text-red-600' : 'text-green-600'}`}>
                                {result.combined_result.breakdown.bert.verdict.toUpperCase()}
                              </p>
                            </div>
                          )}
                          {result.combined_result.breakdown.groq && (
                            <div className="text-center px-8 py-4 bg-gray-50 rounded-lg">
                              <span className="text-gray-600 text-base block mb-2">⚡ Groq</span>
                              <p className={`font-bold text-2xl ${result.combined_result.breakdown.groq.verdict === 'fake' ? 'text-red-600' : 'text-green-600'}`}>
                                {result.combined_result.breakdown.groq.verdict.toUpperCase()}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* New Analysis Button */}
                <button
                  onClick={() => {
                    setText('');
                    setUrl('');
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

          {/* Right Sidebar - Info Panel (1/3 width) */}
          <div className="lg:col-span-1">
            {/* Quick Stats */}
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">📊 Quick Stats</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Model</span>
                  <span className="font-semibold text-blue-600">
                    {language === 'en' ? 'RoBERTa' : 'PhoBERT'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Accuracy</span>
                  <span className="font-semibold text-green-600">92%+</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">AI Verify</span>
                  <span className={`font-semibold ${useAI ? 'text-green-600' : 'text-gray-400'}`}>
                    {useAI ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg shadow-lg p-6 mb-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">💡 Tips</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex gap-2">
                  <span>•</span>
                  <span>Paste complete article for better accuracy</span>
                </li>
                <li className="flex gap-2">
                  <span>•</span>
                  <span>Enable AI cross-verification for higher confidence</span>
                </li>
                <li className="flex gap-2">
                  <span>•</span>
                  <span>Check your history to track past analyses</span>
                </li>
              </ul>
            </div>

            {/* Model Info */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">🤖 Model Info</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="font-semibold text-gray-700 mb-1">English Model</p>
                  <p className="text-gray-600">RoBERTa fine-tuned on news dataset</p>
                </div>
                <div>
                  <p className="font-semibold text-gray-700 mb-1">Vietnamese Model</p>
                  <p className="text-gray-600">PhoBERT optimized for Vietnamese</p>
                </div>
                <div>
                  <p className="font-semibold text-gray-700 mb-1">AI Verification</p>
                  <p className="text-gray-600">Groq AI (Llama 3.1 8B)</p>
                </div>
              </div>
            </div>
          </div>
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
        .line-clamp-3 {
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
      </div>
    </div>
  );
}
