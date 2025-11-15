import React, { useState } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Detector() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [useOpenAI, setUseOpenAI] = useState(false);
  const [mcDropout, setMcDropout] = useState(false);

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
      const response = await axios.post(`${API_URL}/predict`, {
        text: text.trim(),
        verify_with_openai: useOpenAI,
        mc_dropout: mcDropout,
      });

      setResult(response.data);
    } catch (err) {
      const message = err.response?.data?.detail || err.message || 'Failed to analyze text';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const getVerdictColor = (label) => {
    return label === 'fake' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'bg-gradient-to-r from-red-500 to-red-600';
    if (confidence > 0.6) return 'bg-gradient-to-r from-orange-500 to-orange-600';
    return 'bg-gradient-to-r from-green-500 to-green-600';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence > 0.8) return 'Very High';
    if (confidence > 0.6) return 'High';
    if (confidence > 0.4) return 'Moderate';
    return 'Low';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 via-blue-600 to-purple-700 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Fake News Detector</h1>
          <p className="text-blue-100">Analyze news articles using BERT + OpenAI verification</p>
        </div>

        {/* Form Card */}
        <div className="bg-white rounded-lg shadow-2xl p-8 mb-6">
          <form onSubmit={handleSubmit}>
            {/* Text Input */}
            <div className="mb-6">
              <label className="block text-gray-700 font-bold mb-2">News Text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your news article here..."
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
                  checked={useOpenAI}
                  onChange={(e) => setUseOpenAI(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded"
                />
                <span className="text-gray-700">OpenAI Verification</span>
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
                  Analyzing...
                </span>
              ) : (
                'Analyze News'
              )}
            </button>
          </form>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {/* BERT Result */}
            <div className={`bg-white rounded-lg shadow-lg p-6 border-l-4 ${
              result.label === 'fake' ? 'border-l-red-500' : 'border-l-green-500'
            }`}>
              <h2 className="text-xl font-bold text-gray-800 mb-4">BERT Model Result</h2>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Verdict</p>
                  <p className="text-2xl font-bold uppercase">
                    <span className={`px-3 py-1 rounded-full ${
                      result.label === 'fake'
                        ? 'bg-red-100 text-red-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {result.label}
                    </span>
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-600 mb-1">Confidence</p>
                  <p className="text-2xl font-bold">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-600 mb-2">Confidence Level</p>
                <div className="confidence-bar">
                  <div
                    className={`confidence-bar-fill ${getConfidenceColor(result.confidence)}`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">{getConfidenceLabel(result.confidence)}</p>
              </div>

              {/* Probabilities */}
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-2">Score Breakdown</p>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Real</span>
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
                      <span>Fake</span>
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
            </div>

            {/* OpenAI Result */}
            {result.openai_result && result.openai_result.is_available && (
              <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-l-yellow-500">
                <h2 className="text-xl font-bold text-gray-800 mb-4">OpenAI Verification</h2>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Verdict</p>
                    <p className="text-2xl font-bold uppercase">
                      <span className={`px-3 py-1 rounded-full ${
                        result.openai_result.verdict === 'fake'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {result.openai_result.verdict}
                      </span>
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Confidence</p>
                    <p className="text-2xl font-bold">
                      {(result.openai_result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="mb-4">
                  <p className="text-sm font-semibold text-gray-700 mb-2">Reasoning</p>
                  <p className="text-gray-600">{result.openai_result.reasoning}</p>
                </div>

                {result.openai_result.concerns && result.openai_result.concerns.length > 0 && (
                  <div>
                    <p className="text-sm font-semibold text-gray-700 mb-2">Concerns</p>
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
                <h2 className="text-xl font-bold text-gray-800 mb-4">🎯 Combined Verdict</h2>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Final Verdict</p>
                    <p className="text-3xl font-bold uppercase">
                      <span className={`px-3 py-1 rounded-full ${
                        result.combined_result.verdict === 'fake'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {result.combined_result.verdict}
                      </span>
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Confidence</p>
                    <p className="text-3xl font-bold">
                      {(result.combined_result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-2">Model Weights</p>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-700">BERT (60%)</span>
                      <p className="font-semibold text-gray-900">{result.combined_result.bert_verdict}</p>
                    </div>
                    <div>
                      <span className="text-gray-700">OpenAI (40%)</span>
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
              New Analysis
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
