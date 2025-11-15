import React from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 via-blue-600 to-purple-700 flex items-center justify-center py-12 px-4">
      <div className="max-w-2xl w-full">
        {/* Main Card */}
        <div className="bg-white rounded-lg shadow-2xl p-8 md:p-12">
          {/* Logo/Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-2">
              Fake News Detector
            </h1>
            <p className="text-lg text-gray-600">
              Advanced AI-powered news verification system
            </p>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-500 text-white">
                  <span className="text-xl">🤖</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">BERT Model</h3>
                <p className="text-gray-600 text-sm">
                  Fine-tuned RoBERTa for fake news detection with 95%+ accuracy
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-purple-500 text-white">
                  <span className="text-xl">🔍</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">OpenAI Verification</h3>
                <p className="text-gray-600 text-sm">
                  Cross-verify predictions with GPT-3.5-turbo for accuracy
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-green-500 text-white">
                  <span className="text-xl">📊</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">Detailed Analysis</h3>
                <p className="text-gray-600 text-sm">
                  Get confidence scores and reasoning for each verdict
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-yellow-500 text-white">
                  <span className="text-xl">⚡</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">Real-time Processing</h3>
                <p className="text-gray-600 text-sm">
                  Get instant results with GPU-accelerated inference
                </p>
              </div>
            </div>
          </div>

          {/* CTA Button */}
          <Link href="/detector">
            <button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-4 px-6 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all text-lg">
              Start Detection →
            </button>
          </Link>

          {/* Info */}
          <div className="mt-8 pt-8 border-t border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">How it works</h3>
            <ol className="space-y-3 text-gray-600">
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">1.</span>
                <span>Paste your news article text (10-5000 characters)</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">2.</span>
                <span>Choose to enable OpenAI verification (optional)</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">3.</span>
                <span>Get instant analysis with confidence scores</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">4.</span>
                <span>Review detailed reasoning and concerns</span>
              </li>
            </ol>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-white text-sm">
          <p>Powered by BERT + OpenAI | Built with Next.js + FastAPI</p>
        </div>
      </div>
    </div>
  );
}
