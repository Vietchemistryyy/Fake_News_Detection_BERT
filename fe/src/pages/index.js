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
              🛡️ Fake News Detector
            </h1>
            <p className="text-lg text-gray-600">
              AI-Powered Multi-Language News Verification System
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Powered by BERT, PhoBERT & Advanced AI Models
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
                <h3 className="text-lg font-medium text-gray-900">Dual BERT Models</h3>
                <p className="text-gray-600 text-sm">
                  RoBERTa (English) & PhoBERT (Vietnamese) fine-tuned for 92%+ accuracy
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-purple-500 text-white">
                  <span className="text-xl">🌍</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">Multi-Language</h3>
                <p className="text-gray-600 text-sm">
                  Support for English and Vietnamese news articles
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-green-500 text-white">
                  <span className="text-xl">🔍</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">AI Verification</h3>
                <p className="text-gray-600 text-sm">
                  Optional cross-verification with Gemini, Groq, or OpenAI
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-yellow-500 text-white">
                  <span className="text-xl">📊</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">Query History</h3>
                <p className="text-gray-600 text-sm">
                  Track your analysis history with detailed statistics
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-red-500 text-white">
                  <span className="text-xl">⚡</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">Real-time Analysis</h3>
                <p className="text-gray-600 text-sm">
                  Instant results with GPU-accelerated inference
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white">
                  <span className="text-xl">🛡️</span>
                </div>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900">Admin Dashboard</h3>
                <p className="text-gray-600 text-sm">
                  Comprehensive system monitoring and user management
                </p>
              </div>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="space-y-4">
            <Link href="/login">
              <button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-4 px-6 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all text-lg">
                Login to Start 
              </button>
            </Link>
            <Link href="/register">
              <button className="w-full bg-gradient-to-r from-gray-600 to-gray-700 text-white font-bold py-4 px-6 rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all text-lg">
                Create Account
              </button>
            </Link>
          </div>

          {/* Info */}
          <div className="mt-8 pt-8 border-t border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">How it works</h3>
            <ol className="space-y-3 text-gray-600">
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">1.</span>
                <span>Login or create an account to get started</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">2.</span>
                <span>Select language (English or Vietnamese) and paste your news article</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">3.</span>
                <span>Optionally enable AI verification for cross-checking</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">4.</span>
                <span>Get instant analysis with confidence scores and detailed reasoning</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 font-bold text-blue-600">5.</span>
                <span>View your query history and statistics anytime</span>
              </li>
            </ol>
          </div>

          {/* Tech Stack */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Technology Stack</h3>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 text-xs font-semibold rounded-full">RoBERTa</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">PhoBERT</span>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-semibold rounded-full">FastAPI</span>
              <span className="px-3 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">Next.js</span>
              <span className="px-3 py-1 bg-red-100 text-red-700 text-xs font-semibold rounded-full">MongoDB</span>
              <span className="px-3 py-1 bg-indigo-100 text-indigo-700 text-xs font-semibold rounded-full">PyTorch</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-white">
          <p className="text-sm font-medium">
            Powered by BERT, PhoBERT & Advanced AI Models
          </p>
          <p className="text-xs mt-2 opacity-90">
            © 2025 Nguyen Quoc Viet. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  );
}
