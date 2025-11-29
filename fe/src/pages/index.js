export default function Home() {
  return (
    <div>
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-100 py-32">
        <div className="w-full px-8">
          <div className="text-left max-w-5xl">
            <h1 className="text-7xl font-bold text-gray-900 mb-6">
              Fake News Detector
            </h1>
            <p className="text-3xl text-gray-700 mb-4">
              Multi-Language News Verification System
            </p>
            <p className="text-2xl text-gray-600">
              Powered by BERT, PhoBERT & Groq AI (Llama 3.1)
            </p>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="w-full px-8 py-24">
        <div className="text-left mb-20">
          <h2 className="text-5xl font-bold text-gray-900 mb-6">Powerful Features</h2>
          <p className="text-2xl text-gray-600">Advanced technology for accurate fake news detection</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-12">
          <div className="bg-white p-10 rounded-xl shadow-lg hover:shadow-2xl transition-shadow border border-gray-200">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Dual BERT Models</h3>
            <p className="text-gray-600 text-xl leading-relaxed">
              RoBERTa (English) & PhoBERT (Vietnamese) fine-tuned for 92%+ accuracy
            </p>
          </div>

          <div className="bg-white p-10 rounded-xl shadow-lg hover:shadow-2xl transition-shadow border border-gray-200">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Multi-Language</h3>
            <p className="text-gray-600 text-xl leading-relaxed">
              Support for English and Vietnamese news articles
            </p>
          </div>

          <div className="bg-white p-10 rounded-xl shadow-lg hover:shadow-2xl transition-shadow border border-gray-200">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Groq AI Cross-Check</h3>
            <p className="text-gray-600 text-xl leading-relaxed">
              Optional second opinion from Groq AI (FREE)
            </p>
          </div>

          <div className="bg-white p-10 rounded-xl shadow-lg hover:shadow-2xl transition-shadow border border-gray-200">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Query History</h3>
            <p className="text-gray-600 text-xl leading-relaxed">
              Track your analysis history with detailed statistics
            </p>
          </div>

          <div className="bg-white p-10 rounded-xl shadow-lg hover:shadow-2xl transition-shadow border border-gray-200">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Real-time Analysis</h3>
            <p className="text-gray-600 text-xl leading-relaxed">
              Instant results with GPU-accelerated inference
            </p>
          </div>

          <div className="bg-white p-10 rounded-xl shadow-lg hover:shadow-2xl transition-shadow border border-gray-200">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Admin Dashboard</h3>
            <p className="text-gray-600 text-xl leading-relaxed">
              Comprehensive system monitoring and user management
            </p>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-lg text-gray-600">Simple steps to verify your news</p>
          </div>

          <div className="grid md:grid-cols-5 gap-8">
            <div className="text-center">
              <div className="flex items-center justify-center h-16 w-16 rounded-full bg-blue-600 text-white mb-4 mx-auto text-2xl font-bold">
                1
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Login</h3>
              <p className="text-gray-600 text-sm">Create account or login</p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center h-16 w-16 rounded-full bg-blue-600 text-white mb-4 mx-auto text-2xl font-bold">
                2
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Select Language</h3>
              <p className="text-gray-600 text-sm">Choose English or Vietnamese</p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center h-16 w-16 rounded-full bg-blue-600 text-white mb-4 mx-auto text-2xl font-bold">
                3
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Paste Article</h3>
              <p className="text-gray-600 text-sm">Enter your news text</p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center h-16 w-16 rounded-full bg-blue-600 text-white mb-4 mx-auto text-2xl font-bold">
                4
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Enable AI</h3>
              <p className="text-gray-600 text-sm">Optional cross-verification</p>
            </div>

            <div className="text-center">
              <div className="flex items-center justify-center h-16 w-16 rounded-full bg-blue-600 text-white mb-4 mx-auto text-2xl font-bold">
                5
              </div>
              <h3 className="font-bold text-gray-900 mb-2">Get Results</h3>
              <p className="text-gray-600 text-sm">Instant analysis with confidence</p>
            </div>
          </div>
        </div>
      </div>

      {/* Tech Stack Section */}
      <div className="bg-gray-100 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Technology Stack</h2>
            <p className="text-lg text-gray-600">Built with cutting-edge technologies</p>
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            <span className="px-6 py-3 bg-blue-100 text-blue-700 text-lg font-semibold rounded-full">RoBERTa</span>
            <span className="px-6 py-3 bg-green-100 text-green-700 text-lg font-semibold rounded-full">PhoBERT</span>
            <span className="px-6 py-3 bg-purple-100 text-purple-700 text-lg font-semibold rounded-full">Groq AI</span>
            <span className="px-6 py-3 bg-yellow-100 text-yellow-700 text-lg font-semibold rounded-full">FastAPI</span>
            <span className="px-6 py-3 bg-red-100 text-red-700 text-lg font-semibold rounded-full">Next.js</span>
            <span className="px-6 py-3 bg-indigo-100 text-indigo-700 text-lg font-semibold rounded-full">MongoDB</span>
            <span className="px-6 py-3 bg-pink-100 text-pink-700 text-lg font-semibold rounded-full">PyTorch</span>
          </div>
        </div>
      </div>
    </div>
  );
}
