import React from 'react';
import { useRouter } from 'next/router';

export default function Layout({ children, user, onLogout }) {
  const router = useRouter();
  const currentPath = router.pathname;

  const navigation = [
    ...(user?.role !== 'admin' ? [
      { name: 'Detector', href: '/detector', icon: '🔍' },
      { name: 'History', href: '/history', icon: '📊' },
    ] : []),
    ...(user?.role === 'admin' ? [{ name: 'Admin', href: '/admin', icon: '⚙️' }] : []),
  ];

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Top Navigation Bar */}
      <nav className="bg-gradient-to-r from-blue-600 to-purple-700 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="text-3xl">🛡️</div>
              <div>
                <h1 className="text-xl font-bold">Fake News Detector</h1>
                <p className="text-xs text-blue-100">AI-Powered News Verification</p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="text-sm">
                <span className="text-blue-100">Welcome, </span>
                <span className="font-semibold">{user?.username}</span>
                {user?.role === 'admin' && (
                  <span className="ml-2 px-2 py-1 bg-yellow-400 text-yellow-900 text-xs rounded">Admin</span>
                )}
              </div>
              <button
                onClick={onLogout}
                className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors text-sm font-medium"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <div className="flex-1 flex">
        {/* Sidebar */}
        <aside className="w-64 bg-white shadow-lg">
          <nav className="p-4 space-y-2">
            {navigation.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${currentPath === item.href
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'text-gray-700 hover:bg-gray-100'
                  }`}
              >
                <span className="text-2xl">{item.icon}</span>
                <span className="font-medium">{item.name}</span>
              </a>
            ))}
          </nav>

          {/* Sidebar Info */}
          <div className="p-4 mt-8 border-t">
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 mb-2">System Info</h3>
              <div className="space-y-2 text-sm text-blue-700">
                <div className="flex items-center gap-2">
                  <span>🤖</span>
                  <span>RoBERTa + PhoBERT</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>⚡</span>
                  <span>Groq AI Enabled</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>🌍</span>
                  <span>Multi-Language</span>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-bold text-lg mb-3">About</h3>
              <p className="text-gray-300 text-sm">
                AI-powered fake news detection system using BERT models and advanced machine learning.
              </p>
            </div>
            <div>
              <h3 className="font-bold text-lg mb-3">Technology</h3>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• RoBERTa (English)</li>
                <li>• PhoBERT (Vietnamese)</li>
                <li>• Groq AI (Llama 3.1)</li>
                <li>• FastAPI + Next.js</li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold text-lg mb-3">Contact</h3>
              <p className="text-gray-300 text-sm">
                For support or inquiries, please contact your system administrator.
              </p>
            </div>
          </div>
          <div className="mt-6 pt-6 border-t border-gray-700 text-center text-sm text-gray-400">
            © 2024 Fake News Detector. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
