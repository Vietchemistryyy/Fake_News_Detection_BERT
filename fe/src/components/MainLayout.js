import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

export default function MainLayout({ children, user, onLogout, showAuthButtons = true }) {
  const router = useRouter();

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Top Navigation Bar - Fixed */}
      <nav className="bg-white shadow-md border-b-2 border-blue-500 sticky top-0 z-50">
        <div className="w-full px-8">
          <div className="flex justify-between items-center h-24">
            {/* Left: Title */}
            <Link href="/">
              <div className="cursor-pointer">
                <h1 className="text-3xl font-bold text-gray-900">Fake News Detector</h1>
                <p className="text-lg text-gray-600">AI-Powered News Verification</p>
              </div>
            </Link>

            {/* Right: Auth Buttons or User Menu */}
            <div className="flex gap-5 items-center">
              {user ? (
                <>
                  <span className="text-xl text-gray-700">
                    Welcome, <span className="font-semibold">{user.username}</span>
                    {user.role === 'admin' && (
                      <span className="ml-2 px-3 py-1 bg-yellow-400 text-yellow-900 text-sm rounded">Admin</span>
                    )}
                  </span>
                  {user.role !== 'admin' && (
                    <>
                      <Link href="/detector">
                        <button className="bg-blue-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-blue-700 transition-all text-lg">
                          Detector
                        </button>
                      </Link>
                      <Link href="/history">
                        <button className="bg-green-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-green-700 transition-all text-lg">
                          History
                        </button>
                      </Link>
                    </>
                  )}
                  {user.role === 'admin' && (
                    <Link href="/admin">
                      <button className="bg-purple-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-purple-700 transition-all text-lg">
                        Admin
                      </button>
                    </Link>
                  )}
                  <button
                    onClick={onLogout}
                    className="bg-red-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-red-700 transition-all text-lg"
                  >
                    Logout
                  </button>
                </>
              ) : showAuthButtons && (
                <>
                  <Link href="/login">
                    <button className="bg-blue-600 text-white font-semibold py-4 px-10 rounded-lg hover:bg-blue-700 transition-all shadow-md text-lg">
                      Login to Start →
                    </button>
                  </Link>
                  <Link href="/register">
                    <button className="bg-gray-800 text-white font-semibold py-4 px-10 rounded-lg hover:bg-gray-900 transition-all shadow-md text-lg">
                      Create Account
                    </button>
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer - Fixed */}
      <footer className="bg-gray-800 text-white py-10">
        <div className="w-full px-8">
          <div className="grid md:grid-cols-3 gap-12 mb-8">
            <div>
              <h3 className="text-2xl font-bold mb-4">Fake News Detector</h3>
              <p className="text-gray-300 text-lg">
                AI-powered fake news detection system using advanced BERT models and machine learning.
              </p>
            </div>
            <div>
              <h3 className="font-bold text-xl mb-4">Features</h3>
              <ul className="text-gray-300 space-y-2 text-lg">
                <li>• Multi-language support</li>
                <li>• AI cross-verification</li>
                <li>• Query history tracking</li>
                <li>• Admin dashboard</li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold text-xl mb-4">Technology</h3>
              <ul className="text-gray-300 space-y-2 text-lg">
                <li>• BERT & PhoBERT models</li>
                <li>• Groq AI (Llama 3.1)</li>
                <li>• FastAPI backend</li>
                <li>• Next.js frontend</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-700 pt-6 text-center text-gray-400 text-lg">
            <p>© 2024 Fake News Detector. All rights reserved.</p>
            <p className="text-base mt-2">Powered by BERT, PhoBERT & Advanced AI Models</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
