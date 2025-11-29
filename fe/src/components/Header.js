import React from 'react';
import Link from 'next/link';
import { useAuth } from '../contexts/AuthContext';

export default function Header() {
    const { user, logout } = useAuth();

    return (
        <nav className="bg-white shadow-md border-b-2 border-blue-500 sticky top-0 z-50">
            <div className="w-full px-8">
                <div className="flex justify-between items-center h-24">
                    {/* Left: Title */}
                    <Link href="/">
                        <div className="cursor-pointer">
                            <h1 className="text-3xl font-bold text-gray-900">Fake News Detector</h1>
                            <p className="text-lg text-gray-600">Multi-Language News Verification</p>
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
                                {user.role === 'admin' && (
                                    <Link href="/admin">
                                        <button className="bg-purple-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-purple-700 transition-all text-lg">
                                            Admin
                                        </button>
                                    </Link>
                                )}
                                <button
                                    onClick={logout}
                                    className="bg-red-600 text-white font-semibold py-3 px-8 rounded-lg hover:bg-red-700 transition-all text-lg"
                                >
                                    Logout
                                </button>
                            </>
                        ) : (
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
    );
}
