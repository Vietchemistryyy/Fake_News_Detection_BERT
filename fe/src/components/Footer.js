import React from 'react';

export default function Footer() {
    return (
        <footer className="bg-gray-800 text-white py-10">
            <div className="w-full px-8">
                <div className="grid md:grid-cols-3 gap-12 mb-8">
                    <div>
                        <h3 className="text-2xl font-bold mb-4">Fake News Detector</h3>
                        <p className="text-gray-300 text-lg">
                            Advanced fake news detection system using BERT models and machine learning.
                        </p>
                    </div>
                    <div>
                        <h3 className="font-bold text-xl mb-4">Features</h3>
                        <ul className="text-gray-300 space-y-2 text-lg">
                            <li>• Multi-language support</li>
                            <li>• Cross-verification</li>
                            <li>• Query history tracking</li>
                            <li>• Admin dashboard</li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-bold text-xl mb-4">Technology</h3>
                        <ul className="text-gray-300 space-y-2 text-lg">
                            <li>• BERT & PhoBERT models</li>
                            <li>• Groq (Llama 3.1)</li>
                            <li>• FastAPI backend</li>
                            <li>• Next.js frontend</li>
                        </ul>
                    </div>
                </div>
                <div className="border-t border-gray-700 pt-6 text-center text-gray-400 text-lg">
                    <p>© 2025 Nguyen Quoc Viet. All rights reserved.</p>
                    <p className="text-base mt-2">Powered by BERT, PhoBERT & Advanced Models</p>
                </div>
            </div>
        </footer>
    );
}
