import React, { useState, useEffect } from 'react';
import { kioskScan } from '../services/api';
import { QrCode, Sparkles, ArrowRight, Camera } from 'lucide-react';
import { Html5QrcodeScanner } from 'html5-qrcode';

export default function KioskDashboard() {
    const [sessionId, setSessionId] = useState<string>('');
    const [scanned, setScanned] = useState(false);
    const [loading, setLoading] = useState(false);
    const [showScanner, setShowScanner] = useState(false);
    const [data, setData] = useState<{ welcome_message: string; recommendations: string } | null>(null);

    useEffect(() => {
        // Simulate getting the session ID from a QR scan (using the one stored by the mobile app)
        const storedSession = localStorage.getItem('retail_session_id');
        if (storedSession) {
            setSessionId(storedSession);
        }
    }, []);

    useEffect(() => {
        if (showScanner && !scanned) {
            const scanner = new Html5QrcodeScanner(
                "reader",
                { fps: 10, qrbox: { width: 250, height: 250 } },
            /* verbose= */ false
            );

            scanner.render(async (decodedText) => {
                console.log(`Scan result: ${decodedText}`);
                scanner.clear();
                setShowScanner(false);
                await processScan(decodedText);
            }, (/* error */) => {
                // console.warn(`Code scan error = ${error}`);
            });

            return () => {
                scanner.clear().catch(error => console.error("Failed to clear scanner", error));
            };
        }
    }, [showScanner, scanned]);

    const processScan = async (sid: string) => {
        setSessionId(sid);
        setLoading(true);
        try {
            const result = await kioskScan(sid);
            setData(result);
            setScanned(true);
        } catch (error) {
            console.error("Scan failed:", error);
            alert("Failed to sync session. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleScan = async () => {
        if (!sessionId) {
            alert("No active mobile session found. Please start a chat in the Mobile App first!");
            return;
        }
        await processScan(sessionId);
    };

    if (scanned && data) {
        return (
            <div className="max-w-4xl mx-auto p-8">
                <div className="bg-white rounded-3xl shadow-2xl overflow-hidden border border-gray-100 animate-fade-in">
                    <div className="bg-gradient-to-r from-primary to-secondary p-12 text-white text-center">
                        <h2 className="text-4xl font-bold mb-4">{data.welcome_message}</h2>
                        <p className="text-xl opacity-90">We've prepared something special for you.</p>
                    </div>

                    <div className="p-12">
                        <div className="flex items-start space-x-6 mb-8">
                            <div className="p-4 bg-blue-50 rounded-2xl text-accent">
                                <Sparkles size={32} />
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold text-gray-800 mb-2">Personalized Recommendation</h3>
                                <p className="text-lg text-gray-600 leading-relaxed">{data.recommendations}</p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
                            <div className="group cursor-pointer">
                                <div className="bg-gray-100 rounded-2xl h-64 mb-4 overflow-hidden relative">
                                    <img src="https://images.unsplash.com/photo-1449505278894-297fdb3edbc1?w=800&q=80" alt="Shoes" className="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-500" />
                                    <div className="absolute bottom-4 right-4 bg-white px-4 py-2 rounded-full shadow-lg font-bold text-primary">$129.00</div>
                                </div>
                                <h4 className="text-xl font-bold">Brown Leather Oxfords</h4>
                                <p className="text-gray-500">Perfect match for your Blue Suit</p>
                            </div>
                            <div className="group cursor-pointer">
                                <div className="bg-gray-100 rounded-2xl h-64 mb-4 overflow-hidden relative">
                                    <img src="https://images.unsplash.com/photo-1594938298603-c8148c472997?w=800&q=80" alt="Suit" className="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-500" />
                                    <div className="absolute bottom-4 right-4 bg-white px-4 py-2 rounded-full shadow-lg font-bold text-primary">$499.00</div>
                                </div>
                                <h4 className="text-xl font-bold">Navy Blue Suit</h4>
                                <p className="text-gray-500">Your current selection</p>
                            </div>
                        </div>

                        <div className="mt-12 flex justify-end">
                            <button className="flex items-center space-x-2 bg-primary text-white px-8 py-4 rounded-xl hover:bg-secondary transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                                <span>Proceed to Fitting Room</span>
                                <ArrowRight size={20} />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col items-center justify-center h-[70vh]">
            <div className="text-center space-y-8 max-w-lg">
                <div className="bg-white p-8 rounded-3xl shadow-xl border border-gray-100">
                    <div className="w-64 h-64 bg-gray-900 mx-auto rounded-2xl flex items-center justify-center mb-6 relative overflow-hidden group">
                        <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-black opacity-90"></div>
                        <QrCode size={120} className="text-white relative z-10 group-hover:scale-110 transition-transform duration-300" />
                        <div className="absolute inset-0 border-4 border-accent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl scale-90"></div>
                    </div>
                    <h2 className="text-3xl font-bold text-gray-800 mb-2">Welcome to the Kiosk</h2>
                    <p className="text-gray-500 mb-8">Scan your QR code to sync your mobile session.</p>

                    {showScanner ? (
                        <div className="mb-6">
                            <div id="reader" className="w-full max-w-sm mx-auto overflow-hidden rounded-xl border-2 border-accent"></div>
                            <button onClick={() => setShowScanner(false)} className="mt-4 text-red-500 hover:underline">Cancel Scan</button>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <button
                                onClick={() => setShowScanner(true)}
                                className="w-full py-4 bg-primary text-white rounded-xl font-bold text-lg hover:bg-secondary transition-all shadow-lg flex items-center justify-center space-x-2"
                            >
                                <Camera size={20} />
                                <span>Scan with Camera</span>
                            </button>

                            <div className="relative flex py-2 items-center">
                                <div className="flex-grow border-t border-gray-200"></div>
                                <span className="flex-shrink mx-4 text-gray-400 text-sm">Or for demo</span>
                                <div className="flex-grow border-t border-gray-200"></div>
                            </div>

                            <button
                                onClick={handleScan}
                                disabled={loading}
                                className="w-full py-4 bg-white border-2 border-gray-200 text-gray-600 rounded-xl font-bold text-lg hover:bg-gray-50 transition-all flex items-center justify-center space-x-2"
                            >
                                {loading ? (
                                    <span>Syncing Profile...</span>
                                ) : (
                                    <>
                                        <QrCode size={20} />
                                        <span>Simulate Scan (Debug)</span>
                                    </>
                                )}
                            </button>
                        </div>
                    )}
                </div>

                {!sessionId && (
                    <p className="text-sm text-red-500 bg-red-50 px-4 py-2 rounded-lg">
                        Debug: No active session found. Please use the Mobile App first.
                    </p>
                )}
            </div>
        </div>
    );
}
