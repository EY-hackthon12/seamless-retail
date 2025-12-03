import React, { useState, useEffect, useRef } from 'react';
import { mobileChat } from '../services/api';
import { Send, ShoppingBag, QrCode as QrIcon, X } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';

interface Message {
    id: number;
    text: string;
    sender: 'user' | 'agent';
}

export default function MobileChat() {
    const [messages, setMessages] = useState<Message[]>([
        { id: 1, text: "Hi! I'm your personal shopping assistant. How can I help you today?", sender: 'agent' }
    ]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState<string | undefined>(undefined);
    const [showQr, setShowQr] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg: Message = { id: Date.now(), text: input, sender: 'user' };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const data = await mobileChat(userMsg.text, sessionId);
            if (data.session_id) {
                setSessionId(data.session_id);
                // Persist session ID for Kiosk demo
                localStorage.setItem('retail_session_id', data.session_id);
            }

            const agentMsg: Message = { id: Date.now() + 1, text: data.response, sender: 'agent' };
            setMessages(prev => [...prev, agentMsg]);
        } catch (error) {
            console.error("Error sending message:", error);
            // Handle error gracefully
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-100px)] max-w-md mx-auto bg-white shadow-2xl rounded-xl overflow-hidden border border-gray-100">
            <div className="bg-primary p-4 text-white flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-accent rounded-full flex items-center justify-center">
                        <ShoppingBag size={18} />
                    </div>
                    <span className="font-semibold">UrbanVogue Assistant</span>
                </div>
                <button
                    onClick={() => sessionId && setShowQr(true)}
                    className={`p-2 rounded-full hover:bg-white/10 transition-colors ${!sessionId ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={!sessionId}
                >
                    <QrIcon size={20} className="text-white" />
                </button>
            </div>

            {showQr && sessionId && (
                <div className="absolute inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-sm">
                    <div className="bg-white p-8 rounded-3xl shadow-2xl w-full max-w-sm text-center relative animate-fade-in">
                        <button onClick={() => setShowQr(false)} className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-full transition-colors">
                            <X size={24} />
                        </button>
                        <h3 className="text-2xl font-bold mb-2">Connect to Kiosk</h3>
                        <p className="text-gray-500 mb-6">Scan this code at the in-store kiosk to sync your session.</p>
                        <div className="flex justify-center p-4 bg-white rounded-xl">
                            <QRCodeSVG value={sessionId} size={200} level="H" includeMargin={true} />
                        </div>
                        <p className="mt-4 text-xs text-gray-400 font-mono">{sessionId}</p>
                    </div>
                </div>
            )}

            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
                {messages.map((msg) => (
                    <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] p-3 rounded-2xl ${msg.sender === 'user'
                            ? 'bg-primary text-white rounded-br-none'
                            : 'bg-white text-gray-800 shadow-sm border border-gray-100 rounded-bl-none'
                            }`}>
                            {msg.text}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-white p-3 rounded-2xl rounded-bl-none shadow-sm border border-gray-100">
                            <div className="flex space-x-1">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSend} className="p-4 bg-white border-t border-gray-100 flex space-x-2">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type a message..."
                    className="flex-1 p-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent transition-all"
                />
                <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="p-3 bg-primary text-white rounded-xl hover:bg-secondary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Send size={20} />
                </button>
            </form>
        </div>
    );
}
