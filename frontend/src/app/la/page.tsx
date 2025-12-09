"use client";
import { useEffect, useRef, useState } from "react";
import { MessageCircle, LifeBuoy } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function LuChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  // 1. Create chat session on mount
  useEffect(() => {
    (async () => {
      const res = await fetch("http://localhost:8000/api/v1/sessions", {
        method: "POST",
      });
      const data = await res.json();
      setSessionId(data.session_id);
    })();
  }, []);

  // 2. Connect WebSocket once we have sessionId
  useEffect(() => {
    if (!sessionId) return;
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onmessage = async (e) => {
      const data = JSON.parse(e.data);

      // Streamed tokens
      if (data.type === "message") {
        setIsStreaming(true);
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant") {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...last,
              content: last.content + data.content,
            };
            return updated;
          } else {
            return [...prev, { role: "assistant", content: data.content }];
          }
        });
      }

      // Mark end of streaming
      if (data.type === "end") {
        setIsStreaming(false);
      }
    };

    ws.onclose = () => setIsStreaming(false);
    ws.onerror = () => setIsStreaming(false);

    return () => ws.close();
  }, [sessionId]);

  // Scroll to bottom when new messages
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current) return;

    // push user message
    setMessages((m) => [...m, { role: "user", content: input }]);
    wsRef.current.send(JSON.stringify({ message: input }));
    setInput("");
  };

  return (
    <div className="flex h-screen bg-[#0D1117] text-[#E6EDF3]">
      {/* Sidebar */}
      <aside className="hidden md:flex flex-col justify-between w-72 bg-[#161B22] border-r border-[#21262D] p-6">
        <div>
          <h1 className="text-2xl font-semibold text-[#1F6FEB] mb-2">LA²</h1>
          <p className="text-sm text-[#8B949E] leading-relaxed">
            Your personal AI Loan Sales Assistant 💬 <br />
            LA² explains loan terms, runs eligibility checks, and even prepares sanction letters — 
            in simple natural conversation.
          </p>
        </div>
        <button
          className="flex items-center justify-center gap-2 bg-[#1F6FEB]/10 hover:bg-[#1F6FEB]/20 text-[#58A6FF] border border-[#1F6FEB]/40 px-4 py-2 rounded-lg mt-6 transition-all duration-200"
          onClick={() => alert("Support requested!")}
        >
          <LifeBuoy className="w-4 h-4" />
          Support
        </button>
      </aside>

      {/* Main Chat */}
      <main className="flex flex-col flex-1 items-center overflow-hidden">
        <div className="w-full max-w-3xl flex flex-col flex-1">
          <header className="px-6 rounded-4xl mt-2 py-4 border-b border-[#21262D] bg-[#161B22] text-lg font-medium flex items-center gap-2">
            <MessageCircle className="w-5 h-5 text-[#1F6FEB]" />
            LA² — Loan Assistant
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 scrollbar-thin scrollbar-thumb-[#1F6FEB]/30 scrollbar-track-transparent">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`max-w-xl px-4 py-2 rounded-2xl break-words whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-[#30363D] self-end ml-auto"
                    : "bg-[#1F6FEB] text-white self-start"
                }`}
              >
                {msg.content}
                {i === messages.length - 1 && msg.role === "assistant" && isStreaming && (
                  <span className="animate-pulse ml-1">▌</span>
                )}
              </div>
            ))}
            <div ref={endRef} />
          </div>

          {/* Input */}
          <footer className="border-t rounded-3xl mb-10 md:mb-4 border-[#21262D] bg-[#161B22] p-4 flex gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Ask LA² something..."
              className="flex-1 bg-[#1E2630] text-[#E6EDF3] rounded-lg px-4 py-2 focus:outline-none"
            />
            <button
              onClick={sendMessage}
              disabled={isStreaming}
              className={`px-4 py-2 rounded-lg text-white transition-all ${
                isStreaming
                  ? "bg-[#1F6FEB]/40 cursor-not-allowed"
                  : "bg-[#1F6FEB] hover:bg-[#388BFD]"
              }`}
            >
              Send
            </button>
          </footer>
        </div>
      </main>
    </div>
  );
}
