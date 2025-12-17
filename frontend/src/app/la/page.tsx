"use client";
import { useEffect, useRef, useState } from "react";
import { MessageCircle, LifeBuoy, Plus, MessageSquare, Trash2 } from "lucide-react";
import { v4 as uuidv4 } from 'uuid';

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface SavedChat {
  id: string;
  title: string;
  date: number;
  messages: Message[];
}

export default function LuChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chats, setChats] = useState<SavedChat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  // --- Persistence Logic ---

  // 1. Load Chats from LocalStorage on Mount
  useEffect(() => {
    const saved = localStorage.getItem("debature_sr2_chats");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setChats(parsed);
        // Load the most recent chat if available
        if (parsed.length > 0) {
          const last = parsed[0];
          setActiveChatId(last.id);
          setMessages(last.messages);
        } else {
          startNewChat();
        }
      } catch (e) {
        console.error("Failed to parse chats", e);
        startNewChat();
      }
    } else {
      startNewChat();
    }
  }, []);

  // 2. Save Chats to LocalStorage whenever messages or activeChatId changes
  useEffect(() => {
    if (!activeChatId) return;

    setChats(prev => {
      const idx = prev.findIndex(c => c.id === activeChatId);
      if (idx === -1) return prev;

      const updated = [...prev];
      updated[idx] = { ...updated[idx], messages: messages };

      // Update title based on first message if generic
      if (messages.length > 0 && updated[idx].title === "New Chat") {
        updated[idx].title = messages[0].content.slice(0, 30);
      }

      localStorage.setItem("debature_sr2_chats", JSON.stringify(updated));
      return updated;
    });
  }, [messages, activeChatId]);

  // --- Session & WebSocket Logic ---

  const startNewChat = () => {
    const newId = uuidv4();
    const newChat: SavedChat = {
      id: newId,
      title: "New Chat",
      date: Date.now(),
      messages: []
    };

    setChats(prev => [newChat, ...prev]);
    setActiveChatId(newId);
    setMessages([]);

    // Refresh backend session for new context
    fetchSession();
  };

  const deleteChat = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const newChats = chats.filter(c => c.id !== id);
    setChats(newChats);
    localStorage.setItem("debature_sr2_chats", JSON.stringify(newChats));

    if (activeChatId === id) {
      if (newChats.length > 0) {
        loadChat(newChats[0]);
      } else {
        startNewChat();
      }
    }
  };

  const loadChat = (chat: SavedChat) => {
    setActiveChatId(chat.id);
    setMessages(chat.messages);
    // Note: We keep the same backend session or fetch a new one? 
    // Usually backend session is ephemeral. We just reconnect.
    if (!sessionId) fetchSession();
  };

  const fetchSession = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/v1/sessions", { method: "POST" });
      const data = await res.json();
      setSessionId(data.session_id);
    } catch (e) {
      console.error("Backend offline?", e);
    }
  };

  // Ensure we have a backend session on mount
  useEffect(() => {
    if (!sessionId) fetchSession();
  }, []);

  // Connect WebSocket
  useEffect(() => {
    if (!sessionId) return;
    // Close existing
    if (wsRef.current) wsRef.current.close();

    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.type === "message") {
        setIsStreaming(true);
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          // If last message is assistant, append. Otherwise create new.
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

      if (data.type === "end") {
        setIsStreaming(false);
      }
    };

    ws.onclose = () => setIsStreaming(false);

    return () => {
      ws.close();
    };
  }, [sessionId]);

  // Scroll to bottom
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current) return;

    const userText = input;
    // push user message
    setMessages((m) => [...m, { role: "user", content: userText }]);

    // Send to backend
    // Note: To support history awareness, we MIGHT send history here if backend supported it.
    // Currently backend is stateless per turn, so we just send the new message.
    wsRef.current.send(JSON.stringify({ message: userText }));

    setInput("");
  };

  return (
    <div className="flex h-screen bg-[#0D1117] text-[#E6EDF3] font-sans">
      {/* Sidebar */}
      <aside className="hidden md:flex flex-col w-64 bg-[#161B22] border-r border-[#30363D]">
        <div className="p-4 border-b border-[#30363D]">
          <button
            onClick={startNewChat}
            className="w-full flex items-center justify-center gap-2 bg-[#1F6FEB] hover:bg-[#388BFD] text-white py-2 px-4 rounded-md transition-colors font-medium text-sm"
          >
            <Plus className="w-4 h-4" /> New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <h3 className="text-xs font-semibold text-[#8B949E] px-2 py-2 uppercase tracking-wider">History</h3>
          {chats.map(chat => (
            <div
              key={chat.id}
              onClick={() => loadChat(chat)}
              className={`group flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer text-sm transition-colors ${activeChatId === chat.id
                ? "bg-[#30363D] text-[#E6EDF3]"
                : "text-[#8B949E] hover:bg-[#21262D] hover:text-[#C9D1D9]"
                }`}
            >
              <MessageSquare className="w-4 h-4 shrink-0" />
              <span className="truncate flex-1">{chat.title || "Untitled Chat"}</span>
              <Trash2
                className="w-3 h-3 opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                onClick={(e) => deleteChat(e, chat.id)}
              />
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-[#30363D] text-xs text-[#8B949E] text-center">
          SR² History
        </div>
      </aside>

      {/* Main Chat */}
      <main className="flex flex-col flex-1 items-center overflow-hidden relative">
        <div className="w-full max-w-3xl flex flex-col flex-1 relative z-10">
          <header className="px-6 py-4 flex items-center justify-between border-b border-[#30363D] bg-[#0D1117]/80 backdrop-blur-sm z-20">
            <div className="flex items-center gap-2 text-lg font-medium text-[#E6EDF3]">
              <MessageCircle className="w-6 h-6 text-[#1F6FEB]" />
              Debature — SR²
            </div>
            <div className="md:hidden">
              {/* Mobile Menu Toggle could go here */}
            </div>
          </header>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto px-4 md:px-0 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-[#8B949E] opacity-50 select-none">
                <LifeBuoy className="w-16 h-16 mb-4 stroke-1" />
                <p>Start a new conversation with SR²</p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex gap-4 max-w-2xl mx-auto w-full ${msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
              >
                <div
                  className={`relative px-5 py-3.5 rounded-2xl max-w-[90%] text-sm leading-relaxed shadow-sm ${msg.role === "user"
                    ? "bg-[#1F6FEB] text-white rounded-br-none"
                    : "bg-[#21262D] text-[#E6EDF3] border border-[#30363D] rounded-bl-none w-full"
                    }`}
                >
                  <MessageContent content={msg.content} />
                </div>
              </div>
            ))}
            {isStreaming && (
              <div className="flex gap-4 max-w-2xl mx-auto w-full justify-start">
                <div className="bg-[#21262D] px-5 py-3.5 rounded-2xl rounded-bl-none border border-[#30363D] flex gap-1 items-center h-10">
                  <span className="w-2 h-2 bg-[#8B949E] rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                  <span className="w-2 h-2 bg-[#8B949E] rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                  <span className="w-2 h-2 bg-[#8B949E] rounded-full animate-bounce"></span>
                </div>
              </div>
            )}
            <div ref={endRef} className="h-4" />
          </div>

          {/* Input Area */}
          <footer className="p-4 md:p-6 bg-gradient-to-t from-[#0D1117] via-[#0D1117] to-transparent">
            {/* ... input ... */}
            <div className="max-w-2xl mx-auto relative group">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask about customer profiles, inventory, or omnichannel strategies..."
                disabled={isStreaming}
                className="w-full bg-[#161B22] text-[#E6EDF3] rounded-xl pl-5 pr-24 py-4 border border-[#30363D] focus:border-[#58A6FF] focus:ring-1 focus:ring-[#58A6FF] focus:outline-none transition-all shadow-lg placeholder:text-[#484F58]"
              />
              <button
                onClick={sendMessage}
                disabled={isStreaming || !input.trim()}
                className={`absolute right-2 top-2 bottom-2 px-4 rounded-lg font-medium text-sm transition-all ${isStreaming || !input.trim()
                  ? "bg-[#30363D] text-[#8B949E] cursor-not-allowed"
                  : "bg-[#1F6FEB] hover:bg-[#388BFD] text-white shadow-md hover:shadow-lg active:scale-95"
                  }`}
              >
                Send
              </button>
            </div>
            <div className="text-center mt-3 text-xs text-[#484F58]">
              SR² can make mistakes. Verify important retail data.
            </div>
          </footer>
        </div>
      </main>
    </div>
  );
}

// --- Helper Component for Rendering Code Blocks ---
function MessageContent({ content }: { content: string }) {
  // 1. Clean invisible tokens
  const clean = content.replace(/<\|im_end\|>/g, "").trim();

  // 2. Simple Regex to find code blocks: ```lang ... ```
  const parts = clean.split(/(```[\s\S]*?```)/g);

  return (
    <div className="space-y-3">
      {parts.map((part, idx) => {
        if (part.startsWith("```")) {
          // Extract language and code
          const match = part.match(/```(\w*)\n?([\s\S]*?)```/);
          const lang = match ? match[1] : "code";
          const code = match ? match[2] : part.slice(3, -3);

          return (
            <div key={idx} className="rounded-lg overflow-hidden border border-[#30363D] bg-[#0D1117] my-2">
              <div className="flex items-center justify-between px-4 py-2 bg-[#161B22] border-b border-[#30363D]">
                <span className="text-xs font-medium text-[#8B949E] uppercase tracking-wider">
                  {lang || "Snippet"}
                </span>
                <button
                  onClick={() => navigator.clipboard.writeText(code)}
                  className="text-xs text-[#8B949E] hover:text-white transition-colors"
                >
                  Copy
                </button>
              </div>
              <pre className="p-4 overflow-x-auto text-sm font-mono text-[#E6EDF3] bg-[#0D1117]">
                <code>{code}</code>
              </pre>
            </div>
          );
        }
        // Regular Text
        return <div key={idx} className="whitespace-pre-wrap">{part}</div>;
      })}
    </div>
  );
}
