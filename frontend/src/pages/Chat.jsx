import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Loader2, Info } from 'lucide-react';

const API = "http://127.0.0.1:8000";
const SUGGESTIONS = [
  "What is retrieval augmented generation?",
  "How do vector databases work?",
  "What are the main challenges of LLMs?"
];

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function handleAsk(text = question) {
    if (!text.trim()) return;
    const q = text.trim();
    setQuestion("");
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: data.answer, 
        sources: data.sources 
      }]);
    } catch {
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: "Sorry, I am unable to reach the knowledge base right now. Please check your backend connection.", 
        sources: [] 
      }]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="chat-layout">
      <div className="chat-scroll-area">
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="chat-welcome"
            >
              <div className="welcome-icon"><Bot size={40} /></div>
              <h2>How can I help you today?</h2>
              <p>Ask a question, and I'll search through your indexed knowledge base to find the answer.</p>
              
              <div className="suggestion-grid">
                {SUGGESTIONS.map(s => (
                  <button key={s} className="suggestion-card" onClick={() => handleAsk(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </motion.div>
          </div>
        ) : (
          <div className="chat-history">
            <AnimatePresence initial={false}>
              {messages.map((msg, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`chat-row ${msg.role === 'ai' ? 'ai-row' : 'user-row'}`}
                >
                  <div className="chat-row-content">
                    <div className={`avatar-box ${msg.role === 'ai' ? 'ai-avatar' : 'user-avatar'}`}>
                      {msg.role === 'ai' ? <Bot size={20} /> : <User size={20} />}
                    </div>
                    <div className="message-body">
                      <div className="message-text">{msg.content}</div>
                      
                      {msg.role === 'ai' && msg.sources?.length > 0 && (
                        <div className="citations-box">
                          <div className="citations-header">
                            <Info size={14} /> Sources
                          </div>
                          <div className="citations-list">
                            {msg.sources.map((src, j) => (
                              <span key={j} className="cite-badge">
                                {src.file_name.replace(/^(Wikipedia: |CSV: |Web: )/, "")} 
                                <span className="cite-score">{(src.relevance * 100).toFixed(0)}%</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
              
              {isLoading && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} 
                  className="chat-row ai-row"
                >
                  <div className="chat-row-content">
                    <div className="avatar-box ai-avatar"><Bot size={20} /></div>
                    <div className="message-body">
                      <div className="loading-indicator">
                        <Loader2 size={18} className="animate-spin text-accent" /> Searching knowledge base...
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={bottomRef} className="scroll-anchor" />
          </div>
        )}
      </div>

      <div className="chat-input-wrapper">
        <div className="chat-input-container">
          <input
            type="text"
            className="chat-input"
            placeholder="Message NexusRAG..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleAsk()}
            disabled={isLoading}
          />
          <button 
            className={`send-button ${question.trim() ? 'active' : ''}`}
            onClick={() => handleAsk()} 
            disabled={isLoading || !question.trim()}
          >
            <Send size={18} />
          </button>
        </div>
        <div className="chat-footer-text">
          NexusRAG can make mistakes. Consider verifying important information with the provided citations.
        </div>
      </div>
    </div>
  );
}
