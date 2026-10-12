import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FileText, Globe, UploadCloud, Link as LinkIcon, AlertCircle, Database } from 'lucide-react';

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const SOURCE_COLORS = {
  pdf: { color: "#ef4444", icon: <FileText size={16}/> },
  wikipedia: { color: "#3b82f6", icon: <Globe size={16}/> },
  csv: { color: "#10b981", icon: <Database size={16}/> },
  website: { color: "#8b5cf6", icon: <LinkIcon size={16}/> },
};

export default function KnowledgeBase() {
  const [sources, setSources] = useState([]);
  const [stats, setStats] = useState(null);
  const [urlInput, setUrlInput] = useState("");
  const [uploadMsg, setUploadMsg] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    fetchSources();
    fetchStats();
  }, []);

  async function fetchSources() {
    try {
      const res = await fetch(`${API}/sources`);
      const data = await res.json();
      setSources(data.sources || []);
    } catch (e) {
      console.error(e);
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch(`${API}/health`);
      setStats(await res.json());
    } catch (e) {
      console.error(e);
    }
  }

  async function handleUploadPDF(e) {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadMsg(`Uploading ${file.name}...`);
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/upload/pdf`, { method: "POST", body: form });
      const data = await res.json();
      setUploadMsg(`o. ${data.filename} added - ${data.chunks} chunks`);
      fetchSources();
      fetchStats();
    } catch {
      setUploadMsg("?O Upload failed. Check if API is running.");
    }
    setIsUploading(false);
  }

  async function handleUploadURL() {
    if (!urlInput.trim()) return;
    setIsUploading(true);
    setUploadMsg(`Scraping ${urlInput}...`);

    try {
      const res = await fetch(`${API}/upload/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: urlInput }),
      });
      const data = await res.json();
      setUploadMsg(`o. "${data.title}" added - ${data.chunks} chunks`);
      setUrlInput("");
      fetchSources();
      fetchStats();
    } catch {
      setUploadMsg("?O URL ingestion failed.");
    }
    setIsUploading(false);
  }

  return (
    <div className="kb-container">
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="kb-sidebar"
      >
        <div className="kb-header">
          <h2>Indexed Sources</h2>
          {stats && <span className="badge">{stats.total_chunks} chunks</span>}
        </div>
        
        <div className="source-list">
          {sources.length === 0 ? (
            <p className="text-muted text-center mt-8">No sources found.</p>
          ) : (
            sources.map((src, i) => {
              const s = SOURCE_COLORS[src.source_type] || SOURCE_COLORS.pdf;
              return (
                <div key={i} className="source-card">
                  <div className="source-icon" style={{ backgroundColor: s.color + "22", color: s.color }}>
                    {s.icon}
                  </div>
                  <div className="source-info">
                    <h4>{src.file_name.replace(/^(Wikipedia: |CSV: |Web: )/, "")}</h4>
                    <span className="source-type">{src.source_type}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="kb-main"
      >
        <h2>Add New Data</h2>
        <p className="text-muted mb-8">Ingest new files or websites into your vector database instantly.</p>

        <div className="upload-grid">
          <div className="upload-box">
            <UploadCloud size={32} className="text-accent mb-4" />
            <h3>Upload PDF Document</h3>
            <p>Extract text and layout from PDF files to query later.</p>
            <label className="btn-primary mt-4 cursor-pointer">
              {isUploading ? 'Uploading...' : 'Choose File'}
              <input type="file" accept=".pdf" onChange={handleUploadPDF} hidden disabled={isUploading} />
            </label>
          </div>

          <div className="upload-box">
            <Globe size={32} className="text-accent mb-4" />
            <h3>Scrape Website URL</h3>
            <p>Provide a public link to extract article text.</p>
            <div className="url-input-group mt-4">
              <input 
                type="text" 
                placeholder="https://..." 
                value={urlInput}
                onChange={e => setUrlInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleUploadURL()}
                disabled={isUploading}
              />
              <button className="btn-primary" onClick={handleUploadURL} disabled={isUploading || !urlInput}>
                Add
              </button>
            </div>
          </div>
        </div>

        {uploadMsg && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }} 
            animate={{ opacity: 1, y: 0 }}
            className="upload-alert"
          >
            <AlertCircle size={18} /> {uploadMsg}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
