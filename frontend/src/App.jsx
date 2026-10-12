import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Chat from './pages/Chat';
import KnowledgeBase from './pages/KnowledgeBase';
import bgNeuralNetwork from './assets/bg.neural_networks.gif';

export default function App() {
  return (
    <div className="app-shell">
      <div className="bg-canvas" aria-hidden="true">
        <img src={bgNeuralNetwork} alt="" className="bg-media" />
        <div className="bg-overlay" />
      </div>
      <Navbar />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/knowledge-base" element={<KnowledgeBase />} />
        </Routes>
      </div>
    </div>
  );
}