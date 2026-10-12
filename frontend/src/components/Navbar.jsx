import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Bot, MessageSquare, Database } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  const isLanding = location.pathname === '/';

  return (
    <nav className={`navbar ${isLanding ? 'navbar-transparent' : 'navbar-solid'}`}>
      <div className="nav-container">
        <NavLink to="/" className="nav-brand">
          <div className="logo-box">
            <Bot size={20} className="text-white" />
          </div>
          <h2>NexusRAG</h2>
        </NavLink>
        
        <div className="nav-links">
          {!isLanding && (
            <>
              <NavLink to="/chat" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>
                <MessageSquare size={16} /> Chat
              </NavLink>
              <NavLink to="/knowledge-base" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>
                <Database size={16} /> Knowledge Base
              </NavLink>
            </>
          )}
          {isLanding && (
            <>
              <NavLink to="/chat" className="nav-link">App</NavLink>
              <a href="#" className="nav-link">Documentation</a>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
