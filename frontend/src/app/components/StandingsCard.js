"use client";

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

export default function StandingsCard({ title, standings, type = 'driver', limit = 3, showMore = true }) {
  // Determina a URL de redirecionamento baseada no tipo
  const standingsUrl = type === 'driver' ? '/standings?tab=drivers' : '/standings?tab=constructors';

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: { 
        delay: i * 0.1,
        duration: 0.5,
        ease: "easeOut" 
      }
    })
  };
  
  // Função para determinar a classe de estilo da mudança de posição
  const getChangeClass = (change) => {
    if (change.startsWith('+')) return 'change-positive';
    if (change.startsWith('-')) return 'change-negative';
    return 'change-neutral';
  };

  return (
    <div className="standings-card">
      <h2 className="standings-title">
        <svg 
          className="standings-title-icon" 
          xmlns="http://www.w3.org/2000/svg" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d={type === 'driver' 
              ? "M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" 
              : "M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"}
          />
        </svg>
        {title}
      </h2>
      
      <ul className="standings-list">
        {standings.slice(0, limit).map((item, index) => (
          <motion.li 
            key={index} 
            className="standings-item"
            custom={index}
            initial="hidden"
            animate="visible"
            variants={itemVariants}
          >
            <div className="standings-position-container">
              <span className={`standings-position position-${item.position <= 3 ? item.position : 'other'}`}>
                {item.position}
              </span>
            </div>
            
            <div className="standings-info">
              <div className="standings-name">{item.name}</div>
              {type === 'driver' && (
                <div className="standings-team">{item.team}</div>
              )}
            </div>
            
            <div className="standings-points-container">
              <span className="standings-points">{item.points}</span>
              <span className={`standings-change ${getChangeClass(item.change)}`}>
                {item.change}
              </span>
            </div>
          </motion.li>
        ))}
      </ul>
      
      {showMore && (
        <div style={{ textAlign: 'center', marginTop: '1rem' }}>
          <Link 
            href={standingsUrl}
            className="card-link"
            style={{ 
              display: 'inline-flex',
              alignItems: 'center', 
              justifyContent: 'center',
              marginTop: '0.5rem',
              padding: '0.5rem 1rem' 
            }}
          >
            View Full Standings
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
              style={{ width: '1rem', height: '1rem', marginLeft: '0.25rem' }}
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M9 5l7 7-7 7" 
              />
            </svg>
          </Link>
        </div>
      )}
    </div>
  );
}