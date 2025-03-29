"use client";

import { useState, useEffect } from 'react';
import Image from 'next/image';

export default function RaceCountdownCard({ race }) {
  const [timeRemaining, setTimeRemaining] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0
  });
  
  // Se não for fornecida uma corrida como prop, usa esta como padrão
  const nextRace = race || {
    name: "Australian Grand Prix",
    circuit: "Albert Park Circuit",
    location: "Melbourne, Australia",
    flag: "/images/flags/australia.png", // Substitua por caminho real da imagem
    date: "March 30, 2025",
    time: "06:00 GMT",
    dateTime: new Date('2025-03-30T06:00:00Z').getTime()
  };
  
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date().getTime();
      const distance = nextRace.dateTime - now;
      
      if (distance < 0) {
        clearInterval(interval);
        setTimeRemaining({
          days: 0,
          hours: 0,
          minutes: 0,
          seconds: 0
        });
        return;
      }
      
      const days = Math.floor(distance / (1000 * 60 * 60 * 24));
      const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((distance % (1000 * 60)) / 1000);
      
      setTimeRemaining({ days, hours, minutes, seconds });
    }, 1000);
    
    return () => clearInterval(interval);
  }, [nextRace.dateTime]);
  
  return (
    <div className="race-countdown-card">
      <div className="race-countdown-glow"></div>
      <div className="race-countdown-content">
        <div className="race-info">
          <h2 className="race-name">{nextRace.name}</h2>
          
          <div className="race-location">
            {nextRace.flag ? (
              <Image 
                src={nextRace.flag} 
                alt={`${nextRace.location} flag`} 
                width={24} 
                height={16}
                className="race-flag"
                // Fallback em caso de erro de carregamento da imagem
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            ) : null}
            <span>{nextRace.location}</span>
          </div>
          
          <div className="race-circuit">{nextRace.circuit}</div>
          <div className="race-date">{nextRace.date} - {nextRace.time}</div>
        </div>
        
        <h3 className="text-lg font-bold mb-2">Countdown to Race Start</h3>
        
        <div className="countdown-grid">
          <div className="countdown-item">
            <span className="countdown-value">{timeRemaining.days}</span>
            <span className="countdown-label">Days</span>
          </div>
          <div className="countdown-item">
            <span className="countdown-value">{timeRemaining.hours}</span>
            <span className="countdown-label">Hours</span>
          </div>
          <div className="countdown-item">
            <span className="countdown-value">{timeRemaining.minutes}</span>
            <span className="countdown-label">Minutes</span>
          </div>
          <div className="countdown-item">
            <span className="countdown-value">{timeRemaining.seconds}</span>
            <span className="countdown-label">Seconds</span>
          </div>
        </div>
      </div>
    </div>
  );
}