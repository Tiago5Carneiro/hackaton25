"use client";

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion'; // Certifique-se de importar motion

export default function RaceCountdownCard({ race }) {
  const [timeRemaining, setTimeRemaining] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0
  });
  
  const [isLoaded, setIsLoaded] = useState(false);
  
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
    // Definir isLoaded como true após um pequeno delay para permitir a animação
    const loadTimer = setTimeout(() => {
      setIsLoaded(true);
    }, 300);
    
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
    
    return () => {
      clearInterval(interval);
      clearTimeout(loadTimer);
    };
  }, [nextRace.dateTime]);
  
  // Variants para animação de contagem
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3
      }
    }
  };
  
  const itemVariants = {
    hidden: { 
      opacity: 0, 
      y: 20, 
      scale: 0.8 
    },
    visible: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 10
      }
    }
  };
  
  // Variants para animação dos números
  const numberVariants = {
    hidden: { 
      opacity: 0,
      scale: 0.5
    },
    visible: { 
      opacity: 1,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 200,
        damping: 10
      }
    },
    update: (i) => ({
      scale: [1, 1.2, 1],
      transition: {
        duration: 0.3
      }
    })
  };
  
  return (
    <div className="race-countdown-card">
      <div className="race-countdown-glow"></div>
      <div className="race-countdown-content">
        <motion.div 
          className="race-info"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.h2 
            className="race-name"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {nextRace.name}
          </motion.h2>
          
          <motion.div 
            className="race-location"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            {nextRace.flag ? (
              <Image 
                src={nextRace.flag} 
                alt={`${nextRace.location} flag`} 
                width={24} 
                height={16}
                className="race-flag"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            ) : null}
            <span>{nextRace.location}</span>
          </motion.div>
          
          <motion.div 
            className="race-circuit"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            {nextRace.circuit}
          </motion.div>
          
          <motion.div 
            className="race-date"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            {nextRace.date} - {nextRace.time}
          </motion.div>
        </motion.div>
        
        <motion.h3 
          className="text-lg font-bold mb-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          Countdown to Race Start
        </motion.h3>
        
        <motion.div 
          className="countdown-grid"
          variants={containerVariants}
          initial="hidden"
          animate={isLoaded ? "visible" : "hidden"}
        >
          <motion.div 
            className="countdown-item"
            variants={itemVariants}
          >
            <motion.span 
              className="countdown-value"
              variants={numberVariants}
              key={`days-${timeRemaining.days}`}
              animate={isLoaded ? "visible" : "hidden"}
              custom={1}
            >
              {timeRemaining.days}
            </motion.span>
            <span className="countdown-label">Days</span>
          </motion.div>
          
          <motion.div 
            className="countdown-item"
            variants={itemVariants}
          >
            <motion.span 
              className="countdown-value"
              variants={numberVariants}
              key={`hours-${timeRemaining.hours}`}
              animate={isLoaded ? "visible" : "hidden"}
              custom={2}
            >
              {timeRemaining.hours}
            </motion.span>
            <span className="countdown-label">Hours</span>
          </motion.div>
          
          <motion.div 
            className="countdown-item"
            variants={itemVariants}
          >
            <motion.span 
              className="countdown-value"
              variants={numberVariants}
              key={`minutes-${timeRemaining.minutes}`}
              animate={isLoaded ? "visible" : "hidden"}
              custom={3}
            >
              {timeRemaining.minutes}
            </motion.span>
            <span className="countdown-label">Minutes</span>
          </motion.div>
          
          <motion.div 
            className="countdown-item"
            variants={itemVariants}
          >
            <motion.span 
              className="countdown-value"
              variants={numberVariants}
              key={`seconds-${timeRemaining.seconds}`}
              animate={isLoaded ? "visible" : "hidden"}
              custom={4}
              // Adiciona uma animação suave a cada segundo
              style={{ display: 'inline-block' }}
            >
              {timeRemaining.seconds}
            </motion.span>
            <span className="countdown-label">Seconds</span>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}