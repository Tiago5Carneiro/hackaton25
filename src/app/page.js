"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import './styles/homepage.css';

import NewsCard from './components/NewsCard';
import RaceCountdownCard from './components/RaceCountdown';
import StandingsCard from './components/StandingsCard';
export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false);
  
  useEffect(() => {
    setIsLoaded(true);
  }, []);

  const featuredNews = [
    {
      id: 1,
      title: "F1 2025 Season Preview",
      description: "Get ready for the new season with our comprehensive preview of all teams and drivers.",
      image: "https://placehold.co/600x400?text=F1+Season+Preview",
      date: "March 27, 2025",
      tag: "Preview"
    },
    {
      id: 2,
      title: "New Regulations for 2025",
      description: "Learn about the technical regulations that will transform the upcoming season with more competitive racing.",
      image: "https://placehold.co/600x400?text=New+Regulations+2025",
      date: "March 25, 2025",
      tag: "Technical"
    },
    {
      id: 3,
      title: "Driver Transfers and Rumors",
      description: "Stay updated on the latest driver transfers and rumors that could shape the F1 grid next season.",
      image: "https://placehold.co/600x400?text=Driver+Transfers",
      date: "March 22, 2025",
      tag: "Transfers"
    },
  ];
  
  const driverStandings = [
    { position: 1, name: "Max Verstappen", team: "Red Bull Racing", points: 342, change: "+2" },
    { position: 2, name: "Lando Norris", team: "McLaren", points: 275, change: "0" },
    { position: 3, name: "Charles Leclerc", team: "Ferrari", points: 246, change: "+1" },
  ];

  const constructorStandings = [
    { position: 1, name: "Red Bull Racing", points: 492, change: "0" },
    { position: 2, name: "McLaren", points: 452, change: "+1" },
    { position: 3, name: "Ferrari", points: 431, change: "-1" },
  ];
  
  // Exemplo de informação sobre a próxima corrida
  const nextRace = {
    name: "Australian Grand Prix",
    circuit: "Albert Park Circuit",
    location: "Melbourne, Australia",
    flag: "https://placehold.co/30x20?text=AUS", // Substituir por caminho real da imagem
    date: "March 30, 2025",
    time: "06:00 GMT",
    dateTime: new Date('2025-03-30T06:00:00Z').getTime()
  };
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        staggerChildren: 0.2,
        delayChildren: 0.3
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { 
        type: "spring", 
        stiffness: 100 
      }
    }
  };

  return (
    <div className={`${isLoaded ? 'animate-fade-in' : 'opacity-0'}`}>
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="hero-overlay"></div>
        </div>
        
        <div className="hero-content">
          <h1 className="hero-title">
            F1 HUB
          </h1>
          <p className="hero-subtitle">
            Your ultimate destination for F1 news, schedules, and standings
          </p>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link href="/schedule" className="hero-button">
              View 2025 Calendar
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Next Race Countdown Section */}
      <motion.section 
        className="next-race-container"
        initial={{ opacity: 0, y: 20 }}
        animate={isLoaded ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: 0.2, duration: 0.8 }}
      >
        <RaceCountdownCard race={nextRace} />
      </motion.section>

      {/* Featured News Section */}
      <section className="section-container">
        <div className="section-header">
          <h2 className="section-title">Latest News</h2>
          <Link href="/news" className="view-all-link">
            View All
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </Link>
        </div>

        <motion.div 
          className="news-grid"
          variants={containerVariants}
          initial="hidden"
          animate={isLoaded ? "visible" : "hidden"}
        >
          {featuredNews.map((news) => (
            <motion.div key={news.id} variants={itemVariants}>
              <NewsCard news={news} />
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Standings Section */}
      <section className="section-container">
        <div className="section-header">
          <h2 className="section-title">Championship Standings</h2>
          <Link href="/standings" className="view-all-link">
            Full Standings
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </Link>
        </div>

        <div className="standings-grid">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={isLoaded ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 0.4, duration: 0.6 }}
          >
            <StandingsCard 
              title="Driver Standings" 
              standings={driverStandings} 
              type="driver" 
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={isLoaded ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 0.5, duration: 0.6 }}
          >
            <StandingsCard 
              title="Constructor Standings" 
              standings={constructorStandings} 
              type="constructor" 
            />
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <motion.section 
        className="cta-section"
        initial={{ opacity: 0, y: 40 }}
        animate={isLoaded ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: 0.7, duration: 0.8 }}
      >
        <div className="cta-overlay"></div>
        
        <div className="cta-content">
          <h2 className="cta-title">Try our F1 Pro Analysis Chat</h2>
          <p className="cta-description">
            Ask questions, get insights, and discuss strategies
          </p>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link href="/chat" className="cta-button">
              Chat
            </Link>
          </motion.div>
        </div>
      </motion.section>
    </div>
  );
}