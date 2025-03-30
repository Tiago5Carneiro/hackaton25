"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';
import './styles/homepage.css';
import { useF1Data } from './context/F1DataContext';

import NewsCard from './components/NewsCard';
import RaceCountdownCard from './components/RaceCountdown';
import StandingsCard from './components/StandingsCard';

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false);
  
  // Get data from context with try/catch
  let contextData = { drivers: [], constructors: [], nextRace: null, loading: true, error: null };
  try {
    contextData = useF1Data();
  } catch (error) {
    console.error("Failed to get F1 data from context:", error);
  }
  
  const { drivers, constructors, nextRace, loading, error, getNextRace } = contextData;
  
  useEffect(() => {
    console.log("HomePage: Component mounted");
    
    // If next race is null, try to fetch it specifically
    if (!nextRace && getNextRace) {
      console.log("HomePage: Fetching next race specifically");
      getNextRace().then(raceData => {
        console.log("HomePage: Next race fetched:", raceData);
      });
    }
    
    setIsLoaded(true);
    
    // Detailed logging of context data
    console.log('HomePage: Drivers data received:', drivers?.length);
    console.log('HomePage: Constructors data received:', constructors?.length);
    console.log('HomePage: Next race data received:', nextRace);
    console.log('HomePage: Loading state:', loading);
    console.log('HomePage: Error state:', error);
    
  }, [drivers, constructors, nextRace, loading, error, getNextRace]);

  // Transform data for display with safer handling
  const driverStandings = Array.isArray(drivers) && drivers.length > 0 
    ? drivers.slice(0, 3).map(driver => ({
        position: driver.position || '?',
        name: driver.name || 'Unknown Driver',
        team: driver.team || 'Unknown Team',
        points: driver.points || 0,
        change: "+0"
      }))
    : [
        { position: 1, name: "Max Verstappen", team: "Red Bull Racing", points: 240, change: "+0" },
        { position: 2, name: "Lando Norris", team: "McLaren", points: 189, change: "+0" },
        { position: 3, name: "Charles Leclerc", team: "Ferrari", points: 162, change: "+0" }
      ];

  const constructorStandings = Array.isArray(constructors) && constructors.length > 0 
    ? constructors.slice(0, 3).map(team => ({
        position: team.position || '?',
        name: team.name || 'Unknown Team',
        points: team.points || 0,
        change: "0"
      }))
    : [
        { position: 1, name: "Red Bull Racing", points: 400, change: "0" },
        { position: 2, name: "McLaren", points: 310, change: "0" },
        { position: 3, name: "Ferrari", points: 296, change: "0" }
      ];

  // Use default news data
  const featuredNews = [
    {
      id: 1,
      title: "F1 2025 Season Preview",
      description: "Get ready for the new season with our comprehensive preview of all teams and drivers.",
      image: "https://picsum.photos/1000/600?random=7",
      date: "March 27, 2025",
      tag: "Preview"
    },
    {
      id: 2,
      title: "New Regulations for 2025",
      description: "Learn about the technical regulations that will transform the upcoming season with more competitive racing.",
      image: "https://picsum.photos/1000/600?random=7",
      date: "March 25, 2025",
      tag: "Technical"
    },
    {
      id: 3,
      title: "Driver Transfers and Rumors",
      description: "Stay updated on the latest driver transfers and rumors that could shape the F1 grid next season.",
      image: "https://picsum.photos/1000/600?random=7",
      date: "March 22, 2025",
      tag: "Transfers"
    },
  ];
  
  // Handle next race data with fallback
  const defaultRace = {
    name: "Australian Grand Prix",
    circuit: "Albert Park Circuit",
    location: "Melbourne, Australia",
    flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/australia-flag.png",
    date: "March 30, 2025",
    time: "06:00 GMT",
    dateTime: new Date('2025-03-30T06:00:00Z').getTime()
  };
  
  // Animation configurations
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

  // Show loading state if context is still loading
  if (loading) {
    console.log("HomePage: Displaying loading spinner");
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading F1 Hub data...</p>
      </div>
    );
  }

  return (
    <div className={`${isLoaded ? 'animate-fade-in' : 'opacity-0'}`}>
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="hero-overlay"></div>
        </div>
        
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="hero-logo-icon">
              <Image 
                src="/images/f1_logo.png"
                alt="F1 Hub Logo" 
                width={90}
                height={90}
                className="hero-favicon"
              />
            </span>
            Hub
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
        className="next-race-container racing-section"
        initial={{ opacity: 0, y: 20 }}
        animate={isLoaded ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: 0.2, duration: 0.8 }}
      >
        {/* Use the actual nextRace from context if available, otherwise use default */}
        <RaceCountdownCard race={nextRace || defaultRace} />
      </motion.section>

      {/* Featured News Section */}
      <section className="section-container">
        <div className="section-header">
         <h2 className="section-title">Latest News</h2>
          <Link href="/news" className="view-all-link">
            All News
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

      {/* CTA Section with real background image */}
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
            Get expert insights, race predictions, and technical analysis from our AI-powered F1 assistant.
          </p>
          <motion.div 
            className="cta-button-container"
            whileHover={{ scale: 1.05 }} 
            whileTap={{ scale: 0.95 }}
          >
            <Link href="/chat" className="cta-button">
              Start Chatting
            </Link>
          </motion.div>
        </div>
      </motion.section>
    
      
      {/* Error display */}
      {error && (
        <div className="error-notification">
          <p>There was an error loading some content. Please try refreshing.</p>
          <button onClick={() => window.location.reload()}>Refresh</button>
        </div>
      )}
    </div>
  );
}