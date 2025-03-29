"use client";
import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import '../styles/schedule.css';

export default function SchedulePage() {
  const [selectedRace, setSelectedRace] = useState(null);
  const [filterStatus, setFilterStatus] = useState("all"); // "all", "completed", "upcoming"
  const [isLoaded, setIsLoaded] = useState(false);
  
  useEffect(() => {
    setIsLoaded(true);
  }, []);
  
  const currentDate = new Date();
  
  const races = [
    {
      id: 1,
      name: "Bahrain Grand Prix",
      circuit: "Bahrain International Circuit",
      location: "Sakhir, Bahrain",
      date: "March 2, 2025",
      time: "16:00 GMT",
      completed: true,
      winner: "Max Verstappen",
      fastestLap: "Lewis Hamilton",
      image: "https://placehold.co/600x400",
      description: "The Bahrain Grand Prix is a Formula One championship race which takes place at the Bahrain International Circuit. The 5.412 km circuit was designed by Hermann Tilke, and is located in the middle of the Sakhir desert."
    },
    {
      id: 2,
      name: "Saudi Arabian Grand Prix",
      circuit: "Jeddah Corniche Circuit",
      location: "Jeddah, Saudi Arabia",
      date: "March 16, 2025",
      time: "17:00 GMT",
      completed: true,
      winner: "Charles Leclerc",
      fastestLap: "Max Verstappen",
      image: "https://placehold.co/600x400",
      description: "The Saudi Arabian Grand Prix takes place on the Jeddah Corniche Circuit, which is a street circuit running along the shores of the Red Sea. It is the fastest street circuit on the Formula One calendar, with an average speed of 250 km/h."
    },
    {
      id: 3,
      name: "Australian Grand Prix",
      circuit: "Albert Park Circuit",
      location: "Melbourne, Australia",
      date: "March 30, 2025",
      time: "06:00 GMT",
      completed: false,
      image: "https://placehold.co/600x400",
      description: "The Australian Grand Prix is held at the Albert Park Circuit, a street circuit around Albert Park Lake. The circuit has hosted the Australian Grand Prix since 1996, with the event marking the start of the Formula One season for many years."
    },
    {
      id: 4,
      name: "Japanese Grand Prix",
      circuit: "Suzuka Circuit",
      location: "Suzuka, Japan",
      date: "April 13, 2025",
      time: "05:00 GMT",
      completed: false,
      image: "https://placehold.co/600x400",
      description: "The Japanese Grand Prix is held at the Suzuka Circuit, which is one of the most challenging and popular circuits on the Formula One calendar. The figure-8 layout is unique in Formula One and features a wide variety of corners."
    },
    {
      id: 5,
      name: "Miami Grand Prix",
      circuit: "Miami International Autodrome",
      location: "Miami, United States",
      date: "May 4, 2025",
      time: "19:30 GMT",
      completed: false,
      image: "https://placehold.co/600x400",
      description: "The Miami Grand Prix takes place at the Miami International Autodrome, set around Hard Rock Stadium. The 5.41 km circuit features 19 corners and three DRS zones, with top speeds expected to reach around 320 km/h."
    },
    // Add more races as needed
  ];

  const filteredRaces = filterStatus === "all" 
    ? races 
    : filterStatus === "completed" 
      ? races.filter(race => race.completed)
      : races.filter(race => !race.completed);
  
  // Find the next race (closest upcoming race)
  const nextRace = races
    .filter(race => !race.completed)
    .reduce((closest, race) => {
      if (!closest) return race;
      const closestDate = new Date(closest.date);
      const raceDate = new Date(race.date);
      return raceDate < closestDate ? race : closest;
    }, null);

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: "spring", stiffness: 70, damping: 10 }
    }
  };

  return (
    <div className="schedule-container">
      <motion.div 
        className="schedule-header"
        initial={{ opacity: 0, y: -20 }}
        animate={isLoaded ? { opacity: 1, y: 0 } : { opacity: 0, y: -20 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="page-title">F1 2025 Race Calendar</h1>
        <p className="schedule-subheading">
          Follow the full schedule of the 2025 Formula 1 season. Click on any race for more details.
        </p>
      </motion.div>
      
      {/* Filter buttons */}
      <motion.div 
        className="filter-buttons"
        initial={{ opacity: 0, y: -10 }}
        animate={isLoaded ? { opacity: 1, y: 0 } : { opacity: 0, y: -10 }}
        transition={{ delay: 0.2, duration: 0.4 }}
      >
        <button 
          onClick={() => setFilterStatus("all")}
          className={`filter-button ${filterStatus === "all" ? "active" : ""}`}
        >
          All Races
        </button>
        <button 
          onClick={() => setFilterStatus("completed")}
          className={`filter-button ${filterStatus === "completed" ? "active" : ""}`}
        >
          Completed Races
        </button>
        <button 
          onClick={() => setFilterStatus("upcoming")}
          className={`filter-button ${filterStatus === "upcoming" ? "active" : ""}`}
        >
          Upcoming Races
        </button>
      </motion.div>
      
      {selectedRace ? (
        <motion.div 
          className="race-detail"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
        >
          <button 
            onClick={() => setSelectedRace(null)}
            className="back-button"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M12 19l-7-7 7-7"/>
            </svg>
            Back to schedule
          </button>
          
          <div className="race-detail-header">
            <h2 className="race-detail-name">{selectedRace.name}</h2>
            <div className="race-detail-meta">
              <p>{selectedRace.date} • {selectedRace.time}</p>
              <p>{selectedRace.circuit}, {selectedRace.location}</p>
            </div>
          </div>
          
          <div className="race-image">
            <img 
              src={selectedRace.image} 
              alt={selectedRace.name}
            />
          </div>
          
          <div className="race-detail-content">
            <div className="race-detail-section">
              <h3 className="section-title">About the race</h3>
              <p className="race-description">{selectedRace.description}</p>
            </div>
            
            {selectedRace.completed && (
              <div className="results-container">
                <h3 className="section-title">Race Results</h3>
                <div className="results-grid">
                  <div className="result-item">
                    <span className="result-label">Winner</span>
                    <span className="result-value">{selectedRace.winner}</span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Fastest Lap</span>
                    <span className="result-value">{selectedRace.fastestLap}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      ) : (
        <motion.div 
          className="race-grid"
          variants={containerVariants}
          initial="hidden"
          animate={isLoaded ? "visible" : "hidden"}
        >
          {filteredRaces.map(race => {
            const isNextRace = nextRace && race.id === nextRace.id;
            const cardClassName = `race-card ${race.completed ? 'race-card-completed' : ''} ${isNextRace ? 'race-card-next' : ''}`;
            
            return (
              <motion.div 
                key={race.id}
                variants={itemVariants}
                onClick={() => setSelectedRace(race)}
                className={cardClassName}
              >
                <div className="race-content">
                  <div className="race-header">
                    <h2 className="race-name">{race.name}</h2>
                    {race.completed ? (
                      <span className="race-status status-completed">Completed</span>
                    ) : isNextRace ? (
                      <span className="race-status status-next">Next Race</span>
                    ) : (
                      <span className="race-status status-upcoming">Upcoming</span>
                    )}
                  </div>
                  
                  <p className="race-date">{race.date} • {race.time}</p>
                  <p className="race-circuit">{race.circuit}, {race.location}</p>
                  
                  {race.completed && (
                    <div className="race-results">
                      <div className="winner-info">
                        <span className="trophy-icon">🏆</span>
                        <span className="winner-name">{race.winner}</span>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      )}
    </div>
  );
}