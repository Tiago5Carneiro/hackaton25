"use client";
import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import '../styles/schedule.css';
import { useF1Data } from '../context/F1DataContext';

export default function SchedulePage() {
  const [selectedRace, setSelectedRace] = useState(null);
  const [filterStatus, setFilterStatus] = useState("all"); // "all", "completed", "upcoming"
  const [isLoaded, setIsLoaded] = useState(false);
  const [localRaces, setLocalRaces] = useState([]);
  
  // Get races from F1Data context
  const { races, loading, error, getEventById } = useF1Data();
  
  useEffect(() => {
    setIsLoaded(true);
    
    if (races && races.length > 0) {
      setLocalRaces(races);
      console.log("Loaded races from context:", races.length);
    }
  }, [races]);
  
  // Handle clicking on a race to show details
  const handleRaceClick = async (raceId) => {
    try {
      // Try to get detailed race information
      const raceDetails = await getEventById(raceId);
      if (raceDetails) {
        setSelectedRace(raceDetails);
      } else {
        // If API call fails, use basic race data
        const basicRace = localRaces.find(r => r.id === raceId);
        setSelectedRace(basicRace);
      }
    } catch (error) {
      console.error("Failed to get race details:", error);
      // Fallback to basic race data
      const basicRace = localRaces.find(r => r.id === raceId);
      setSelectedRace(basicRace);
    }
  };
  
  const currentDate = new Date();

  // Filter races based on selected filter
  const filteredRaces = localRaces.length > 0 ? (
    filterStatus === "all" 
      ? localRaces 
      : filterStatus === "completed" 
        ? localRaces.filter(race => race.completed)
        : localRaces.filter(race => !race.completed)
  ) : [];
  
  // Find the next race (closest upcoming race)
  const nextRace = localRaces
    .filter(race => !race.completed)
    .reduce((closest, race) => {
      if (!closest) return race;
      const closestDate = new Date(closest.startDate || closest.date);
      const raceDate = new Date(race.startDate || race.date);
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

  // Show loading state if API is still loading
  if (loading && !localRaces.length) {
    return (
      <div className="schedule-container loading-state">
        <div className="loading-spinner"></div>
        <p>Loading race calendar...</p>
      </div>
    );
  }

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
              {selectedRace.flag && (
                <div className="country-flag">
                  <img src={selectedRace.flag} alt={selectedRace.country} width={30} height={20} />
                  <span>{selectedRace.country}</span>
                </div>
              )}
            </div>
          </div>
          
          <div className="race-image">
            <img 
              src={selectedRace.image || "https://placehold.co/600x400?text=Circuit+Map"}
              alt={selectedRace.name}
            />
          </div>
          
          <div className="race-detail-content">
            <div className="race-detail-section">
              <h3 className="section-title">About the race</h3>
              <p className="race-description">{selectedRace.description || `The ${selectedRace.name} takes place at the ${selectedRace.circuit} in ${selectedRace.location}.`}</p>
            </div>
            
            {selectedRace.completed && (
              <div className="results-container">
                <h3 className="section-title">Race Results</h3>
                <div className="results-grid">
                  <div className="result-item">
                    <span className="result-label">Winner</span>
                    <span className="result-value">{selectedRace.winner || "TBD"}</span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Fastest Lap</span>
                    <span className="result-value">{selectedRace.fastestLap || "N/A"}</span>
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
          {filteredRaces.length > 0 ? (
            filteredRaces.map(race => {
              const isNextRace = nextRace && race.id === nextRace.id;
              const cardClassName = `race-card ${race.completed ? 'race-card-completed' : ''} ${isNextRace ? 'race-card-next' : ''}`;
              
              return (
                <motion.div 
                  key={race.id}
                  variants={itemVariants}
                  onClick={() => handleRaceClick(race.id)}
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
                    
                    {race.flag && (
                      <div className="race-flag">
                        <img src={race.flag} alt={race.country} width={25} height={15} />
                      </div>
                    )}
                    
                    {race.completed && race.winner && (
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
            })
          ) : (
            <div className="no-races-message">
              <p>No races found matching the current filter.</p>
              {error && <p className="error-message">Error: {error}</p>}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}