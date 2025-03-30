"use client";
import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import '../styles/schedule.css';
import { useF1Data } from '../context/F1DataContext';
import { useRouter } from 'next/navigation';

export default function SchedulePage() {
  const [selectedRace, setSelectedRace] = useState(null);
  const [filterStatus, setFilterStatus] = useState("all"); // "all", "completed", "upcoming"
  const [isLoaded, setIsLoaded] = useState(false);
  const [localRaces, setLocalRaces] = useState([]);
  
  // Get races from F1Data context
  const { races, loading, error, getEventById } = useF1Data();

  console.log(races);
  
  const router = useRouter();
  
  useEffect(() => {
    setIsLoaded(true);
    
    // Add a hardcoded completed race for testing the results page
    const completedRace = {
      id: "bahrain-2025", // Make sure this ID matches exactly
      name: "Bahrain Grand Prix",
      circuit: "Bahrain International Circuit",
      location: "Sakhir, Bahrain",
      country: "Bahrain",
      flag: "https://flagcdn.com/w320/bh.png",
      date: "March 2, 2025",
      time: "18:00 GMT",
      completed: true,
      winner: "Max Verstappen",
      fastestLap: "Charles Leclerc - 1:32.458",
      image: "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Bahrain_Circuit.png",
      description: "The Bahrain Grand Prix is held at the Bahrain International Circuit and was the season opener for the 2025 Formula 1 World Championship. The race was dominated by Red Bull with Max Verstappen taking the first win of the season.",
      results: [
        { position: 1, driver: "Max Verstappen", team: "Red Bull Racing", time: "1:31:25.896", points: 25 },
        { position: 2, driver: "Lando Norris", team: "McLaren", time: "+5.023s", points: 18 },
        { position: 3, driver: "Charles Leclerc", team: "Ferrari", time: "+12.458s", points: 15 },
        { position: 4, driver: "Oscar Piastri", team: "McLaren", time: "+21.540s", points: 12 },
        { position: 5, driver: "Carlos Sainz", team: "Ferrari", time: "+28.742s", points: 10 },
        { position: 6, driver: "Lewis Hamilton", team: "Mercedes", time: "+34.226s", points: 8 },
        { position: 7, driver: "George Russell", team: "Mercedes", time: "+38.359s", points: 6 },
        { position: 8, driver: "Fernando Alonso", team: "Aston Martin", time: "+45.612s", points: 4 },
        { position: 9, driver: "Sergio Perez", team: "Red Bull Racing", time: "+56.123s", points: 2, fastestLap: true },
        { position: 10, driver: "Lance Stroll", team: "Aston Martin", time: "+62.575s", points: 1 },
        { position: 11, driver: "Alex Albon", team: "Williams", time: "+71.234s", points: 0 },
        { position: 12, driver: "Pierre Gasly", team: "Alpine", time: "+82.423s", points: 0 },
        { position: 13, driver: "Esteban Ocon", team: "Alpine", time: "+83.579s", points: 0 },
        { position: 14, driver: "Kevin Magnussen", team: "Haas", time: "+92.645s", points: 0 },
        { position: 15, driver: "Nico Hulkenberg", team: "Haas", time: "+96.831s", points: 0 },
        { position: 16, driver: "Valtteri Bottas", team: "Kick Sauber", time: "+1 lap", points: 0 },
        { position: 17, driver: "Zhou Guanyu", team: "Kick Sauber", time: "+1 lap", points: 0 },
        { position: 18, driver: "Yuki Tsunoda", team: "VCARB", time: "+1 lap", points: 0 },
        { position: 19, driver: "Daniel Ricciardo", team: "VCARB", time: "+1 lap", points: 0 },
        { position: 20, driver: "Logan Sargeant", team: "Williams", time: "DNF", points: 0 }
      ]
    };
    
    // Always include the test race in local races
    if (races && races.length > 0) {
      // Check if Bahrain race already exists to avoid duplicates
      if (!races.some(race => race.id === "bahrain-2025")) {
        setLocalRaces([completedRace, ...races]);
        console.log("Added test race to races from context");
      } else {
        setLocalRaces(races);
        console.log("Using races from context (includes test race)");
      }
    } else {
      // If no context races, use just the test race
      setLocalRaces([completedRace]);
      console.log("No context races, using test race only");
    }
  }, [races]);
  
  // Update the handleRaceClick function to redirect to results page for completed races
  const handleRaceClick = async (raceId) => {
    try {
      // Get basic race info first
      const basicRace = localRaces.find(r => r.id === raceId);
      
      // If race is completed, redirect to the results page
      if (basicRace && basicRace.completed) {
        router.push(`/schedule/results/${raceId}`);
        return;
      }
      
      // For upcoming races, show details as before
      const raceDetails = await getEventById(raceId);
      if (raceDetails) {
        setSelectedRace(raceDetails);
      } else {
        setSelectedRace(basicRace);
      }
    } catch (error) {
      console.error("Failed to get race details:", error);
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