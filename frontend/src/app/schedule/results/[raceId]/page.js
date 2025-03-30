"use client";

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useF1Data } from '../../../context/F1DataContext';
import '../../../styles/results.css';

export default function RaceResultsPage() {
  const [raceData, setRaceData] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const params = useParams();
  const router = useRouter();
  const { races } = useF1Data();

  useEffect(() => {
    // Make sure we use the same parameter name consistently
    const raceId = params.raceId;
    console.log("Current raceId:", raceId);
    console.log("Available races:", races);
    
    // Always provide the hardcoded test race for "bahrain-2025"
    const testRace = {
      id: "bahrain-2025",
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

    // Set a small delay to ensure the context is loaded
    setTimeout(() => {
      // Check if we're looking for our test race
      if (raceId === "bahrain-2025") {
        console.log("Setting Bahrain test race data");
        setRaceData(testRace);
      } 
      // Otherwise try to find it in the context
      else if (races && races.length > 0) {
        const contextRace = races.find(r => r.id === raceId);
        if (contextRace) {
          console.log("Found race in context:", contextRace);
          setRaceData(contextRace);
        } else {
          console.log("Race not found in context");
        }
      }
      
      setIsLoaded(true);
    }, 300);
    
  }, [params, races]);

  if (!isLoaded) {
    return (
      <div className="results-container loading-state">
        <div className="loading-spinner"></div>
        <p>Loading race results...</p>
      </div>
    );
  }

  if (!raceData) {
    return (
      <div className="results-container">
        <div className="no-results">
          <h2>Race Results Not Found</h2>
          <p>We couldn't find results for this race.</p>
          <button 
            className="back-button"
            onClick={() => router.push("/schedule")}
          >
            Back to Schedule
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="results-container">
      <motion.div 
        className="results-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <button 
          className="back-button"
          onClick={() => router.push("/schedule")}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
          Back to Schedule
        </button>
      </motion.div>
      <motion.div 
        className="results-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="race-title">{raceData.name} Results</h1>
        <div className="race-meta">
          <p className="race-date">{raceData.date}</p>
          <p className="race-circuit">{raceData.circuit}, {raceData.location}</p>
        </div>
      </motion.div>

      <motion.div 
        className="results-table-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <table className="results-table">
          <thead>
            <tr>
              <th>POS</th>
              <th>DRIVER</th>
              <th>TEAM</th>
              <th>TIME/RETIRED</th>
              <th>PTS</th>
            </tr>
          </thead>
          <tbody>
            {raceData.results?.map((result) => (
              <motion.tr 
                key={result.position}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + result.position * 0.03 }}
                className={result.fastestLap ? "fastest-lap" : ""}
              >
                <td className="position">{result.position}</td>
                <td className="driver">
                  {result.position === 1 && <span className="winner-trophy">🏆</span>}
                  {result.driver}
                  {result.fastestLap && <span className="fastest-lap-indicator" title="Fastest Lap">⚡</span>}
                </td>
                <td className="team">{result.team}</td>
                <td className="time">{result.time}</td>
                <td className="points">{result.points}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </div>
  );
}