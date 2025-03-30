"use client";
import { useState, useEffect } from 'react';
import '../../styles/drivers.css';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { useF1Data } from '../../context/F1DataContext';

// Import Recharts components with dynamic loading
const RechartsComponent = dynamic(() => import('../../components/RechartsComponent'), {
  ssr: false,
  loading: () => <div>Loading Chart...</div>
});

export default function DriverPage() {
  const { slug } = useParams();
  const [driver, setDriver] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Get data from F1DataContext
  const { drivers, loading: contextLoading, error: contextError } = useF1Data();
  
  // API URL - change to your actual backend URL
  const API_URL = "http://localhost:5000/api";
  
  // Recent results - mock data
  const recentResults = [
    { race: "Italian GP", position: 3, points: 15 },
    { race: "Dutch GP", position: 1, points: 25 },
    { race: "Belgian GP", position: 2, points: 18 },
    { race: "Hungarian GP", position: 5, points: 10 },
    { race: "British GP", position: 4, points: 12 },
  ];

  // Function to render individual rating bars
  function renderRatingBar(name, value) {
    let ratingClass = "poor";
    let ratingText = "Poor";
    
    if (value >= 90) {
      ratingClass = "excellent";
      ratingText = "Excellent";
    } else if (value >= 80) {
      ratingClass = "good";
      ratingText = "Good";
    } else if (value >= 70) {
      ratingClass = "average";
      ratingText = "Average";
    }
    
    return (
      <div className="rating-item" key={name}>
        <div className="rating-header">
          <span className="rating-label">{name}</span>
          <span className="rating-value">{value}</span>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${ratingClass}`}
            style={{ width: `${value}%` }}
          ></div>
        </div>
        <div className="rating-category">{ratingText}</div>
      </div>
    );
  }

  // Function to render overall rating
  function renderOverallRating() {
    if (!driver) return null;
    
    // Calculate average rating from all skills
    const driverSkills = {
      pace: driver.pace || 70,
      aggression: driver.agress || 70,
      defense: driver.def || 70,
      tireManagement: driver.tireman || 70,
      consistency: driver.consist || 70,
      qualifying: driver.quali || 70,
    };
    
    const totalSkills = Object.values(driverSkills).reduce((sum, value) => sum + value, 0);
    const overallRating = driver.overall || Math.round(totalSkills / Object.keys(driverSkills).length);
    
    // Determine color based on rating
    let ratingColor;
    let ratingDesc;
    
    if (overallRating >= 90) {
      ratingColor = '#22c55e'; // green
      ratingDesc = 'WORLD CLASS';
    } else if (overallRating >= 85) {
      ratingColor = '#84cc16'; // lime
      ratingDesc = 'ELITE';
    } else if (overallRating >= 80) {
      ratingColor = '#84cc16'; // lime
      ratingDesc = 'EXCELLENT';
    } else if (overallRating >= 75) {
      ratingColor = '#f59e0b'; // yellow
      ratingDesc = 'GREAT';
    } else if (overallRating >= 70) {
      ratingColor = '#f59e0b'; // yellow
      ratingDesc = 'GOOD';
    } else {
      ratingColor = '#ef4444'; // red
      ratingDesc = 'AVERAGE';
    }
    
    // Calculate percentage for the circle
    const ratingPercent = `${overallRating}%`;
    
    return (
      <>
        <div 
          className="rating-circle"
          style={{
            '--rating-percent': ratingPercent,
            '--rating-color': ratingColor
          }}
        >
          <div className="rating-circle-inner">
            <div className="overall-value">{overallRating}</div>
            <div className="overall-label">OVERALL</div>
          </div>
        </div>
        <div 
          className="rating-description"
          style={{ '--rating-color': ratingColor }}
        >
          {ratingDesc}
        </div>
      </>
    );
  }

  useEffect(() => {
    const fetchDriverData = async () => {
      // If drivers data is already loaded in context, use it instead
      if (drivers && drivers.length > 0 && !contextLoading) {
        const driverFromContext = drivers.find(d => d.id === slug || d.slug === slug);
        
        if (driverFromContext) {
          try {
            
            // Enhance driver data with additional info
            const enhancedDriver = {
              ...driverFromContext,
              position: drivers.findIndex(d => d.id === driverFromContext.id) + 1,
              dob: getRandomBirthday(), // This would come from API in a real app
              teamLogoUrl: getTeamLogoUrl(driverFromContext.team),
            };
            
            setDriver(enhancedDriver);
            setLoading(false);
            return;
          } catch (err) {
            console.error('Error fetching driver details:', err);
          }
        }
      }
      
      // If not found in context or context is loading, fetch directly
      setLoading(true);
      try {
        // Fetch driver directly by slug
        const driverResponse = await fetch(`${API_URL}/driver/slug/${slug}`);
        if (!driverResponse.ok) {
          throw new Error('Failed to fetch driver details');
        }
        
        const driverData = await driverResponse.json();
        
        // Use drivers from context if available, otherwise fetch them
        let driversData = drivers;
        if (!driversData || driversData.length === 0) {
          const driversResponse = await fetch(`${API_URL}/standings/drivers`);
          if (!driversResponse.ok) {
            throw new Error('Failed to fetch drivers data for standings');
          }
          driversData = await driversResponse.json();
        }
        
        const driverPosition = driversData.findIndex(d => d.id === driverData.id) + 1;
        
        // Enhance driver data with additional info
        const enhancedDriver = {
          ...driverData,
          position: driverPosition,
          dob: getRandomBirthday(), // This would come from API in a real app
          teamLogoUrl: getTeamLogoUrl(driverData.team),
        };
        
        setDriver(enhancedDriver);
        setError(null);
      } catch (err) {
        console.error('Error fetching driver data:', err);
        setError(`Failed to load driver data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    fetchDriverData();
  }, [slug, drivers, contextLoading]);
  
  // Format skills data for Recharts
  const getSkillsData = () => {
    if (!driver) return [];
    
    return [
      { skill: "Pace", value: driver.pace || 70, fullMark: 100 },
      { skill: "Aggression", value: driver.agress || 70, fullMark: 100 },
      { skill: "Defense", value: driver.def || 70, fullMark: 100 },
      { skill: "Tire Management", value: driver.tireman || 70, fullMark: 100 },
      { skill: "Consistency", value: driver.consist || 70, fullMark: 100 },
      { skill: "Qualifying", value: driver.quali || 70, fullMark: 100 },
    ];
  };
  
  // Show loading state if either local or context loading is true
  if (loading || contextLoading) {
    return <div className="driver-page loading">Loading driver data...</div>;
  }
  
  // Show error if either local or context error exists
  if ((error || contextError) || !driver) {
    return (
      <div className="driver-page error">
        <h1>{error || contextError || 'Driver not found'}</h1>
        <Link href="/standings" className="back-link">
          Return to Standings
        </Link>
      </div>
    );
  }
  
  return (
    <div className="driver-page">
      <Link href="/standings" className="back-link">
        ← Back to Standings
      </Link>
      
      <div className="driver-header">
        <div className="driver-info">
          <h1 className="driver-name">{driver.name}</h1>
          <div className="driver-position">
            <span className="position-number">#{driver.position}</span> in {new Date().getFullYear()} Championship
          </div>
          <div className="driver-nationality">
            {driver.flag && (
              <img 
                src={driver.flag} 
                alt="Driver nationality flag" 
                className="nationality-flag"
              />
            )}
            <span className="nationality-text">{driver.nationality || 'Unknown'}</span>
          </div>
          <div className="team-info">
            <span className="team-name">{driver.team}</span>
          </div>
          <div className="driver-basic-info">
            <div className="info-item">
              <span className="label">Points:</span> 
              <span className="value">{driver.points}</span>
            </div>
          </div>
        </div>
        
        <div className="driver-image">
          {driver.image ? (
            <img 
              src={driver.image} 
              alt={driver.name} 
              className="driver-portrait"
            />
          ) : (
            <div className="image-placeholder">
              {driver.name.split(' ').map(n => n[0]).join('')}
            </div>
          )}
        </div>
      </div>
      
      <div className="driver-stats-container">
        <div className="driver-stats">
          <h2>Career Statistics</h2>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{driver.wins || 0}</div>
              <div className="stat-label">Race Wins</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{driver.podiums || 0}</div>
              <div className="stat-label">Podiums</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{driver.championships || 0}</div>
              <div className="stat-label">Championships</div>
            </div>
          </div>
        </div>
        
        <div className="driver-skills">
          <h2>Driver Skills</h2>
          
          {/* Radar chart on top */}
          <div className="radar-chart-container">
            <RechartsComponent driver={driver} skillsData={getSkillsData()} />
          </div>
          
          {/* Skills section with side-by-side layout */}
          <div className="skills-layout">
            {/* Football Manager style ratings on the left */}
            <div className="football-manager-ratings">
              <div className="rating-bars">
                {renderRatingBar("Pace", driver.pace || 70)}
                {renderRatingBar("Aggression", driver.agress || 70)}
                {renderRatingBar("Defense", driver.def || 70)}
                {renderRatingBar("Tire Management", driver.tireman || 70)}
                {renderRatingBar("Consistency", driver.consist || 70)}
                {renderRatingBar("Qualifying", driver.quali || 70)}
              </div>
            </div>
            
            {/* Overall rating on the right */}
            <div className="overall-rating">
              {renderOverallRating("Overall Rating", driver.overall || 70)}
            </div>
          </div>
        </div>
      </div>
          
      <div className="recent-results">
        <h2>Recent Results</h2>
        <div className="results-table-container">
          <table className="results-table">
            <thead>
              <tr>
                <th>Race</th>
                <th>Position</th>
                <th>Points</th>
              </tr>
            </thead>
            <tbody>
              {recentResults.map((result, index) => (
                <tr key={index}>
                  <td>{result.race}</td>
                  <td>{result.position}</td>
                  <td>{result.points}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="compare-section">
        <Link href={`/driver_comparison?driver1=${slug}`} className="compare-button">
          Compare With Another Driver
        </Link>
      </div>
    </div>
  );
}

// Helper function to get team logo URL
function getTeamLogoUrl(team) {
  if (!team) return "";
  
  // Replace spaces with hyphens and convert to lowercase for URL
  const teamSlug = team.toLowerCase().replace(/\s+/g, '-');
  return `https://www.formula1.com/content/dam/fom-website/teams/2023/${teamSlug}.png.transform/2col/image.png`;
}

// Helper function to generate a random birthday (just for demo purposes)
function getRandomBirthday() {
  const year = Math.floor(Math.random() * 15) + 1985; // Random year between 1985-2000
  const month = Math.floor(Math.random() * 12) + 1;
  const day = Math.floor(Math.random() * 28) + 1; // Keeping it simple, no month-specific day counts
  
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];
  
  return `${day} ${months[month-1]} ${year}`;
}