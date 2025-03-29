"use client";
import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { useRouter, useSearchParams } from 'next/navigation';
import { usePathname } from 'next/navigation';
import '../styles/standings.css';

export default function Standings() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  
  // Using the URL parameter to initialize the tab state ensures consistency
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState(tabParam === 'constructors' ? 'constructors' : 'drivers');
  const [isLoaded, setIsLoaded] = useState(false);
  
  // Track if we're coming back from a driver profile page
  const [comeBackFromDriver, setComeBackFromDriver] = useState(false);
  
  useEffect(() => {
    // Check if this is the first render
    const isFirstRender = !isLoaded;
    
    // Get tab from URL if available - only on first render
    const tabFromUrl = searchParams.get('tab');
    if (isFirstRender && tabFromUrl && (tabFromUrl === 'drivers' || tabFromUrl === 'constructors')) {
      setActiveTab(tabFromUrl);
    }
    
    setIsLoaded(true);
    
    // Track path changes to detect when we come back from driver page
    const handleRouteChange = () => {
      if (pathname === '/standings') {
        setComeBackFromDriver(true);
      }
    };
    
    // Clean up function runs when component unmounts
    return () => {
      setComeBackFromDriver(false);
    };
  }, [pathname, searchParams, isLoaded]);
  
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
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: { 
        delay: i * 0.05,
        type: "spring", 
        stiffness: 70, 
        damping: comeBackFromDriver ? 20 : 10 
      }
    })
  };
  
  // Function to handle driver click
  const handleDriverClick = (slug) => {
    router.push(`/drivers/${slug}`);
  };
  
  // Sample data for drivers standings with online image sources
  const drivers = [
    {
      id: 1,
      position: 1,
      name: "Max Verstappen",
      slug: "max-verstappen",
      nationality: "Dutch",
      team: "Red Bull Racing",
      teamColor: "#0600EF",
      nationality: "Netherlands",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/netherlands-flag.png",
      points: 349,
      wins: 7,
      podiums: 12,
      image: "https://media.formula1.com/content/dam/fom-website/drivers/2023Drivers/verstappen.jpg.img.1920.medium.jpg"
    },
    {
      id: 2,
      position: 2,
      name: "Lando Norris",
      slug: "lando-norris",
      nationality: "British",
      team: "McLaren",
      teamColor: "#FF8700",
      nationality: "United Kingdom",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/great-britain-flag.png",
      points: 253,
      wins: 2,
      podiums: 10,
      image: "https://media.formula1.com/content/dam/fom-website/drivers/2023Drivers/norris.jpg.img.1920.medium.jpg"
    },
    {
      id: 3,
      position: 3,
      name: "Charles Leclerc",
      slug: "charles-leclerc",
      nationality: "Monégasque",
      team: "Ferrari",
      teamColor: "#DC0000",
      nationality: "Monaco",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/monaco-flag.png",
      points: 203,
      wins: 1,
      podiums: 6,
      image: "https://media.formula1.com/content/dam/fom-website/drivers/2023Drivers/leclerc.jpg.img.1920.medium.jpg"
    },
    {
      id: 4,
      position: 4,
      name: "Oscar Piastri",
      abbreviation: "PIA",
      team: "McLaren",
      teamColor: "#FF8700",
      nationality: "Australia",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/australia-flag.png",
      points: 195,
      wins: 1,
      podiums: 5,
      image: "https://media.formula1.com/content/dam/fom-website/drivers/2023Drivers/piastri.jpg.img.1920.medium.jpg"
    },
    {
      id: 5,
      position: 5,
      name: "Carlos Sainz",
      abbreviation: "SAI",
      team: "Ferrari",
      teamColor: "#DC0000",
      nationality: "Spain",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/spain-flag.png",
      points: 184,
      wins: 1,
      podiums: 6,
      image: "https://media.formula1.com/content/dam/fom-website/drivers/2023Drivers/sainz.jpg.img.1920.medium.jpg"
    },
    {
      id: 6,
      position: 6,
      name: "Lewis Hamilton",
      abbreviation: "HAM",
      team: "Mercedes",
      teamColor: "#00D2BE",
      nationality: "United Kingdom",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/great-britain-flag.png",
      points: 177,
      wins: 1,
      podiums: 6,
      image: "https://media.formula1.com/content/dam/fom-website/drivers/2023Drivers/hamilton.jpg.img.1920.medium.jpg"
    },
    // Add more drivers as needed
  ];

  // Sample data for constructors standings with online image sources
  const constructors = [
    {
      id: 1,
      position: 1,
      name: "Red Bull Racing",
      color: "#0600EF",
      points: 499,
      wins: 8,
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/red%20bull.jpg",
      country: "Austria",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/austria-flag.png"
    },
    {
      id: 2,
      position: 2,
      name: "McLaren",
      color: "#FF8700",
      points: 448,
      wins: 3,
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/mclaren.jpg",
      country: "United Kingdom",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/great-britain-flag.png"
    },
    {
      id: 3,
      position: 3,
      name: "Ferrari",
      color: "#DC0000",
      points: 387,
      wins: 2,
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/ferrari.jpg",
      country: "Italy",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/italy-flag.png"
    },
    {
      id: 4,
      position: 4,
      name: "Mercedes",
      color: "#00D2BE",
      points: 334,
      wins: 1,
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/mercedes.jpg",
      country: "Germany",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/germany-flag.png"
    },
    {
      id: 5,
      position: 5,
      name: "Aston Martin",
      color: "#006F62",
      points: 76,
      wins: 0,
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/aston%20martin.jpg",
      country: "United Kingdom",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/great-britain-flag.png"
    },
    {
      id: 6,
      position: 6,
      name: "RB",
      color: "#000080",
      points: 35,
      wins: 0,
      logo: "https://media.formula1.com/content/dam/fom-website/teams/2024/rb-logo.jpg.img.640.medium.jpg",
      country: "Italy",
      flag: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/italy-flag.png"
    },
    // Add more constructors as needed
  ];

  // Rest of the component remains the same
   // Handle tab change with URL update
   const handleTabChange = (tab) => {
    if (tab !== activeTab) {
      setActiveTab(tab);
      
      // Update URL without page reload
      const url = new URL(window.location);
      url.searchParams.set('tab', tab);
      window.history.pushState({}, '', url);
    }
  };
  
  // Using a key on the motion.div ensures proper re-render
  const motionKey = `${activeTab}-${isLoaded}-${comeBackFromDriver}`;
  
  return (
    <div className="standings-container">
      <motion.div 
        className="standings-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="page-title">2025 Championship Standings</h1>
      </motion.div>
      
      <motion.div 
        className="tabs-container"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
      >
        <div className="tabs-nav">
          <button 
            className={`tab-button ${activeTab === 'drivers' ? 'active' : ''}`}
            onClick={() => handleTabChange('drivers')}
          >
            Drivers Championship
          </button>
          <button 
            className={`tab-button ${activeTab === 'constructors' ? 'active' : ''}`}
            onClick={() => handleTabChange('constructors')}
          >
            Constructors Championship
          </button>
        </div>
      </motion.div>
      
      {activeTab === 'drivers' && (
        <motion.div
          key={motionKey}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
          className="drivers-main-container"
        >
          {/* Top drivers */}
          <div className="drivers-container">
            {drivers.map((driver, index) => (
              <motion.div 
                key={driver.id}
                className={index < 3 ? "highlighted-driver-card" : "driver-card"}
                variants={itemVariants}
                custom={index}
                style={{
                  borderLeftColor: driver.teamColor,
                  cursor: 'pointer'
                }}
                onClick={() => handleDriverClick(driver.slug)}
                layoutId={`driver-${driver.id}`}
              >
                <div className="driver-image-container">
                  <Image 
                    src={driver.image}
                    alt={driver.name}
                    width={index < 3 ? 300 : 150}
                    height={index < 3 ? 300 : 150}
                    priority={index < 3}
                    style={{ 
                      objectFit: 'cover', 
                      objectPosition: 'center top'
                    }}
                    unoptimized={true}
                  />
                </div>
                
                <div className="highlighted-content">
                  <div className="driver-header">
                    <span className={`position-badge ${index < 3 ? `position-${driver.position}` : ''}`}>
                      {driver.position}
                    </span>
                    <h3 className="driver-name">
                      {driver.name}
                      <span className="flag-icon">
                        <Image
                          src={driver.flag}
                          alt={driver.nationality}
                          width={20}
                          height={15}
                          style={{ objectFit: 'contain' }}
                          unoptimized={true}
                        />
                      </span>
                    </h3>
                  </div>
                  
                  <div className="driver-details">
                    <div className="detail-group">
                      <span className="detail-label">Team</span>
                      <span className="detail-value" style={{ color: driver.teamColor }}>
                        {driver.team}
                      </span>
                    </div>
                    
                    <div className="detail-group">
                      <span className="detail-label">Wins</span>
                      <span className="detail-value">{driver.wins}</span>
                    </div>
                    
                    {index < 3 && (
                      <div className="detail-group">
                        <span className="detail-label">Podiums</span>
                        <span className="detail-value">{driver.podiums}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="points-display">
                    <span className="points-label">{index < 3 ? "Points" : ""}</span>
                    <span className="points-value">
                      {driver.points}{index >= 3 ? " pts" : ""}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
      
      {activeTab === 'constructors' && (
        <motion.div
          key={`constructors-${isLoaded}`}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="constructors-container"
        >
          <table className="constructors-table">
            <thead>
              <tr>
                <th>Pos</th>
                <th>Team</th>
                <th>Points</th>
                <th>Wins</th>
              </tr>
            </thead>
            <tbody>
              {constructors.map((team, index) => (
                <motion.tr 
                  key={team.id}
                  variants={itemVariants}
                  custom={index}
                >
                  <td>
                    <span className="team-position">
                      <span 
                        className="position-indicator" 
                        style={{ 
                          backgroundColor: team.position <= 3 ? team.color : 'var(--card-border)' 
                        }}
                      >
                        {team.position}
                      </span>
                    </span>
                  </td>
                  <td>
                    <div className="team-name">
                      <div className="team-logo" style={{ backgroundColor: 'white' }}>
                        <Image 
                          src={team.logo}
                          alt={team.name}
                          width={40}
                          height={25}
                          style={{ objectFit: 'contain' }}
                          unoptimized={true}
                        />
                      </div>
                      {team.name}
                      <span className="flag-icon">
                        <Image
                          src={team.flag}
                          alt={team.country}
                          width={20}
                          height={15}
                          style={{ objectFit: 'contain' }}
                          unoptimized={true}
                        />
                      </span>
                    </div>
                  </td>
                  <td className="team-points">{team.points}</td>
                  <td className="team-wins">{team.wins}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}
    </div>
  );
}