"use client";
import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { useRouter, useSearchParams } from 'next/navigation';
import { usePathname } from 'next/navigation';
import { useF1Data } from '../context/F1DataContext';
import '../styles/standings.css';

export default function Standings() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  
  // Get data from F1DataContext
  const { drivers, constructors, loading: contextLoading, error: contextError, refreshDrivers, refreshConstructors } = useF1Data();
  
  // Filter active drivers
  const activeDrivers = drivers.filter(driver => driver.isActive === 1);

  // Using the URL parameter to initialize the tab state ensures consistency
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState(tabParam === 'constructors' ? 'constructors' : 'drivers');
  const [isLoaded, setIsLoaded] = useState(false);
  
  // Track if we're coming back from a driver profile page
  const [comeBackFromDriver, setComeBackFromDriver] = useState(false);
  
  // Local loading state for UI purposes
  const [loading, setLoading] = useState(true);
  
  
  // Refresh data when component mounts
  useEffect(() => {
    const refreshData = async () => {
      setLoading(true);
      
      try {
        // Refresh data from context
        if (activeTab === 'drivers') {
          await refreshDrivers();
        } else {
          await refreshConstructors();
        }
      } catch (err) {
        console.error('Error refreshing data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    // Use context data directly if already loaded
    if (!contextLoading) {
      setLoading(false);
    } else {
      refreshData();
    }
  }, [activeTab, contextLoading, refreshDrivers, refreshConstructors]);
  
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
  
  // Render loading state
  if (loading || contextLoading) {
    return (
      <div className="standings-container">
        <div className="loading-spinner">Loading standings data...</div>
      </div>
    );
  }

  // Render error state
  if (contextError) {
    return (
      <div className="standings-container">
        <div className="error-message">Failed to load data. Please try again later.</div>
      </div>
    );
  }
  
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
      
      {activeTab === 'drivers' && activeDrivers.length > 0 && (
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
            {activeDrivers.map((driver, index) => (
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
                    src={driver.image || '/placeholder-driver.png'} // Fallback image
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
                      {driver.flag && (
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
                      )}
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
                        <span className="detail-value">{driver.podiums || 0}</span>
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
      
      {activeTab === 'constructors' && constructors.length > 0 && (
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
                      {team.logo && (
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
                      )}
                      {team.name}
                      {team.flag && (
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
                      )}
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