"use client";
import { useState, useEffect, useRef } from 'react';
import '../../styles/drivers.css';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import Image from 'next/image';

// Import Recharts components with dynamic loading (correct implementation)
const RechartsComponent = dynamic(() => import('../../components/RechartsComponent'), {
  ssr: false,
  loading: () => <div>Loading Chart...</div>
});

export default function DriverPage() {
  const { slug } = useParams();
  const [driver, setDriver] = useState(null);
  const [loading, setLoading] = useState(true);
  const [contentReady, setContentReady] = useState(false);
  
  // Mock data for driver skills (would come from API in real app)
  const driverSkills = {
    pace: Math.floor(Math.random() * 30) + 70, // 70-100
    consistency: Math.floor(Math.random() * 30) + 70,
    racecraft: Math.floor(Math.random() * 30) + 70,
    tireManagement: Math.floor(Math.random() * 30) + 70,
    wetWeatherDriving: Math.floor(Math.random() * 30) + 70,
  };

  // Format skills data for Recharts
  const skillsData = [
    { skill: "Pace", value: driverSkills.pace, fullMark: 100 },
    { skill: "Consistency", value: driverSkills.consistency, fullMark: 100 },
    { skill: "Racecraft", value: driverSkills.racecraft, fullMark: 100 },
    { skill: "Tire Management", value: driverSkills.tireManagement, fullMark: 100 },
    { skill: "Wet Weather", value: driverSkills.wetWeatherDriving, fullMark: 100 },
    { skill: "Consiste", value: driverSkills.consistency, fullMark: 100 },
    { skill: "Racecr", value: driverSkills.racecraft, fullMark: 100 },
    { skill: "Tire Manage", value: driverSkills.tireManagement, fullMark: 100 },
    { skill: "Wet Wea", value: driverSkills.wetWeatherDriving, fullMark: 100 },

  ];
  
  // Recent results - mock data
  const recentResults = [
    { race: "Italian GP", position: 3, points: 15 },
    { race: "Dutch GP", position: 1, points: 25 },
    { race: "Belgian GP", position: 2, points: 18 },
    { race: "Hungarian GP", position: 5, points: 10 },
    { race: "British GP", position: 4, points: 12 },
  ];
  
  // All drivers data - same as in standings page
  const driverStandings = [
    {
      id: 1,
      position: 1,
      name: "Max Verstappen",
      slug: "max-verstappen",
      nationality: "Dutch",
      team: "Red Bull Racing",
      points: 342,
      dob: "30 Sep 1997",
      wins: 56,
      podiums: 102,
      championships: 3,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png"
    },
    {
      id: 2,
      position: 2,
      name: "Lando Norris",
      slug: "lando-norris",
      nationality: "British",
      team: "McLaren",
      points: 275,
      dob: "13 Nov 1999",
      wins: 2,
      podiums: 17,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/2col/image.png"
    },
    {
      id: 3,
      position: 3,
      name: "Charles Leclerc",
      slug: "charles-leclerc",
      nationality: "Monégasque",
      team: "Ferrari",
      points: 246,
      dob: "16 Oct 1997",
      wins: 5,
      podiums: 28,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/2col/image.png"
    },
    {
      id: 4,
      position: 4,
      name: "Lewis Hamilton",
      slug: "lewis-hamilton",
      nationality: "British",
      team: "Mercedes",
      points: 189,
      dob: "7 Jan 1985",
      wins: 103,
      podiums: 197,
      championships: 7,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/2col/image.png"
    },
    {
      id: 5,
      position: 5,
      name: "Carlos Sainz",
      slug: "carlos-sainz",
      nationality: "Spanish",
      team: "Ferrari",
      points: 185,
      dob: "1 Sep 1994",
      wins: 3,
      podiums: 19,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/2col/image.png"
    },
    {
      id: 6,
      position: 6,
      name: "Oscar Piastri",
      slug: "oscar-piastri",
      nationality: "Australian",
      team: "McLaren",
      points: 177,
      dob: "25 Apr 2001",
      wins: 0,
      podiums: 5,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/2col/image.png"
    },
    {
      id: 7,
      position: 7,
      name: "Sergio Perez",
      slug: "sergio-perez",
      nationality: "Mexican",
      team: "Red Bull Racing",
      points: 150,
      dob: "26 Jan 1990",
      wins: 6,
      podiums: 35,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/S/SERPER01_Sergio_Perez/serper01.png.transform/2col/image.png"
    },
    {
      id: 8,
      position: 8,
      name: "George Russell",
      slug: "george-russell",
      nationality: "British",
      team: "Mercedes",
      points: 143,
      dob: "15 Feb 1998",
      wins: 1,
      podiums: 13,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/2col/image.png"
    },
    {
      id: 9,
      position: 9,
      name: "Fernando Alonso",
      slug: "fernando-alonso",
      nationality: "Spanish",
      team: "Aston Martin",
      points: 58,
      dob: "29 Jul 1981",
      wins: 32,
      podiums: 106,
      championships: 2,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/2col/image.png"
    },
    {
      id: 10,
      position: 10,
      name: "Lance Stroll",
      slug: "lance-stroll",
      nationality: "Canadian",
      team: "Aston Martin",
      points: 24,
      dob: "29 Oct 1998",
      wins: 0,
      podiums: 3,
      championships: 0,
      imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/2col/image.png"
    }
    // Add more drivers as needed...
  ];
  
  // Add a timeout to ensure CSS is applied before showing content
  useEffect(() => {
    if (driver) {
      // Wait for all resources to load before showing content
      const timer = setTimeout(() => {
        setContentReady(true);
      }, 300); // Short delay to ensure CSS is applied
      
      return () => clearTimeout(timer);
    }
  }, [driver]);
  
  // Find driver based on slug
  useEffect(() => {
    setLoading(true);
    setContentReady(false);
    
    const foundDriver = driverStandings.find(driver => driver.slug === slug);
    
    if (foundDriver) {
      // Add flag URL and team logo URL
      foundDriver.flagUrl = `https://flagcdn.com/w80/${getNationalityCode(foundDriver.nationality).toLowerCase()}.png`;
      foundDriver.teamLogoUrl = getTeamLogoUrl(foundDriver.team);
      setDriver(foundDriver);
    }
    
    setLoading(false);
  }, [slug]);
  
  // Substitua a implementação do renderRadarChart por esta versão simplificada:
  const renderRadarChart = () => {
    const ctx = chartRef.current.getContext('2d');
    const centerX = chartRef.current.width / 2;
    const centerY = chartRef.current.height / 2;
    const radius = Math.min(centerX, centerY) - 60; // Reduce radius to leave more space for labels
    
    // Limpar o canvas
    ctx.clearRect(0, 0, chartRef.current.width, chartRef.current.height);
    
    // Preparar dados
    const skills = Object.entries(driverSkills);
    const numSkills = skills.length;
    
    // Desenhar o fundo
    ctx.beginPath();
    ctx.fillStyle = 'rgba(225, 6, 0, 0.1)';
    
    // Desenhar o hexágono de habilidades
    skills.forEach((skill, i) => {
      const angle = (i / numSkills) * Math.PI * 2 - Math.PI / 2;
      const value = skill[1];
      const normalizedValue = value / 100;
      const x = centerX + Math.cos(angle) * radius * normalizedValue;
      const y = centerY + Math.sin(angle) * radius * normalizedValue;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    // Fechar o caminho e preencher
    ctx.closePath();
    ctx.fill();
    
    // Desenhar as linhas do gráfico
    ctx.strokeStyle = '#e10600';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Desenhar os pontos
    skills.forEach((skill, i) => {
      const angle = (i / numSkills) * Math.PI * 2 - Math.PI / 2;
      const value = skill[1];
      const normalizedValue = value / 100;
      const x = centerX + Math.cos(angle) * radius * normalizedValue;
      const y = centerY + Math.sin(angle) * radius * normalizedValue;
      
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = getSkillColor(value);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Adicionar rótulos - increase distance from center and adjust alignment
      const labelRadius = radius + 50; // Increased from 20 to 50
      const labelX = centerX + Math.cos(angle) * labelRadius;
      const labelY = centerY + Math.sin(angle) * labelRadius;
      ctx.fillStyle = '#b0b0b0';
      ctx.font = '14px Arial'; // Slightly larger font
      
      // Adjust text alignment based on position
      if (Math.cos(angle) > 0.1) {
        ctx.textAlign = 'left';
      } else if (Math.cos(angle) < -0.1) {
        ctx.textAlign = 'right';
      } else {
        ctx.textAlign = 'center';
      }
      
      ctx.fillText(skill[0].replace(/([A-Z])/g, ' $1').trim(), labelX, labelY);
    });
  };
  
  // Move the renderOverallRating function inside the component so it has access to driverSkills
  function renderOverallRating() {
    // Calculate average rating from all skills
    const totalSkills = Object.values(driverSkills).reduce((sum, value) => sum + value, 0);
    const overallRating = Math.round(totalSkills / Object.keys(driverSkills).length);
    
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

  if (loading) {
    return (
      <div className="driver-page loading-container">
        <div className="f1-loader"></div>
      </div>
    );
  }

  if (!contentReady) {
    return (
      <div className="driver-page loading-container">
        <div className="f1-loader"></div>
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
            <img 
              src={driver.flagUrl} 
              alt={`${driver.nationality} flag`} 
              className="nationality-flag"
            />
            <span className="nationality-text">{driver.nationality}</span>
          </div>
          <div className="team-info">
            <img 
              src={driver.teamLogoUrl} 
              alt={`${driver.team} logo`} 
              className="team-logo"
            />
            <span className="team-name">{driver.team}</span>
          </div>
          <div className="driver-basic-info">
            <div className="info-item">
              <span className="label">Points:</span> 
              <span className="value">{driver.points}</span>
            </div>
            <div className="info-item">
              <span className="label">Date of Birth:</span> 
              <span className="value">{driver.dob}</span>
            </div>
          </div>
        </div>
        
        <div className="driver-image">
          {driver.imageUrl ? (
            <img 
              src={driver.imageUrl} 
              alt={driver.name} 
              className="driver-portrait"
              key={`${driver.slug}-${Date.now()}`} // Force re-render with unique key
              loading="eager"
              onLoad={() => {
                // Image loaded successfully
                console.log("Driver image loaded");
              }}
              onError={(e) => {
                console.log("Image failed to load, using placeholder");
                e.target.style.display = "none";
                const parent = e.target.parentNode;
                const placeholder = document.createElement("div");
                placeholder.className = "image-placeholder";
                placeholder.textContent = driver.name.split(' ').map(n => n[0]).join('');
                parent.appendChild(placeholder);
              }}
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
              <div className="stat-value">{driver.wins}</div>
              <div className="stat-label">Race Wins</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{driver.podiums}</div>
              <div className="stat-label">Podiums</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{driver.championships}</div>
              <div className="stat-label">Championships</div>
            </div>
          </div>
        </div>
        <div className="driver-skills">
          <h2>Driver Skills</h2>
          <div className="radar-chart-container">
            <RechartsComponent driver={driver} skillsData={skillsData} />
          </div>
          
          {/* Add Football Manager style detailed ratings */}
          <div className="football-manager-ratings">
  <h3>
    Driver Attributes
    <span></span>
  </h3>
  <div className="rating-grid">
    <div className="rating-bars">
      {/* Core Driving */}
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Pace</div>
          <div className="rating-value">{driverSkills.pace}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(driverSkills.pace)}`}
            style={{"--final-width": `${driverSkills.pace}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(driverSkills.pace)}</div>
      </div>
      
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Wet Weather Driving</div>
          <div className="rating-value">{driverSkills.wetWeatherDriving}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(driverSkills.wetWeatherDriving)}`}
            style={{"--final-width": `${driverSkills.wetWeatherDriving}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(driverSkills.wetWeatherDriving)}</div>
      </div>
      
      {/* Consistency */}
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Consistency</div>
          <div className="rating-value">{driverSkills.consistency}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(driverSkills.consistency)}`}
            style={{"--final-width": `${driverSkills.consistency}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(driverSkills.consistency)}</div>
      </div>
      
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Tire Management</div>
          <div className="rating-value">{driverSkills.tireManagement}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(driverSkills.tireManagement)}`}
            style={{"--final-width": `${driverSkills.tireManagement}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(driverSkills.tireManagement)}</div>
      </div>
      
      {/* Racing Skills */}
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Racecraft</div>
          <div className="rating-value">{driverSkills.racecraft}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(driverSkills.racecraft)}`}
            style={{"--final-width": `${driverSkills.racecraft}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(driverSkills.racecraft)}</div>
      </div>
      
      {/* Additional detail attributes - these are random to add more detail */}
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Qualifying</div>
          <div className="rating-value">{70 + Math.floor(Math.random() * 30)}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(70 + Math.floor(Math.random() * 30))}`}
            style={{"--final-width": `${70 + Math.floor(Math.random() * 30)}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(70 + Math.floor(Math.random() * 30))}</div>
      </div>
      
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Overtaking</div>
          <div className="rating-value">{70 + Math.floor(Math.random() * 30)}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(70 + Math.floor(Math.random() * 30))}`}
            style={{"--final-width": `${70 + Math.floor(Math.random() * 30)}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(70 + Math.floor(Math.random() * 30))}</div>
      </div>
      
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Defending</div>
          <div className="rating-value">{70 + Math.floor(Math.random() * 30)}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(70 + Math.floor(Math.random() * 30))}`}
            style={{"--final-width": `${70 + Math.floor(Math.random() * 30)}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(70 + Math.floor(Math.random() * 30))}</div>
      </div>
      
      <div className="rating-item">
        <div className="rating-header">
          <div className="rating-label">Start Performance</div>
          <div className="rating-value">{70 + Math.floor(Math.random() * 30)}</div>
        </div>
        <div className="rating-bar-container">
          <div 
            className={`rating-bar ${getRatingCategory(70 + Math.floor(Math.random() * 30))}`}
            style={{"--final-width": `${70 + Math.floor(Math.random() * 30)}%`}}
          ></div>
        </div>
        <div className="rating-category">{getRatingText(70 + Math.floor(Math.random() * 30))}</div>
      </div>
    </div>
    
    {/* Overall Rating Circle */}
    <div className="overall-rating">
      {renderOverallRating()}
    </div>
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

// Helper function to get color based on skill value
function getSkillColor(value) {
  if (value >= 90) return '#22c55e'; // brighter green
  if (value >= 80) return '#84cc16'; // brighter lime
  if (value >= 70) return '#eab308'; // brighter yellow
  return '#ef4444'; // brighter red
}

// Helper function to get country code from nationality
function getNationalityCode(nationality) {
  const nationalityCodes = {
    'Dutch': 'nl',
    'British': 'gb',
    'Monégasque': 'mc',
    'Spanish': 'es',
    'Australian': 'au',
    'Mexican': 'mx',
    'Finnish': 'fi',
    'Canadian': 'ca',
    'Thai': 'th',
    'Japanese': 'jp',
    'French': 'fr',
    'Danish': 'dk',
    'German': 'de',
    'American': 'us',
    'Chinese': 'cn'
  };
  
  return nationalityCodes[nationality] || 'unknown';
}

// Helper function to get team logo URL
function getTeamLogoUrl(team) {
  // Replace spaces with hyphens and convert to lowercase for URL
  const teamSlug = team.toLowerCase().replace(/\s+/g, '-');
  return `https://www.formula1.com/content/dam/fom-website/teams/2023/${teamSlug}.png.transform/2col/image.png`;
}

// Function to render star ratings (out of 10)
function renderStars(rating) {
  const stars = [];
  const fullStars = Math.floor(rating);
  const hasHalfStar = rating % 1 >= 0.5;
  
  // Add filled stars
  for (let i = 0; i < fullStars; i++) {
    stars.push(<span key={`star-${i}`} className="star filled">★</span>);
  }
  
  // Add half star if needed
  if (hasHalfStar) {
    stars.push(<span key="half-star" className="star half-filled">★</span>);
  }
  
  // Add empty stars
  const emptyStars = 10 - fullStars - (hasHalfStar ? 1 : 0);
  for (let i = 0; i < emptyStars; i++) {
    stars.push(<span key={`empty-${i}`} className="star">★</span>);
  }
  
  return stars;
}

// Add these helper functions before the closing bracket of the file

// Helper function to get rating category based on value
function getRatingCategory(value) {
  if (value >= 90) return 'excellent';
  if (value >= 80) return 'good';
  if (value >= 70) return 'average';
  return 'poor';
}

// Helper function to get rating text
function getRatingText(value) {
  if (value >= 90) return 'EXCELLENT';
  if (value >= 80) return 'GOOD';
  if (value >= 70) return 'AVERAGE';
  return 'NEEDS IMPROVEMENT';
}