"use client";
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  Legend, 
  Tooltip,
  ResponsiveContainer 
} from 'recharts';
import '../styles/comparison.css';

export default function DriverComparison() {
  const searchParams = useSearchParams();
  const driver1Slug = searchParams.get('driver1');
  const driver2Slug = searchParams.get('driver2');
  
  const [drivers, setDrivers] = useState([]);
  const [driver1, setDriver1] = useState(null);
  const [driver2, setDriver2] = useState(null);
  const [selectedDriver1, setSelectedDriver1] = useState(driver1Slug || '');
  const [selectedDriver2, setSelectedDriver2] = useState(driver2Slug || '');
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Load all drivers
  useEffect(() => {
    // Fetch or use hard-coded driver data
    const driverData = [
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
        imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png",
        skills: {
          pace: 97,
          consistency: 95,
          racecraft: 96,
          tireManagement: 92,
          wetWeatherDriving: 98,
        }
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
        imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/2col/image.png",
        skills: {
          pace: 94,
          consistency: 90,
          racecraft: 91,
          tireManagement: 89,
          wetWeatherDriving: 92,
        }
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
        imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/2col/image.png",
        skills: {
          pace: 96,
          consistency: 88,
          racecraft: 90,
          tireManagement: 87,
          wetWeatherDriving: 90,
        }
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
        imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/2col/image.png",
        skills: {
          pace: 95,
          consistency: 94,
          racecraft: 97,
          tireManagement: 95,
          wetWeatherDriving: 99,
        }
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
        imageUrl: "https://www.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/2col/image.png",
        skills: {
          pace: 92,
          consistency: 90,
          racecraft: 89,
          tireManagement: 91,
          wetWeatherDriving: 87,
        }
      },
    ];
    
    // Add flag URLs and team logo URLs to each driver
    const driversWithExtras = driverData.map(driver => ({
      ...driver,
      flagUrl: `https://flagcdn.com/w80/${getNationalityCode(driver.nationality).toLowerCase()}.png`,
      teamLogoUrl: getTeamLogoUrl(driver.team)
    }));
    
    setDrivers(driversWithExtras);
    setLoading(false);
  }, []);

  // Load driver details when selection changes
  useEffect(() => {
    if (drivers.length > 0) {
      const foundDriver1 = selectedDriver1 ? drivers.find(d => d.slug === selectedDriver1) : null;
      const foundDriver2 = selectedDriver2 ? drivers.find(d => d.slug === selectedDriver2) : null;
      
      setDriver1(foundDriver1);
      setDriver2(foundDriver2);
      
      // Create comparison data for the chart
      if (foundDriver1 && foundDriver2) {
        const comparisonData = [
          {
            skill: "Pace",
            [foundDriver1.name]: foundDriver1.skills.pace,
            [foundDriver2.name]: foundDriver2.skills.pace,
            fullMark: 100
          },
          {
            skill: "Consistency",
            [foundDriver1.name]: foundDriver1.skills.consistency,
            [foundDriver2.name]: foundDriver2.skills.consistency,
            fullMark: 100
          },
          {
            skill: "Racecraft",
            [foundDriver1.name]: foundDriver1.skills.racecraft,
            [foundDriver2.name]: foundDriver2.skills.racecraft,
            fullMark: 100
          },
          {
            skill: "Tire Management",
            [foundDriver1.name]: foundDriver1.skills.tireManagement,
            [foundDriver2.name]: foundDriver2.skills.tireManagement,
            fullMark: 100
          },
          {
            skill: "Wet Weather Driving",
            [foundDriver1.name]: foundDriver1.skills.wetWeatherDriving,
            [foundDriver2.name]: foundDriver2.skills.wetWeatherDriving,
            fullMark: 100
          },
        ];
        setComparisonData(comparisonData);
      }
    }
  }, [drivers, selectedDriver1, selectedDriver2]);

  const handleDriver1Change = (e) => {
    setSelectedDriver1(e.target.value);
  };

  const handleDriver2Change = (e) => {
    setSelectedDriver2(e.target.value);
  };

  if (loading) {
    return <div className="comparison-page loading">Loading driver data...</div>;
  }

  return (
    <div className="comparison-page">
      <Link href="/standings" className="back-link">
        ← Back to Standings
      </Link>
      <div/>
      <div className="comparison-header">
      
        <h1 className="comparison-title">Driver Comparison</h1>

        </div>
      
      <div className="driver-selector-container">
        <div className="driver-selector">
          <h2>Driver 1</h2>
          <select 
            value={selectedDriver1} 
            onChange={handleDriver1Change}
            className="driver-select"
          >
            <option value="">Select Driver 1</option>
            {drivers.map(driver => (
              <option key={`d1-${driver.id}`} value={driver.slug}>
                {driver.name}
              </option>
            ))}
          </select>
          
          {driver1 && (
            <div className="selected-driver-card">
              <div className="driver-card-header">
                <img 
                  src={driver1.imageUrl} 
                  alt={driver1.name} 
                  className="driver-mini-image"
                />
                <div className="driver-mini-info">
                  <h3>{driver1.name}</h3>
                  <div className="driver-mini-team">
                    <img 
                      src={driver1.teamLogoUrl} 
                      alt={driver1.team} 
                      className="team-mini-logo"
                    />
                    <span>{driver1.team}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        <div className="versus">VS</div>
        
        <div className="driver-selector">
          <h2>Driver 2</h2>
          <select 
            value={selectedDriver2} 
            onChange={handleDriver2Change}
            className="driver-select"
          >
            <option value="">Select Driver 2</option>
            {drivers.map(driver => (
              <option key={`d2-${driver.id}`} value={driver.slug}>
                {driver.name}
              </option>
            ))}
          </select>
          
          {driver2 && (
            <div className="selected-driver-card">
              <div className="driver-card-header">
                <img 
                  src={driver2.imageUrl} 
                  alt={driver2.name} 
                  className="driver-mini-image"
                />
                <div className="driver-mini-info">
                  <h3>{driver2.name}</h3>
                  <div className="driver-mini-team">
                    <img 
                      src={driver2.teamLogoUrl} 
                      alt={driver2.team} 
                      className="team-mini-logo"
                    />
                    <span>{driver2.team}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {driver1 && driver2 ? (
        <div className="comparison-container">
          <div className="radar-comparison-container">
            <h2 className="chart-title">Skill Comparison</h2>
            <ResponsiveContainer width="100%" height={500}>
              <RadarChart outerRadius="80%" data={comparisonData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="skill" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                
                <Radar
                  name={driver1.name}
                  dataKey={driver1.name}
                  stroke="#e10600"
                  fill="#e10600"
                  fillOpacity={0.6}
                />
                
                <Radar
                  name={driver2.name}
                  dataKey={driver2.name}
                  stroke="#1E88E5"
                  fill="#1E88E5"
                  fillOpacity={0.6}
                />
                
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="stats-comparison">
            <h2 className="stats-title">Career Comparison</h2>
            <div className="stats-grid">
              <div className="stat-comparison">
                <div className="stat-label">Championships</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.championships}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.championships}</div>
                </div>
              </div>
              
              <div className="stat-comparison">
                <div className="stat-label">Race Wins</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.wins}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.wins}</div>
                </div>
              </div>
              
              <div className="stat-comparison">
                <div className="stat-label">Podiums</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.podiums}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.podiums}</div>
                </div>
              </div>
              
              <div className="stat-comparison">
                <div className="stat-label">Current Points</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.points}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.points}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="no-selection">
          <p>Select two drivers to compare their skills and statistics</p>
        </div>
      )}
    </div>
  );
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