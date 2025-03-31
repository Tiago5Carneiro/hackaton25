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
import { useF1Data } from '../context/F1DataContext';

export default function DriverComparison() {
  const searchParams = useSearchParams();
  const driver1Slug = searchParams.get('driver1');
  const driver2Slug = searchParams.get('driver2');
  
  const { drivers, loading: contextLoading, error: contextError } = useF1Data();
  
  const [driver1, setDriver1] = useState(null);
  const [driver2, setDriver2] = useState(null);
  const [selectedDriver1, setSelectedDriver1] = useState(driver1Slug || '');
  const [selectedDriver2, setSelectedDriver2] = useState(driver2Slug || '');
  const [comparisonData, setComparisonData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Generate skill values for drivers that don't have them
  const generateDriverSkills = (driver) => {
    if (!driver) return null;
    
    // If driver already has skills, use them
    if (driver.skills) return driver;
    
    // Use the same properties as in the driver detail page
    const enhancedDriver = {
      ...driver,
      skills: {
        pace: driver.pace || 85,
        consist: driver.consist || 85, // Alterado de consistency para consist
        tireman: driver.tireman || 85, // Mantém o mesmo nome que está sendo usado
        def: driver.def || 85,
        agress: driver.agress || 85,
        quali: driver.quali || 85,
      },
      flagUrl: `https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/Flags%2016x9/${getNationalityCode(driver.nationality).toLowerCase()}-flag.png`,
      teamLogoUrl: getTeamLogoUrl(driver.team),
      imageUrl: driver.image
    };
    return enhancedDriver;
  };
  
  useEffect(() => {
    if (!contextLoading && drivers && drivers.length > 0) {
      // Create enhanced drivers with additional properties
      const enhancedDrivers = drivers.map(driver => ({
        ...driver,
        slug: driver.id, // Use id as slug if not provided
        flagUrl: `https://flagcdn.com/w80/${getNationalityCode(driver.nationality || '').toLowerCase()}.png`,
        teamLogoUrl: getTeamLogoUrl(driver.team),
        imageUrl: driver.image,
        championships: driver.championships || 0,
        wins: driver.wins || 0,
        podiums: driver.podiums || 0,
        points: driver.points || 0,
        // Use os mesmos nomes de propriedades
        skills: {
          pace: driver.pace || 85,
          consist: driver.consist || 85, // Alterado de consistency para consist
          tireman: driver.tireman || 85,
          def: driver.def || 85,
          agress: driver.agress || 85,
          quali: driver.quali || 85,
        }
      }));
      
      setLoading(false);
      
      // If driver1Slug is provided in URL, find and select that driver
      if (driver1Slug && !selectedDriver1) {
        const found = enhancedDrivers.find(d => d.id === driver1Slug || d.slug === driver1Slug);
        if (found) {
          setSelectedDriver1(found.id);
          setDriver1(generateDriverSkills(found));
        }
      }
      
      // If driver2Slug is provided in URL, find and select that driver
      if (driver2Slug && !selectedDriver2) {
        const found = enhancedDrivers.find(d => d.id === driver2Slug || d.slug === driver2Slug);
        if (found) {
          setSelectedDriver2(found.id);
          setDriver2(generateDriverSkills(found));
        }
      }
    }
  }, [contextLoading, drivers, driver1Slug, driver2Slug, selectedDriver1, selectedDriver2]);
  

  // Update the selected drivers effect to use the same approach
useEffect(() => {
  if (!contextLoading && drivers && drivers.length > 0) {
    // Find selected drivers
    const foundDriver1 = selectedDriver1 ? drivers.find(d => d.id === selectedDriver1 || d.slug === selectedDriver1) : null;
    const foundDriver2 = selectedDriver2 ? drivers.find(d => d.id === selectedDriver2 || d.slug === selectedDriver2) : null;
    
    // Generate skills for selected drivers using consistent values
    const enhancedDriver1 = foundDriver1 ? generateDriverSkills(foundDriver1) : null;
    const enhancedDriver2 = foundDriver2 ? generateDriverSkills(foundDriver2) : null;
    
    setDriver1(enhancedDriver1);
    setDriver2(enhancedDriver2);
    
    // Create comparison data for the chart
    if (enhancedDriver1 && enhancedDriver2) {
      // Use the same skill names as in the driver detail page
      const comparisonData = [
        {
          skill: "Pace",
          [enhancedDriver1.name]: enhancedDriver1.skills.pace,
          [enhancedDriver2.name]: enhancedDriver2.skills.pace,
          fullMark: 100
        },
        {
          skill: "Consistency",
          [enhancedDriver1.name]: enhancedDriver1.skills.consist,
          [enhancedDriver2.name]: enhancedDriver2.skills.consist,
          fullMark: 100
        },
        {
          skill: "Agression",
          [enhancedDriver1.name]: enhancedDriver1.skills.agress, // Use skills.agress here
          [enhancedDriver2.name]: enhancedDriver2.skills.agress, // Use skills.agress here
          fullMark: 100
        },
        {
          skill: "Tire Management",
          [enhancedDriver1.name]: enhancedDriver1.skills.tireman, // Use skills.tireman here
          [enhancedDriver2.name]: enhancedDriver2.skills.tireman, // Use skills.tireman here
          fullMark: 100
        },
        {
          skill: "Qualifying",
          [enhancedDriver1.name]: enhancedDriver1.skills.quali, // Use skills.quali here
          [enhancedDriver2.name]: enhancedDriver2.skills.quali, // Use skills.quali here
          fullMark: 100
        },
        {
          skill: "Defense",
          [enhancedDriver1.name]: enhancedDriver1.skills.def, // Use skills.def here
          [enhancedDriver2.name]: enhancedDriver2.skills.def, // Use skills.def here
          fullMark: 100
        },
      ];
      setComparisonData(comparisonData);
    }
  }
}, [drivers, selectedDriver1, selectedDriver2, contextLoading]);

  const handleDriver1Change = (e) => {
    setSelectedDriver1(e.target.value);
  };

  const handleDriver2Change = (e) => {
    setSelectedDriver2(e.target.value);
  };

  if (loading || contextLoading) {
    return <div className="comparison-page loading">Loading driver data...</div>;
  }

  if (contextError) {
    return <div className="comparison-page error">Error loading driver data: {contextError}</div>;
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
              <option key={`d1-${driver.id}`} value={driver.id}>
                {driver.name}
              </option>
            ))}
          </select>
          
          {driver1 && (
            <div className="selected-driver-card">
              <div className="driver-card-header">
                <img 
                  src={driver1.imageUrl || driver1.image} 
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
              <option key={`d2-${driver.id}`} value={driver.id}>
                {driver.name}
              </option>
            ))}
          </select>
          
          {driver2 && (
            <div className="selected-driver-card">
              <div className="driver-card-header">
                <img 
                  src={driver2.imageUrl || driver2.image} 
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
                <PolarRadiusAxis angle={30} domain={[60, 100]} />
                
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

          {/* Add detailed bar chart comparison */}
          <div className="bar-comparison-container">
            <h2 className="chart-title">Attribute Comparison</h2>
            <div className="attributes-comparison">
              {/* Generate comparison items programmatically */}
              {comparisonData.map((item, index) => (
                <div className="attribute-comparison-item" key={index}>
                  <div className="attribute-label">{item.skill}</div>
                  <div className="attribute-bars">
                    <div className="attribute-bar-wrapper driver1">
                      <div className="attribute-name">{driver1.name.split(' ')[0]}</div>
                      <div className="attribute-bar-container">
                        <div 
                          className="attribute-bar"
                          // Calculate width percentage based on 75-100 scale
                          style={{"--width": `${((item[driver1.name] - 75) / 25) * 100}%`}}
                        ></div>
                        <span className="attribute-value">{item[driver1.name]}</span>
                      </div>
                    </div>
                    <div className="attribute-bar-wrapper driver2">
                      <div className="attribute-name">{driver2.name.split(' ')[0]}</div>
                      <div className="attribute-bar-container">
                        <div 
                          className="attribute-bar"
                          // Calculate width percentage based on 75-100 scale
                          style={{"--width": `${((item[driver2.name] - 75) / 25) * 100}%`}}
                        ></div>
                        <span className="attribute-value">{item[driver2.name]}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="stats-comparison">
            <h2 className="stats-title">Career Comparison</h2>
            <div className="stats-grid">
              <div className="stat-comparison">
                <div className="stat-label">Championships</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.championships || 0}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.championships || 0}</div>
                </div>
              </div>
              
              <div className="stat-comparison">
                <div className="stat-label">Race Wins</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.wins || 0}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.wins || 0}</div>
                </div>
              </div>
              
              <div className="stat-comparison">
                <div className="stat-label">Podiums</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.podiums || 0}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.podiums || 0}</div>
                </div>
              </div>
              
              <div className="stat-comparison">
                <div className="stat-label">Current Points</div>
                <div className="stat-values">
                  <div className="stat-value driver1">{driver1.points || 0}</div>
                  <div className="stat-divider">vs</div>
                  <div className="stat-value driver2">{driver2.points || 0}</div>
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
  if (!nationality) return 'unknown';
  
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
    'Chinese': 'cn',
    'Italian': 'it',
    'Brazilian': 'br'
  };
  
  return nationalityCodes[nationality] || 'unknown';
}

// Helper function to get team logo URL
function getTeamLogoUrl(team) {
  if (!team) return '';
  
  // Replace spaces with hyphens and convert to lowercase for URL
  const teamSlug = team.toLowerCase().replace(/\s+/g, '-');
  return `https://www.formula1.com/content/dam/fom-website/teams/2023/${teamSlug}.png.transform/2col/image.png`;
}


// Helper function to get rating category based on value
function getRatingCategory(value) {
  if (value >= 90) return 'excellent';
  if (value >= 80) return 'good';
  if (value >= 70) return 'average';
  return 'poor';
}