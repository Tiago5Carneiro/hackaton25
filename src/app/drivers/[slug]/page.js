"use client";
import { useState, useEffect, useRef } from 'react';
import '../../styles/drivers.css';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';

// Use dynamic import com lazy loading
const ChartComponent = dynamic(
  () => import('chart.js/auto').then(module => {
    // Registrar os componentes necessários
    return module;
  }),
  { ssr: false, loading: () => <div>Loading Chart...</div> }
);

export default function DriverPage() {
  const { slug } = useParams();
  const [driver, setDriver] = useState(null);
  const [loading, setLoading] = useState(true);
  const [chartInstance, setChartInstance] = useState(null);
  const chartRef = useRef(null);
  const chartLoaded = useRef(false);
  
  // Mock data for driver skills (would come from API in real app)
  const driverSkills = {
    pace: Math.floor(Math.random() * 30) + 70, // 70-100
    consistency: Math.floor(Math.random() * 30) + 70,
    racecraft: Math.floor(Math.random() * 30) + 70,
    tireManagement: Math.floor(Math.random() * 30) + 70,
    wetWeatherDriving: Math.floor(Math.random() * 30) + 70,
  };
  
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
  
  // Setup radar chart when component mounts and Chart.js is loaded
  useEffect(() => {
    if (chartRef.current && driver && typeof ChartComponent !== 'undefined') {
      // Importe o Chart.js apenas quando necessário
      import('chart.js/auto').then((Chart) => {
        renderRadarChart(Chart.default);
        chartLoaded.current = true;
      });
    }
  }, [driver]);
  
  // Find driver based on slug
  useEffect(() => {
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
    const radius = Math.min(centerX, centerY) - 30;
    
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
      
      // Adicionar rótulos
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 20);
      ctx.fillStyle = '#b0b0b0';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(skill[0].replace(/([A-Z])/g, ' $1').trim(), labelX, labelY);
    });
  };
  
  if (loading) {
    return <div className="driver-page loading">Loading driver data...</div>;
  }
  
  if (!driver) {
    return (
      <div className="driver-page error">
        <h1>Driver not found</h1>
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
            <canvas 
              ref={chartRef} 
              width="400" 
              height="400" 
              className="radar-chart"
            ></canvas>
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