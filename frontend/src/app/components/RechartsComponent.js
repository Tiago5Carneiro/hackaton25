'use client';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip } from 'recharts';

// Helper function for colors
function getSkillColor(value) {
  if (value >= 90) return '#22c55e'; // brighter green
  if (value >= 80) return '#84cc16'; // brighter lime
  if (value >= 70) return '#eab308'; // brighter yellow
  return '#ef4444'; // brighter red
}

// Custom tooltip component
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip" style={{
        color: '#000',
        backgroundColor: '#fff',
        padding: '5px 10px',
        border: '1px solid #ccc',
        borderRadius: '4px',
        fontSize: '12px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <p style={{ margin: '0', fontWeight: 'bold' }}>{`${payload[0].payload.skill}: ${payload[0].value}`}</p>
      </div>
    );
  }
  
  return null;
};

export default function RechartsComponent({ driver, skillsData }) {
  return (
    <RadarChart 
      outerRadius={130} 
      width={400} 
      height={300} 
      data={skillsData}
    >
      <PolarGrid />
      <PolarAngleAxis 
        dataKey="skill" 
        tick={{ fontSize: 10, fill: '#666' }} // Reduzindo o tamanho da fonte para 10px
      />
      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} /> 
      <Radar 
        name={driver.name} 
        dataKey="value" 
        stroke="#e10600" 
        fill="#e10600" 
        fillOpacity={0.6}
        dot={{ 
          stroke: "#ffffff",
          strokeWidth: 2,
          fill: (entry) => getSkillColor(entry.value),
          r: 5
        }}
      />
      <Tooltip content={<CustomTooltip />} />
    </RadarChart>
  );
}