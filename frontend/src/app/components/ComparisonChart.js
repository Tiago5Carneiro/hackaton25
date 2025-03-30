"use client";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip, Legend } from 'recharts';

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
        <p style={{ margin: '0', fontWeight: 'bold' }}>{`${payload[0].payload.skill}`}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ margin: '0', color: entry.color }}>
            {`${entry.name}: ${entry.value}`}
          </p>
        ))}
      </div>
    );
  }
  
  return null;
};

export default function ComparisonChart({ data, driver1, driver2 }) {
  return (
    <RadarChart 
      outerRadius={150} 
      width={600} 
      height={350} 
      data={data}
    >
      <PolarGrid />
      <PolarAngleAxis 
        dataKey="skill" 
        tick={{ fontSize: 12, fill: '#333' }}
      />
      <PolarRadiusAxis 
        angle={30} 
        domain={[60, 100]}
        tick={{ fontSize: 10 }}
      />
      <Radar 
        name={driver1} 
        dataKey={driver1} 
        stroke="#e10600" 
        fill="#e10600" 
        fillOpacity={0.3}
        dot={{ 
          stroke: "#ffffff",
          strokeWidth: 2,
          fill: "#e10600",
          r: 4
        }}
      />
      <Radar 
        name={driver2} 
        dataKey={driver2} 
        stroke="#0600e1" 
        fill="#0600e1" 
        fillOpacity={0.3}
        dot={{ 
          stroke: "#ffffff",
          strokeWidth: 2,
          fill: "#0600e1",
          r: 4
        }}
      />
      <Tooltip content={<CustomTooltip />} />
      <Legend />
    </RadarChart>
  );
}