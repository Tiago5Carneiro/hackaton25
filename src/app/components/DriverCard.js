import Image from 'next/image';

export default function DriverCard({ driver }) {
  const flagUrl = `https://flagpedia.net/data/flags/icon/72x54/${driver.nationality.toLowerCase()}.png`;

  return (
    <div className="driver-card">
      <div className="position-badge">{driver.position}</div>
      <div className="vertical-divider"></div>
      <div className="info">
        <h2>{driver.name}</h2>
        <p>
          Nationality:
          <img
            src={flagUrl}
            alt={`${driver.nationality} flag`}
            className="flag-icon"
          />
        </p>
        <p>Team: {driver.team}</p>
        <p>Points: {driver.points}</p>
      </div>
    </div>
  );
}