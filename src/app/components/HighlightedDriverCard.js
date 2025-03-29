export default function HighlightedDriverCard({ driver }) {
  return (
    <div className="highlighted-driver-card">
      <div className="position-badge">{driver.position}</div>
      <div className="vertical-divider"></div>
      <img
        src={`/images/drivers/${driver.id}.jpg`}
        alt={driver.name}
        className="w-40 h-40 rounded-full mb-6"
      />
      <div className="info">
        <h2>{driver.name}</h2>
        <p>
          Nationality: 
          <img
            src={`/images/flags/${driver.nationality.toLowerCase()}.png`}
            alt={`${driver.nationality} flag`}
            className="flag-icon"
          />
          {driver.nationality}
        </p>
        <p>Team: {driver.team}</p>
        <p>Points: {driver.points}</p>
      </div>
    </div>
  );
}