import Link from 'next/link';

export default function StandingsCard({ title, standings, type = 'driver', limit = 3, showMore = true }) {
  // Determina a URL de redirecionamento baseada no tipo
  const standingsUrl = type === 'driver' ? '/standings?tab=drivers' : '/standings?tab=constructors';
  
  return (
    <div className="standings-card">
      <div className="standings-title">
        {type === 'driver' ? (
          <svg className="standings-title-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        ) : (
          <svg className="standings-title-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
        )}
        {title}
      </div>
      
      <ul className="standings-list">
        {standings.slice(0, limit).map((entry) => (
          <li key={entry.position} className="standings-item">
            <div className="flex items-center">
              <span className={`standings-position ${
                entry.position <= 3 ? `position-${entry.position}` : 'position-other'
              }`}>
                {entry.position}
              </span>
              <div className="standings-info">
                <div className="standings-name">{entry.name}</div>
                {type === 'driver' && entry.team && (
                  <div className="standings-team">{entry.team}</div>
                )}
              </div>
            </div>
            <div className="standings-points-container">
              <span className="standings-points">{entry.points}</span>
              {entry.change && (
                <span className={`standings-change ${
                  entry.change.startsWith('+') 
                    ? 'change-positive' 
                    : entry.change.startsWith('-') 
                      ? 'change-negative' 
                      : 'change-neutral'
                }`}>
                  {entry.change}
                </span>
              )}
            </div>
          </li>
        ))}
      </ul>
      
      {showMore && (
        <div className="mt-4 text-center">
          <Link 
            href={standingsUrl} 
            className="text-[var(--primary)] font-medium hover:underline inline-flex items-center"
          >
            View Full Standings 
            <svg className="ml-1 w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
        </div>
      )}
    </div>
  );
}