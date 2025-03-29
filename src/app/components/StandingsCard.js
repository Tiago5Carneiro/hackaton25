import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';

export default function StandingsCard({ title, standings, type = 'driver', limit = 3, showMore = true }) {
  // Determina a URL de redirecionamento baseada no tipo
  const standingsUrl = type === 'driver' ? '/standings?tab=drivers' : '/standings?tab=constructors';

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: { 
        delay: i * 0.1,
        duration: 0.5,
        ease: "easeOut" 
      }
    })
  };

  return (
    <motion.div 
      className="standings-card"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="card-header">
        <h2 className="card-title">{title}</h2>
        {showMore && (
          <Link href={standingsUrl} className="view-all">
            View Full Standings
          </Link>
        )}
      </div>

      <div className="standings-list">
        {type === 'driver' ? (
          // Driver standings
          standings.slice(0, limit).map((driver, index) => (
            <motion.div 
              key={driver.id} 
              className="driver-standing-item"
              custom={index}
              variants={itemVariants}
              initial="hidden"
              animate="visible"
              style={{ '--team-color': driver.teamColor }}
            >
              <div className="position">{driver.position}</div>
              <div className="driver-info">
                <div className="driver-avatar">
                  <Image
                    src={driver.image || "https://placehold.co/60x60?text=Driver"}
                    alt={driver.name}
                    width={40}
                    height={40}
                    style={{ objectFit: 'cover', objectPosition: 'center top' }}
                  />
                </div>
                <div className="driver-details">
                  <span className="name">{driver.name}</span>
                  <span className="team">{driver.team}</span>
                </div>
              </div>
              <div className="driver-points">{driver.points}</div>
            </motion.div>
          ))
        ) : (
          // Constructor standings
          standings.slice(0, limit).map((team, index) => (
            <motion.div 
              key={team.id} 
              className="constructor-standing-item"
              custom={index}
              variants={itemVariants}
              initial="hidden"
              animate="visible"
            >
              <div className="position">{team.position}</div>
              <div className="constructor-info">
                <div className="team-logo">
                  <Image
                    src={team.logo || "https://placehold.co/80x50?text=Team"}
                    alt={team.name}
                    width={40}
                    height={25}
                    style={{ objectFit: 'contain' }}
                  />
                </div>
                <div className="team-name">
                  {team.name}
                  <span className="team-color" style={{ backgroundColor: team.color }}></span>
                </div>
              </div>
              <div className="constructor-points">{team.points}</div>
            </motion.div>
          ))
        )}
      </div>
    </motion.div>
  );
}