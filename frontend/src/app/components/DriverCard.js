import Link from 'next/link';

export default function DriverCard({ driver }) {
  return (
    <Link href={`/drivers/${driver.slug}`} className="w-full">
      <div className="driver-card flex items-center justify-between rounded-lg bg-white shadow-md p-4 hover:shadow-lg transition-shadow duration-200 cursor-pointer">
        <div className="flex items-center">
          <div className="position-indicator mr-4">
            {driver.position <= 3 ? (
              <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-white font-bold ${
                driver.position === 1 ? 'bg-yellow-400' : 
                driver.position === 2 ? 'bg-gray-300' : 'bg-amber-700'
              }`}>
                {driver.position}
              </span>
            ) : (
              <span className="inline-flex items-center justify-center w-8 h-8 bg-gray-100 rounded-full font-medium text-gray-700">
                {driver.position}
              </span>
            )}
          </div>
          <div className="driver-info">
            <h3 className="text-lg font-bold text-gray-900">{driver.name}</h3>
            <p className="text-sm text-gray-500">{driver.team}</p>
          </div>
        </div>
        <div className="driver-points">
          <span className="text-xl font-bold">{driver.points}</span>
          <span className="text-sm text-gray-500 ml-1">PTS</span>
        </div>
      </div>
    </Link>
  );
}