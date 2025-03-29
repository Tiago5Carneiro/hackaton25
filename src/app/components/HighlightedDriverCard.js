import Link from 'next/link';

export default function HighlightedDriverCard({ driver }) {
  return (
    <Link href={`/drivers/${driver.slug}`} className="w-full">
      <div className="highlighted-driver-card bg-gradient-to-r from-red-600 to-red-700 rounded-lg p-6 text-white shadow-lg hover:shadow-xl transition-shadow duration-200 cursor-pointer">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <div className="position-indicator bg-white text-red-600 w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl mr-6">
              {driver.position}
            </div>
            <div>
              <h2 className="text-2xl font-bold">{driver.name}</h2>
              <p className="text-white text-opacity-80">{driver.team}</p>
            </div>
          </div>
          <div className="points text-center">
            <span className="text-3xl font-bold">{driver.points}</span>
            <span className="block text-sm text-white text-opacity-80">POINTS</span>
          </div>
        </div>
      </div>
    </Link>
  );
}