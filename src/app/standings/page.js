"use client";
import { useState } from 'react';
import '../styles/standings.css';
import HighlightedDriverCard from '../components/HighlightedDriverCard';
import DriverCard from '../components/DriverCard';

export default function Standings() {
  const [activeTab, setActiveTab] = useState('drivers');
  
  const driverStandings = [
    {
      id: 1,
      position: 1,
      name: "Max Verstappen",
      slug: "max-verstappen",
      nationality: "Dutch",
      team: "Red Bull Racing",
      points: 342
    },
    {
      id: 2,
      position: 2,
      name: "Lando Norris",
      slug: "lando-norris",
      nationality: "British",
      team: "McLaren",
      points: 275
    },
    {
      id: 3,
      position: 3,
      name: "Charles Leclerc",
      slug: "charles-leclerc",
      nationality: "Monégasque",
      team: "Ferrari",
      points: 246
    },
    {
      id: 4,
      position: 4,
      name: "Lewis Hamilton",
      slug: "lewis-hamilton",
      nationality: "British",
      team: "Mercedes",
      points: 189
    },
    {
      id: 5,
      position: 5,
      name: "Carlos Sainz",
      slug: "carlos-sainz",
      nationality: "Spanish",
      team: "Ferrari",
      points: 185
    },
    {
      id: 6,
      position: 6,
      name: "Oscar Piastri",
      slug: "oscar-piastri",
      nationality: "Australian",
      team: "McLaren",
      points: 177
    },
    {
      id: 7,
      position: 7,
      name: "Sergio Perez",
      slug: "sergio-perez",
      nationality: "Mexican",
      team: "Red Bull Racing",
      points: 150
    },
    {
      id: 8,
      position: 8,
      name: "George Russell",
      slug: "george-russell",
      nationality: "British",
      team: "Mercedes",
      points: 143
    },
    {
      id: 9,
      position: 9,
      name: "Fernando Alonso",
      slug: "fernando-alonso",
      nationality: "Spanish",
      team: "Aston Martin",
      points: 58
    },
    {
      id: 10,
      position: 10,
      name: "Lance Stroll",
      slug: "lance-stroll",
      nationality: "Canadian",
      team: "Aston Martin",
      points: 24
    },
    {
      id: 11,
      position: 11,
      name: "Alex Albon",
      slug: "alex-albon",
      nationality: "Thai",
      team: "Williams",
      points: 18
    },
    {
      id: 12,
      position: 12,
      name: "Yuki Tsunoda",
      slug: "yuki-tsunoda",
      nationality: "Japanese",
      team: "RB",
      points: 14
    },
    {
      id: 13,
      position: 13,
      name: "Daniel Ricciardo",
      slug: "daniel-ricciardo",
      nationality: "Australian",
      team: "RB",
      points: 9
    },
    {
      id: 14,
      position: 14,
      name: "Esteban Ocon",
      slug: "esteban-ocon",
      nationality: "French",
      team: "Alpine",
      points: 6
    },
    {
      id: 15,
      position: 15,
      name: "Pierre Gasly",
      slug: "pierre-gasly",
      nationality: "French",
      team: "Alpine",
      points: 6
    },
    {
      id: 16,
      position: 16,
      name: "Kevin Magnussen",
      slug: "kevin-magnussen",
      nationality: "Danish",
      team: "Haas",
      points: 5
    },
    {
      id: 17,
      position: 17,
      name: "Nico Hulkenberg",
      slug: "nico-hulkenberg",
      nationality: "German",
      team: "Haas",
      points: 5
    },
    {
      id: 18,
      position: 18,
      name: "Logan Sargeant",
      slug: "logan-sargeant",
      nationality: "American",
      team: "Williams",
      points: 0
    },
    {
      id: 19,
      position: 19,
      name: "Zhou Guanyu",
      slug: "zhou-guanyu",
      nationality: "Chinese",
      team: "Stake F1",
      points: 0
    },
    {
      id: 20,
      position: 20,
      name: "Valtteri Bottas",
      slug: "valtteri-bottas",
      nationality: "Finnish",
      team: "Stake F1",
      points: 0
    }
  ];
  
  const constructorStandings = [
    {
      id: 1,
      position: 1,
      name: "Red Bull Racing",
      points: 492,
      wins: 12
    },
    {
      id: 2,
      position: 2,
      name: "McLaren",
      points: 452,
      wins: 5
    },
    {
      id: 3,
      position: 3,
      name: "Ferrari",
      points: 431,
      wins: 3
    },
    {
      id: 4,
      position: 4,
      name: "Mercedes",
      points: 332,
      wins: 2
    },
    {
      id: 5,
      position: 5,
      name: "Aston Martin",
      points: 82,
      wins: 0
    },
    {
      id: 6,
      position: 6,
      name: "RB",
      points: 23,
      wins: 0
    },
    {
      id: 7,
      position: 7,
      name: "Alpine",
      points: 12,
      wins: 0
    },
    {
      id: 8,
      position: 8,
      name: "Haas",
      points: 10,
      wins: 0
    },
    {
      id: 9,
      position: 9,
      name: "Williams",
      points: 18,
      wins: 0
    },
    {
      id: 10,
      position: 10,
      name: "Stake F1",
      points: 0,
      wins: 0
    }
  ];
  
  return (
    <div className="standings-page">
      <h1 className="text-3xl font-bold mb-6">Championship Standings</h1>
      
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('drivers')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'drivers'
                  ? 'border-red-600 text-red-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Drivers Championship
            </button>
            <button
              onClick={() => setActiveTab('constructors')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'constructors'
                  ? 'border-red-600 text-red-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Constructors Championship
            </button>
          </nav>
        </div>
      </div>
      
      {activeTab === 'drivers' ? (
        <div className="flex flex-col items-center gap-8 w-full">
          {/* Highlighted card for the first driver */}
          <HighlightedDriverCard driver={driverStandings[0]} />

          {/* Normal cards for the other drivers */}
          {driverStandings.slice(1).map((driver) => (
            <DriverCard key={driver.id} driver={driver} />
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-300">
            <thead>
              <tr className="bg-gray-50">
                <th className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900">Pos</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Team</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Points</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Wins</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {constructorStandings.map((team) => (
                <tr key={team.id} className={team.position <= 3 ? "bg-gray-50" : ""}>
                  <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900">
                    {team.position === 1 ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-yellow-400 rounded-full text-white font-bold">
                        {team.position}
                      </span>
                    ) : team.position === 2 ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-gray-300 rounded-full text-white font-bold">
                        {team.position}
                      </span>
                    ) : team.position === 3 ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-amber-700 rounded-full text-white font-bold">
                        {team.position}
                      </span>
                    ) : (
                      team.position
                    )}
                  </td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm font-medium text-gray-900">{team.name}</td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm font-semibold text-gray-900">{team.points}</td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{team.wins}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}