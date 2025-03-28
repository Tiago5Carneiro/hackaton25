"use client";
import { useState } from 'react';

export default function Standings() {
  const [activeTab, setActiveTab] = useState('drivers');
  
  const driverStandings = [
    {
      id: 1,
      position: 1,
      name: "Max Verstappen",
      nationality: "Dutch",
      team: "Red Bull Racing",
      points: 342
    },
    {
      id: 2,
      position: 2,
      name: "Lando Norris",
      nationality: "British",
      team: "McLaren",
      points: 275
    },
    {
      id: 3,
      position: 3,
      name: "Charles Leclerc",
      nationality: "Monégasque",
      team: "Ferrari",
      points: 246
    },
    {
      id: 4,
      position: 4,
      name: "Lewis Hamilton",
      nationality: "British",
      team: "Mercedes",
      points: 189
    },
    {
      id: 5,
      position: 5,
      name: "Carlos Sainz",
      nationality: "Spanish",
      team: "Ferrari",
      points: 185
    },
    {
      id: 6,
      position: 6,
      name: "Oscar Piastri",
      nationality: "Australian",
      team: "McLaren",
      points: 177
    },
    {
      id: 7,
      position: 7,
      name: "Sergio Perez",
      nationality: "Mexican",
      team: "Red Bull Racing",
      points: 150
    },
    {
      id: 8,
      position: 8,
      name: "George Russell",
      nationality: "British",
      team: "Mercedes",
      points: 143
    },
    {
      id: 9,
      position: 9,
      name: "Fernando Alonso",
      nationality: "Spanish",
      team: "Aston Martin",
      points: 58
    },
    {
      id: 10,
      position: 10,
      name: "Lance Stroll",
      nationality: "Canadian",
      team: "Aston Martin",
      points: 24
    },
    {
      id: 11,
      position: 11,
      name: "Alex Albon",
      nationality: "Thai",
      team: "Williams",
      points: 18
    },
    {
      id: 12,
      position: 12,
      name: "Yuki Tsunoda",
      nationality: "Japanese",
      team: "RB",
      points: 14
    },
    {
      id: 13,
      position: 13,
      name: "Daniel Ricciardo",
      nationality: "Australian",
      team: "RB",
      points: 9
    },
    {
      id: 14,
      position: 14,
      name: "Esteban Ocon",
      nationality: "French",
      team: "Alpine",
      points: 6
    },
    {
      id: 15,
      position: 15,
      name: "Pierre Gasly",
      nationality: "French",
      team: "Alpine",
      points: 6
    },
    {
      id: 16,
      position: 16,
      name: "Kevin Magnussen",
      nationality: "Danish",
      team: "Haas",
      points: 5
    },
    {
      id: 17,
      position: 17,
      name: "Nico Hulkenberg",
      nationality: "German",
      team: "Haas",
      points: 5
    },
    {
      id: 18,
      position: 18,
      name: "Logan Sargeant",
      nationality: "American",
      team: "Williams",
      points: 0
    },
    {
      id: 19,
      position: 19,
      name: "Zhou Guanyu",
      nationality: "Chinese",
      team: "Stake F1",
      points: 0
    },
    {
      id: 20,
      position: 20,
      name: "Valtteri Bottas",
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
    <div>
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
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-300">
            <thead>
              <tr className="bg-gray-50">
                <th className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900">Pos</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Driver</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Nationality</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Team</th>
                <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Points</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {driverStandings.map((driver) => (
                <tr key={driver.id} className={driver.position <= 3 ? "bg-gray-50" : ""}>
                  <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900">
                    {driver.position === 1 ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-yellow-400 rounded-full text-white font-bold">
                        {driver.position}
                      </span>
                    ) : driver.position === 2 ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-gray-300 rounded-full text-white font-bold">
                        {driver.position}
                      </span>
                    ) : driver.position === 3 ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 bg-amber-700 rounded-full text-white font-bold">
                        {driver.position}
                      </span>
                    ) : (
                      driver.position
                    )}
                  </td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm font-medium text-gray-900">{driver.name}</td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{driver.nationality}</td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{driver.team}</td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm font-semibold text-gray-900">{driver.points}</td>
                </tr>
              ))}
            </tbody>
          </table>
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