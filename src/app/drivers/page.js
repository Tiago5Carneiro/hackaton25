"use client";
import { useState } from 'react';
import Image from 'next/image';

export default function DriversPage() {
  const [selectedDriver, setSelectedDriver] = useState(null);
  
  const drivers = [
    {
      id: 1,
      name: "Max Verstappen",
      number: 1,
      team: "Red Bull Racing",
      nationality: "Dutch",
      image: "https://via.placeholder.com/150?text=Verstappen",
      dob: "September 30, 1997",
      championships: 3,
      wins: 58,
      podiums: 103,
      bio: "Max Verstappen is a Dutch racing driver and the 2021, 2022, and 2023 Formula One World Champion. He competes under the Dutch flag in Formula One for Red Bull Racing."
    },
    {
      id: 2,
      name: "Lewis Hamilton",
      number: 44,
      team: "Mercedes",
      nationality: "British",
      image: "https://via.placeholder.com/150?text=Hamilton",
      dob: "January 7, 1985",
      championships: 7,
      wins: 103,
      podiums: 197,
      bio: "Sir Lewis Hamilton is a British racing driver currently competing in Formula One for Mercedes. A seven-time World Champion, he is widely regarded as one of the greatest drivers in the history of the sport."
    },
    {
      id: 3,
      name: "Charles Leclerc",
      number: 16,
      team: "Ferrari",
      nationality: "Monégasque",
      image: "https://via.placeholder.com/150?text=Leclerc",
      dob: "October 16, 1997", 
      championships: 0,
      wins: 5,
      podiums: 29,
      bio: "Charles Leclerc is a Monégasque racing driver, currently racing in Formula One for Scuderia Ferrari. Leclerc won the GP3 Series championship in 2016 and the FIA Formula 2 Championship in 2017."
    },
    // Add more drivers as needed
  ];

  return (
    <div className="container mx-auto">
      <h1 className="text-3xl font-bold mb-8">F1 Drivers</h1>
      
      {selectedDriver ? (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <button 
            onClick={() => setSelectedDriver(null)}
            className="mb-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
          >
            Back to all drivers
          </button>
          
          <div className="flex flex-col md:flex-row gap-8">
            <div className="w-full md:w-1/3">
              <div className="relative w-full h-80 rounded-lg overflow-hidden">
                <Image
                  src={selectedDriver.image}
                  alt={selectedDriver.name}
                  fill
                  style={{ objectFit: 'cover' }}
                />
              </div>
            </div>
            
            <div className="w-full md:w-2/3">
              <div className="flex items-center mb-4">
                <h2 className="text-4xl font-bold mr-4">{selectedDriver.name}</h2>
                <span className="bg-gray-200 px-3 py-1 rounded-full text-xl font-bold">#{selectedDriver.number}</span>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <p className="text-gray-600">Team</p>
                  <p className="font-semibold">{selectedDriver.team}</p>
                </div>
                <div>
                  <p className="text-gray-600">Nationality</p>
                  <p className="font-semibold">{selectedDriver.nationality}</p>
                </div>
                <div>
                  <p className="text-gray-600">Date of Birth</p>
                  <p className="font-semibold">{selectedDriver.dob}</p>
                </div>
                <div>
                  <p className="text-gray-600">Championships</p>
                  <p className="font-semibold">{selectedDriver.championships}</p>
                </div>
              </div>
              
              <div className="flex gap-8 mb-6">
                <div className="text-center">
                  <p className="text-3xl font-bold">{selectedDriver.wins}</p>
                  <p className="text-gray-600">Wins</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold">{selectedDriver.podiums}</p>
                  <p className="text-gray-600">Podiums</p>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold mb-2">Biography</h3>
                <p className="text-gray-700">{selectedDriver.bio}</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {drivers.map(driver => (
            <div 
              key={driver.id}
              onClick={() => setSelectedDriver(driver)}
              className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
            >
              <div className="relative h-48">
                <Image
                  src={driver.image}
                  alt={driver.name}
                  fill
                  style={{ objectFit: 'cover' }}
                />
              </div>
              <div className="p-4">
                <div className="flex justify-between items-center">
                  <h2 className="text-xl font-bold">{driver.name}</h2>
                  <span className="bg-gray-200 px-2 py-1 rounded-full font-bold">#{driver.number}</span>
                </div>
                <p className="text-gray-600">{driver.team}</p>
                <div className="mt-4 flex justify-between">
                  <span>{driver.nationality}</span>
                  <span>{driver.wins} Wins</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}