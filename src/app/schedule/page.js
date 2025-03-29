"use client";
import { useState } from 'react';

export default function SchedulePage() {
  const [selectedRace, setSelectedRace] = useState(null);
  const [filterStatus, setFilterStatus] = useState("all"); // "all", "completed", "upcoming"
  
  const currentDate = new Date();
  
  const races = [
    {
      id: 1,
      name: "Bahrain Grand Prix",
      circuit: "Bahrain International Circuit",
      location: "Sakhir, Bahrain",
      date: "March 2, 2025",
      time: "16:00 GMT",
      completed: true,
      winner: "Max Verstappen",
      fastestLap: "Lewis Hamilton",
      image: "https://placehold.co/600x400",
      description: "The Bahrain Grand Prix is a Formula One championship race which takes place at the Bahrain International Circuit. The 5.412 km circuit was designed by Hermann Tilke, and is located in the middle of the Sakhir desert."
    },
    {
      id: 2,
      name: "Saudi Arabian Grand Prix",
      circuit: "Jeddah Corniche Circuit",
      location: "Jeddah, Saudi Arabia",
      date: "March 16, 2025",
      time: "17:00 GMT",
      completed: true,
      winner: "Charles Leclerc",
      fastestLap: "Max Verstappen",
      image: "https://placehold.co/600x400",
      description: "The Saudi Arabian Grand Prix takes place on the Jeddah Corniche Circuit, which is a street circuit running along the shores of the Red Sea. It is the fastest street circuit on the Formula One calendar, with an average speed of 250 km/h."
    },
    {
      id: 3,
      name: "Australian Grand Prix",
      circuit: "Albert Park Circuit",
      location: "Melbourne, Australia",
      date: "March 30, 2025",
      time: "06:00 GMT",
      completed: false,
      image: "https://placehold.co/600x400",
      description: "The Australian Grand Prix is held at the Albert Park Circuit, a street circuit around Albert Park Lake. The circuit has hosted the Australian Grand Prix since 1996, with the event marking the start of the Formula One season for many years."
    },
    {
      id: 4,
      name: "Japanese Grand Prix",
      circuit: "Suzuka Circuit",
      location: "Suzuka, Japan",
      date: "April 13, 2025",
      time: "05:00 GMT",
      completed: false,
      image: "https://placehold.co/600x400",
      description: "The Japanese Grand Prix is held at the Suzuka Circuit, which is one of the most challenging and popular circuits on the Formula One calendar. The figure-8 layout is unique in Formula One and features a wide variety of corners."
    },
    {
      id: 5,
      name: "Miami Grand Prix",
      circuit: "Miami International Autodrome",
      location: "Miami, United States",
      date: "May 4, 2025",
      time: "19:30 GMT",
      completed: false,
      image: "https://placehold.co/600x400",
      description: "The Miami Grand Prix takes place at the Miami International Autodrome, set around Hard Rock Stadium. The 5.41 km circuit features 19 corners and three DRS zones, with top speeds expected to reach around 320 km/h."
    },
    // Add more races as needed
  ];

  const filteredRaces = filterStatus === "all" 
    ? races 
    : filterStatus === "completed" 
      ? races.filter(race => race.completed)
      : races.filter(race => !race.completed);
  
  return (
    <div className="container mx-auto">
      <h1 className="text-3xl font-bold mb-8">F1 2025 Schedule</h1>
      
      {/* Filter buttons */}
      <div className="flex gap-4 mb-8">
        <button 
          onClick={() => setFilterStatus("all")}
          className={`px-4 py-2 rounded ${filterStatus === "all" ? 
            "bg-red-600 text-white" : "bg-gray-200"}`}
        >
          All Races
        </button>
        <button 
          onClick={() => setFilterStatus("completed")}
          className={`px-4 py-2 rounded ${filterStatus === "completed" ? 
            "bg-red-600 text-white" : "bg-gray-200"}`}
        >
          Completed
        </button>
        <button 
          onClick={() => setFilterStatus("upcoming")}
          className={`px-4 py-2 rounded ${filterStatus === "upcoming" ? 
            "bg-red-600 text-white" : "bg-gray-200"}`}
        >
          Upcoming
        </button>
      </div>
      
      {selectedRace ? (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <button 
            onClick={() => setSelectedRace(null)}
            className="mb-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
          >
            Back to schedule
          </button>
          
          <div className="mb-6">
            <h2 className="text-3xl font-bold mb-2">{selectedRace.name}</h2>
            <div className="flex flex-wrap gap-x-8 gap-y-2 text-gray-600">
              <p>{selectedRace.date} - {selectedRace.time}</p>
              <p>{selectedRace.circuit}, {selectedRace.location}</p>
            </div>
          </div>
          
          <div className="relative w-full h-72 mb-6 rounded-lg overflow-hidden">
            <img 
              src={selectedRace.image} 
              alt={selectedRace.name}
              className="object-cover w-full h-full"
            />
          </div>
          
          <div className="mb-6">
            <h3 className="text-xl font-semibold mb-2">About the race</h3>
            <p className="text-gray-700">{selectedRace.description}</p>
          </div>
          
          {selectedRace.completed && (
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="text-xl font-semibold mb-2">Results</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-gray-600">Winner</p>
                  <p className="font-semibold">{selectedRace.winner}</p>
                </div>
                <div>
                  <p className="text-gray-600">Fastest Lap</p>
                  <p className="font-semibold">{selectedRace.fastestLap}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {filteredRaces.map(race => {
            const isUpcoming = new Date(race.date) > currentDate;
            
            return (
              <div 
                key={race.id}
                onClick={() => setSelectedRace(race)}
                className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-lg transition-shadow border-l-4 border-red-600"
              >
                <div className="p-4">
                  <div className="flex justify-between items-start">
                    <h2 className="text-xl font-bold mb-2">{race.name}</h2>
                    {race.completed ? (
                      <span className="bg-gray-200 text-gray-800 text-xs px-2 py-1 rounded">Completed</span>
                    ) : (
                      <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">Upcoming</span>
                    )}
                  </div>
                  
                  <p className="text-gray-600 mb-4">{race.date} - {race.time}</p>
                  <p className="mb-4">{race.circuit}, {race.location}</p>
                  
                  {race.completed && (
                    <div className="mt-2">
                      <p className="text-sm"><span className="font-semibold">Winner:</span> {race.winner}</p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}