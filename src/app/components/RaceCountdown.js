"use client";
import { useState, useEffect } from 'react';

export default function RaceCountdown() {
  const [timeRemaining, setTimeRemaining] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0
  });
  
  // Example next race - in a real app this would come from an API or data source
  const nextRace = {
    name: "Australian Grand Prix",
    circuit: "Albert Park Circuit",
    location: "Melbourne, Australia",
    date: "March 30, 2025",
    time: "06:00 GMT",
    // Setting this date to a future time for the countdown
    dateTime: new Date('2025-03-30T06:00:00Z').getTime()
  };
  
  useEffect(() => {
    const updateCountdown = () => {
      const now = new Date().getTime();
      const distance = nextRace.dateTime - now;
      
      if (distance > 0) {
        setTimeRemaining({
          days: Math.floor(distance / (1000 * 60 * 60 * 24)),
          hours: Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)),
          minutes: Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60)),
          seconds: Math.floor((distance % (1000 * 60)) / 1000)
        });
      }
    };
    
    // Update immediately
    updateCountdown();
    
    // Then update every second
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, [nextRace.dateTime]);
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div>
          <h3 className="text-2xl font-bold">{nextRace.name}</h3>
          <p className="text-gray-600">{nextRace.circuit}, {nextRace.location}</p>
          <p className="text-gray-600">{nextRace.date} at {nextRace.time}</p>
        </div>
        
        <div className="mt-4 md:mt-0">
          <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">
            NEXT RACE
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-4 gap-2">
        <div className="bg-gray-100 p-3 rounded-lg text-center">
          <div className="text-3xl font-bold text-red-600">{timeRemaining.days}</div>
          <div className="text-xs uppercase text-gray-600">days</div>
        </div>
        <div className="bg-gray-100 p-3 rounded-lg text-center">
          <div className="text-3xl font-bold text-red-600">{timeRemaining.hours}</div>
          <div className="text-xs uppercase text-gray-600">hours</div>
        </div>
        <div className="bg-gray-100 p-3 rounded-lg text-center">
          <div className="text-3xl font-bold text-red-600">{timeRemaining.minutes}</div>
          <div className="text-xs uppercase text-gray-600">minutes</div>
        </div>
        <div className="bg-gray-100 p-3 rounded-lg text-center">
          <div className="text-3xl font-bold text-red-600">{timeRemaining.seconds}</div>
          <div className="text-xs uppercase text-gray-600">seconds</div>
        </div>
      </div>
      
      {/* Progress bar representing time until the race */}
      <div className="mt-6">
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-red-600 h-2.5 rounded-full" 
            style={{ width: `${Math.min(100, (30 - timeRemaining.days) / 30 * 100)}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
}