"use client";
import { createContext, useContext, useEffect, useState } from 'react';

// API URL - change to your actual backend URL
const API_URL = "http://localhost:5000/api";

// Create the context
const F1DataContext = createContext();

export function F1DataProvider({ children }) {
  // State for storing all data
  const [drivers, setDrivers] = useState([]);
  const [constructors, setConstructors] = useState([]);
  const [races, setRaces] = useState([]);
  const [nextRace, setNextRace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Pre-fetch all data when the app loads
  useEffect(() => {
    async function fetchAllData() {
      setLoading(true);
      try {
        // Fetch data in parallel for better performance
        const [driversResponse, constructorsResponse, racesResponse, nextRaceResponse] = await Promise.all([
          fetch(`${API_URL}/standings/drivers`),
          fetch(`${API_URL}/standings/constructors`),
          fetch(`${API_URL}/events`),
          fetch(`${API_URL}/next-race`) // New endpoint specifically for next race
        ]);

        // Check for errors
        if (!driversResponse.ok) throw new Error('Failed to fetch drivers');
        if (!constructorsResponse.ok) throw new Error('Failed to fetch constructors');

        // Parse responses
        const driversData = await driversResponse.json();
        const constructorsData = await constructorsResponse.json();
        
        // Parse races if available, otherwise use empty array
        let racesData = [];
        if (racesResponse.ok) {
          racesData = await racesResponse.json();
        }
        
        // Get next race data specifically
        if (nextRaceResponse.ok) {
          const nextRaceData = await nextRaceResponse.json();
          if (nextRaceData) {
            // Format the date for display
            if (nextRaceData.startDate) {
              const raceDate = new Date(nextRaceData.startDate);
              nextRaceData.formattedDate = raceDate.toLocaleDateString('en-US', {
                month: 'long',
                day: 'numeric',
                year: 'numeric'
              });
              
              nextRaceData.formattedTime = raceDate.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                timeZoneName: 'short'
              });
              
              // Add countdown timestamp
              nextRaceData.dateTime = raceDate.getTime();
            }
            
            // Set the next race with formatted data
            setNextRace(nextRaceData);
          }
        } else {
          // Fallback to finding next race from races if specific endpoint failed
          const now = new Date();
          const upcomingRaces = racesData
            .filter(race => race.completed === false)
            .sort((a, b) => new Date(a.startDate) - new Date(b.startDate));
          
          if (upcomingRaces.length > 0) {
            const nextRaceFromList = upcomingRaces[0];
            
            // Format the date for display
            if (nextRaceFromList.startDate) {
              const raceDate = new Date(nextRaceFromList.startDate);
              nextRaceFromList.formattedDate = raceDate.toLocaleDateString('en-US', {
                month: 'long',
                day: 'numeric',
                year: 'numeric'
              });
              
              nextRaceFromList.formattedTime = raceDate.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                timeZoneName: 'short'
              });
              
              // Add countdown timestamp
              nextRaceFromList.dateTime = raceDate.getTime();
            }
            
            setNextRace(nextRaceFromList);
          }
        }

        // Update state with fetched data
        setDrivers(driversData);
        setConstructors(constructorsData);
        setRaces(racesData);
        setError(null);
        
        console.log('Data fetched successfully:', { 
          drivers: driversData.length,
          constructors: constructorsData.length,
          races: racesData.length,
          nextRace: nextRace ? 'Found' : 'Not found'
        });
      } catch (err) {
        console.error('Error pre-loading F1 data:', err);
        setError(err.message);
        // Keep previous data if available, otherwise set empty arrays
        setDrivers(prev => prev.length > 0 ? prev : []);
        setConstructors(prev => prev.length > 0 ? prev : []);
        setRaces(prev => prev.length > 0 ? prev : []);
      } finally {
        setLoading(false);
      }
    }

    fetchAllData();
  }, []);

  // Get next race information
  const getNextRace = async () => {
    try {
      const res = await fetch(`${API_URL}/next-race`);
      if (!res.ok) throw new Error('Failed to fetch next race');
      const data = await res.json();
      
      // Format data for display
      if (data && data.startDate) {
        const raceDate = new Date(data.startDate);
        data.formattedDate = raceDate.toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric'
        });
        
        data.formattedTime = raceDate.toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
          timeZoneName: 'short'
        });
        
        // Add countdown timestamp
        data.dateTime = raceDate.getTime();
      }
      
      // Update context state
      setNextRace(data);
      return data;
    } catch (err) {
      console.error(`Error fetching next race:`, err);
      return null;
    }
  };

  // Get driver details by ID
  const getDriverById = async (id) => {
    try {
      const res = await fetch(`${API_URL}/driver/${id}`);
      if (!res.ok) throw new Error('Failed to fetch driver details');
      return await res.json();
    } catch (err) {
      console.error(`Error fetching driver ${id}:`, err);
      return null;
    }
  };
  
  // Get driver details by slug
  const getDriverBySlug = async (slug) => {
    try {
      const res = await fetch(`${API_URL}/driver/slug/${slug}`);
      if (!res.ok) throw new Error('Failed to fetch driver details');
      return await res.json();
    } catch (err) {
      console.error(`Error fetching driver ${slug}:`, err);
      return null;
    }
  };
  
  // Get team details
  const getTeamByName = async (name) => {
    try {
      const res = await fetch(`${API_URL}/team/${name}`);
      if (!res.ok) throw new Error('Failed to fetch team details');
      return await res.json();
    } catch (err) {
      console.error(`Error fetching team ${name}:`, err);
      return null;
    }
  };
  
  // Get event details by ID
  const getEventById = async (id) => {
    try {
      const res = await fetch(`${API_URL}/event/${id}`);
      if (!res.ok) throw new Error('Failed to fetch event details');
      return await res.json();
    } catch (err) {
      console.error(`Error fetching event ${id}:`, err);
      return null;
    }
  };
  
  // Search drivers, teams, and events
  const search = async (query) => {
    if (!query || query.length < 2) return [];
    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error('Search failed');
      return await res.json();
    } catch (err) {
      console.error(`Error searching for ${query}:`, err);
      return [];
    }
  };

  // Value to be provided to consumers
  const value = {
    drivers,
    constructors,
    races,
    nextRace,
    loading,
    error,
    // Data access methods
    getDriverById,
    getDriverBySlug,
    getTeamByName,
    getEventById,
    getNextRace,
    search,
    // Refresh methods
    refreshDrivers: async () => {
      try {
        const res = await fetch(`${API_URL}/standings/drivers`);
        if (!res.ok) throw new Error('Failed to refresh drivers');
        const data = await res.json();
        setDrivers(data);
        return true;
      } catch (err) {
        console.error(err);
        return false;
      }
    },
    refreshConstructors: async () => {
      try {
        const res = await fetch(`${API_URL}/standings/constructors`);
        if (!res.ok) throw new Error('Failed to refresh constructors');
        const data = await res.json();
        setConstructors(data);
        return true;
      } catch (err) {
        console.error(err);
        return false;
      }
    },
    refreshRaces: async () => {
      try {
        const res = await fetch(`${API_URL}/events`);
        if (!res.ok) throw new Error('Failed to refresh races');
        const data = await res.json();
        setRaces(data);
        
        // Also refresh the next race
        getNextRace();
        
        return true;
      } catch (err) {
        console.error(err);
        return false;
      }
    }
  };

  return <F1DataContext.Provider value={value}>{children}</F1DataContext.Provider>;
}

// Custom hook for using the context
export function useF1Data() {
  const context = useContext(F1DataContext);
  if (context === undefined) {
    throw new Error('useF1Data must be used within an F1DataProvider');
  }
  return context;
}