"use client";
import React, { createContext, useContext, useState, useEffect } from 'react';

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
  const [dataInitialized, setDataInitialized] = useState(false);

  // Pre-fetch all data when the app loads
  useEffect(() => {
    // Only run once to prevent infinite loops
    if (!dataInitialized) {
      fetchAllData();
    }
  }, [dataInitialized]);

  // Function to fetch all initial data
  const fetchAllData = async () => {
    setLoading(true);
    try {
      // Concurrent fetch requests for better performance
      const [driversRes, constructorsRes, eventsRes, nextRacesRes] = await Promise.allSettled([
        fetch(`${API_URL}/standings/drivers`),
        fetch(`${API_URL}/standings/constructors`),
        fetch(`${API_URL}/all-events`),
        fetch(`${API_URL}/next-races?limit=5`)
      ]);
      
      // Process driver standings
      if (driversRes.status === 'fulfilled' && driversRes.value.ok) {
        const data = await driversRes.value.json();
        setDrivers(data);
      } else {
        console.error("Failed to fetch driver standings");
      }
      
      // Process constructor standings
      if (constructorsRes.status === 'fulfilled' && constructorsRes.value.ok) {
        const data = await constructorsRes.value.json();
        setConstructors(data);
      } else {
        console.error("Failed to fetch constructor standings");
      }
      
      // Process events/races
      if (eventsRes.status === 'fulfilled' && eventsRes.value.ok) {
        const data = await eventsRes.value.json();
        setRaces(data);
      } else {
        console.error("Failed to fetch events/races");
      }
      
      // Process next races - this is used for the home page countdown
      if (nextRacesRes.status === 'fulfilled' && nextRacesRes.value.ok) {
        const nextRacesData = await nextRacesRes.value.json();
        if (nextRacesData && nextRacesData.length > 0) {
          setNextRace(nextRacesData[0]);
        } else {
          // If next-races endpoint returned empty, try to find the next race from all races
          findNextRaceFromAllRaces();
        }
      } else {
        // If next-races endpoint failed, try to find the next race from all races
        findNextRaceFromAllRaces();
      }
      
      // Mark initialization as complete
      setDataInitialized(true);
    } catch (err) {
      console.error("Error fetching initial data:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to find the next race from all races
  const findNextRaceFromAllRaces = () => {
    if (races && races.length > 0) {
      const now = new Date();
      const upcomingRaces = races.filter(race => {
        const raceDate = new Date(race.startDate || race.date);
        return raceDate > now;
      });
      
      if (upcomingRaces.length > 0) {
        // Sort by date and take the first one
        const sortedRaces = [...upcomingRaces].sort((a, b) => {
          const dateA = new Date(a.startDate || a.date);
          const dateB = new Date(b.startDate || b.date);
          return dateA - dateB;
        });
        
        setNextRace(sortedRaces[0]);
      }
    }
  };

  // Get next race information
  const getNextRace = async () => {
    try {
      const res = await fetch(`${API_URL}/next-race`);
      if (!res.ok) throw new Error(`Failed to fetch next race: ${res.status}`);
      const data = await res.json();
      
      // Format data for display
      if (data && data.startDate) {
        data.formattedDate = formatDate(data.startDate);
        data.formattedTime = formatTime(data.time, data.startDate);
      }
      
      // Update context state
      setNextRace(data);
      return data;
    } catch (err) {
      console.error(`Error fetching next race:`, err);
      return null;
    }
  };

  // Format helper functions (can be used by consumers too)
  const formatDate = (dateString) => {
    if (!dateString) return "TBD";
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { 
        weekday: 'long',
        month: 'long', 
        day: 'numeric', 
        year: 'numeric' 
      });
    } catch (e) {
      return dateString;
    }
  };

  const formatTime = (timeString, dateString) => {
    if (!timeString) {
      if (dateString && dateString.includes('T')) {
        try {
          const date = new Date(dateString);
          return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            timeZoneName: 'short'
          });
        } catch (e) {
          return "TBD";
        }
      }
      return "TBD";
    }
    return timeString;
  };

  // Get next X races (including future events)
  const getNextRaces = async (limit = 5) => {
    try {
      const res = await fetch(`${API_URL}/next-races?limit=${limit}`);
      if (!res.ok) throw new Error(`Failed to fetch upcoming races: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error(`Error fetching upcoming races:`, err);
      return [];
    }
  };

  // Get all events (past, present and future)
  const getAllEvents = async () => {
    try {
      const res = await fetch(`${API_URL}/all-events`);
      if (!res.ok) throw new Error(`Failed to fetch all events: ${res.status}`);
      const data = await res.json();
      setRaces(data); // Update races in context
      return data;
    } catch (err) {
      console.error(`Error fetching all events:`, err);
      return [];
    }
  };

  // Get future events only
  const getFutureEvents = async () => {
    try {
      const res = await fetch(`${API_URL}/future-events`);
      if (!res.ok) throw new Error(`Failed to fetch future events: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error(`Error fetching future events:`, err);
      return [];
    }
  };

  // Get driver details by ID
  const getDriverById = async (id) => {
    try {
      const res = await fetch(`${API_URL}/driver/${id}`);
      if (!res.ok) throw new Error(`Failed to fetch driver ${id}: ${res.status}`);
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
      if (!res.ok) throw new Error(`Failed to fetch driver ${slug}: ${res.status}`);
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
      if (!res.ok) throw new Error(`Failed to fetch team ${name}: ${res.status}`);
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
      if (!res.ok) throw new Error(`Failed to fetch event ${id}: ${res.status}`);
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
      if (!res.ok) throw new Error(`Search failed: ${res.status}`);
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
    // Utility methods
    formatDate,
    formatTime,
    // Data access methods
    getDriverById,
    getDriverBySlug,
    getTeamByName,
    getEventById,
    getNextRace,
    getNextRaces,
    getAllEvents,
    getFutureEvents,
    search,
    // Refresh methods
    refreshDrivers: async () => {
      try {
        const res = await fetch(`${API_URL}/standings/drivers`);
        if (!res.ok) throw new Error(`Failed to refresh drivers: ${res.status}`);
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
        if (!res.ok) throw new Error(`Failed to refresh constructors: ${res.status}`);
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
        const res = await fetch(`${API_URL}/all-events`); // Changed to all-events
        if (!res.ok) throw new Error(`Failed to refresh races: ${res.status}`);
        const data = await res.json();
        setRaces(data);
        
        // Also refresh the next race
        getNextRace();
        
        return true;
      } catch (err) {
        console.error("Failed to refresh races:", err);
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