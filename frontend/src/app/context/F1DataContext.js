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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Pre-fetch all data when the app loads
  useEffect(() => {
    async function fetchAllData() {
      setLoading(true);
      try {
        // Fetch data in parallel for better performance
        const [driversResponse, constructorsResponse] = await Promise.all([
          fetch(`${API_URL}/standings/drivers`),
          fetch(`${API_URL}/standings/constructors`)
        ]);

        // Check for errors
        if (!driversResponse.ok) throw new Error('Failed to fetch drivers');
        if (!constructorsResponse.ok) throw new Error('Failed to fetch constructors');

        console.log('Drivers Response:', driversResponse);

        // Parse responses
        const driversData = await driversResponse.json();
        const constructorsData = await constructorsResponse.json();

        console.log('Drivers Data:', driversData);

        // Update state with fetched data
        setDrivers(driversData);
        setConstructors(constructorsData);
        setError(null);
      } catch (err) {
        console.error('Error pre-loading F1 data:', err);
        setError(err.message);
        // Keep previous data if available, otherwise set empty arrays
        setDrivers(prev => prev.length > 0 ? prev : []);
        setConstructors(prev => prev.length > 0 ? prev : []);
      } finally {
        setLoading(false);
      }
    }

    fetchAllData();
  }, []);

  // Value to be provided to consumers
  const value = {
    drivers,
    constructors,
    loading,
    error,
    // Add refresh methods if needed
    refreshDrivers: async () => {
      try {
        const res = await fetch(`${API_URL}/standings/drivers`);
        if (!res.ok) throw new Error('Failed to refresh drivers');
        setDrivers(await res.json());
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
        setConstructors(await res.json());
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