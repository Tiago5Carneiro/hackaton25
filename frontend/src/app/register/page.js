"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Image from 'next/image';
import '../styles/register.css';

export default function RegisterPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [favoriteDriver, setFavoriteDriver] = useState('');
  const [favoriteTeam, setFavoriteTeam] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [passwordError, setPasswordError] = useState('');

  // F1 drivers data for 2025 (example data)
  const drivers = [
    { id: 'VER', name: 'Max Verstappen', team: 'Red Bull Racing' },
    { id: 'HAM', name: 'Lewis Hamilton', team: 'Ferrari' },
    { id: 'NOR', name: 'Lando Norris', team: 'McLaren' },
    { id: 'LEC', name: 'Charles Leclerc', team: 'Ferrari' },
    { id: 'RUS', name: 'George Russell', team: 'Mercedes' },
    { id: 'PIA', name: 'Oscar Piastri', team: 'McLaren' },
    { id: 'SAI', name: 'Carlos Sainz', team: 'Williams' },
    { id: 'PER', name: 'Sergio Perez', team: 'Red Bull Racing' },
    { id: 'ALO', name: 'Fernando Alonso', team: 'Aston Martin' },
    { id: 'STR', name: 'Lance Stroll', team: 'Aston Martin' },
    { id: 'GAS', name: 'Pierre Gasly', team: 'Alpine' },
    { id: 'OCO', name: 'Esteban Ocon', team: 'Alpine' },
    { id: 'ALB', name: 'Alexander Albon', team: 'Williams' },
    { id: 'TSU', name: 'Yuki Tsunoda', team: 'VCARB' },
    { id: 'LAW', name: 'Liam Lawson', team: 'VCARB' },
    { id: 'ZHO', name: 'Zhou Guanyu', team: 'Sauber' },
    { id: 'BOT', name: 'Valtteri Bottas', team: 'Sauber' },
    { id: 'BEA', name: 'Oliver Bearman', team: 'Haas F1 Team' },
    { id: 'MAG', name: 'Kevin Magnussen', team: 'Haas F1 Team' },
    { id: 'COA', name: 'Jack Coates', team: 'Mercedes' },
  ];

  // F1 teams data with external logo URLs like in standings page
  const teams = [
    { 
      id: 'RBR', 
      name: 'Red Bull Racing', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/red%20bull.jpg"
    },
    { 
      id: 'FER', 
      name: 'Ferrari', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/ferrari.jpg" 
    },
    { 
      id: 'MCL', 
      name: 'McLaren', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/mclaren.jpg" 
    },
    { 
      id: 'MER', 
      name: 'Mercedes', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/mercedes.jpg" 
    },
    { 
      id: 'AST', 
      name: 'Aston Martin', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/aston%20martin.jpg" 
    },
    { 
      id: 'ALP', 
      name: 'Alpine', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/alpine.jpg" 
    },
    { 
      id: 'WIL', 
      name: 'Williams', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/williams.jpg" 
    },
    { 
      id: 'VCB', 
      name: 'VCARB', 
      logo: "https://media.formula1.com/content/dam/fom-website/teams/2024/rb-logo.jpg.img.640.medium.jpg" 
    },
    { 
      id: 'SAU', 
      name: 'Sauber', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/alfa%20romeo.jpg" 
    },
    { 
      id: 'HAA', 
      name: 'Haas F1 Team', 
      logo: "https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/haas.jpg" 
    },
  ];

  const validatePassword = () => {
    if (password !== confirmPassword) {
      setPasswordError('Passwords do not match');
      return false;
    }
    setPasswordError('');
    return true;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validatePassword()) return;
    
    setIsSubmitting(true);
    
    // Simulate a registration request
    setTimeout(() => {
      setIsSubmitting(false);
      // Handle successful registration here
      console.log({
        name,
        email,
        password,
        favoriteDriver,
        favoriteTeam
      });
    }, 1500);
  };

  return (
    <div className="register-container">
      <motion.div 
        className="register-form"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="form-title">Create Account</h2>
        <form onSubmit={handleSubmit}>
          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <label htmlFor="name">Full Name</label>
            <input 
              type="text" 
              id="name" 
              value={name} 
              onChange={(e) => setName(e.target.value)} 
              required 
            />
          </motion.div>
          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <label htmlFor="email">Email</label>
            <input 
              type="email" 
              id="email" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              required 
            />
          </motion.div>
          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <label htmlFor="password">Password</label>
            <input 
              type="password" 
              id="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required 
            />
          </motion.div>
          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input 
              type="password" 
              id="confirmPassword" 
              value={confirmPassword} 
              onChange={(e) => setConfirmPassword(e.target.value)} 
              onBlur={validatePassword}
              required 
            />
            {passwordError && <span className="error-message">{passwordError}</span>}
          </motion.div>

          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <label htmlFor="favoriteDriver">Favorite Driver</label>
            <select
              id="favoriteDriver"
              value={favoriteDriver}
              onChange={(e) => setFavoriteDriver(e.target.value)}
              className="select-input"
            >
              <option value="">Select your favorite driver</option>
              {drivers.map((driver) => (
                <option key={driver.id} value={driver.id}>
                  {driver.name} ({driver.team})
                </option>
              ))}
            </select>
          </motion.div>

          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <label htmlFor="favoriteTeam">Favorite Team</label>
            <div className="team-select-container">
              <select
                id="favoriteTeam"
                value={favoriteTeam}
                onChange={(e) => setFavoriteTeam(e.target.value)}
                className="select-input team-select"
              >
                <option value="">Select your favorite team</option>
                {teams.map((team) => (
                  <option key={team.id} value={team.id}>
                    {team.name}
                  </option>
                ))}
              </select>
              {favoriteTeam && (
                <div className="team-logo-container">
                  <Image 
                    src={teams.find(team => team.id === favoriteTeam)?.logo || 'https://media.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/team-logo-placeholder.jpg'}
                    alt={`${teams.find(team => team.id === favoriteTeam)?.name} logo`}
                    width={40}
                    height={40}
                    className="team-logo"
                    unoptimized={true}
                    priority={true} /* Add priority to load faster */
                    style={{ 
                      objectFit: 'contain',
                      maxWidth: '100%',
                      maxHeight: '100%'
                    }}
                  />
                </div>
              )}
            </div>
          </motion.div>

          <button 
            type="submit" 
            className="submit-button"
            disabled={isSubmitting}
          >
            {isSubmitting ? 
              <span className="loading-spinner"></span> : 
              'Register'
            }
          </button>
        </form>
        <div className="auth-redirect">
          <p>Already have an account? <Link href="/login">Login</Link></p>
        </div>
      </motion.div>
    </div>
  );
}