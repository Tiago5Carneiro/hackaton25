"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import '../styles/login.css';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate a login request
    setTimeout(() => {
      setIsSubmitting(false);
      // Handle successful login here
    }, 1000);
  };

  return (
    <div className="login-container">
      <motion.div 
        className="login-form"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="form-title">Login</h2>
        <form onSubmit={handleSubmit}>
          <motion.div 
            className="input-group"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
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
            transition={{ duration: 0.5, delay: 0.2 }}
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
          <div className="forgot-password">
            <Link href="/forgot-password">Forgot password?</Link>
          </div>
          <button 
            type="submit" 
            className="submit-button"
            disabled={isSubmitting}
          >
            {isSubmitting ? 
              <span className="loading-spinner"></span> : 
              'Login'
            }
          </button>
        </form>
        <div className="auth-redirect">
          <p>Dont have an account? <Link href="/register">Register</Link></p>
        </div>
      </motion.div>
    </div>
  );
}