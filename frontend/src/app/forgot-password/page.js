"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import '../styles/forgotPassword.css';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate a password reset request
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSuccess(true);
      // In a real app, this would send a request to your API
    }, 1500);
  };

  return (
    <div className="forgot-password-container">
      <motion.div 
        className="forgot-password-form"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="form-title">Reset Password</h2>
        
        {!isSuccess ? (
          <>
            <p className="form-description">
              Enter your email address below and we ll send you instructions to reset your password.
            </p>
            
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
                  placeholder="Enter your registered email address"
                />
              </motion.div>
              
              <button 
                type="submit" 
                className="submit-button"
                disabled={isSubmitting}
              >
                {isSubmitting ? 
                  <span className="loading-spinner"></span> : 
                  'Send Reset Link'
                }
              </button>
            </form>
          </>
        ) : (
          <motion.div 
            className="success-message"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="success-icon">✓</div>
            <h3>Password Reset Email Sent</h3>
            <p>
              We ve sent an email to <strong>{email}</strong> with instructions to reset your password.
            </p>
            <p className="info-note">
              If you dont see it in your inbox, please check your spam folder.
            </p>
          </motion.div>
        )}
        
        <div className="auth-redirect">
          <p>
            Remember your password? <Link href="/login">Back to Login</Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
}