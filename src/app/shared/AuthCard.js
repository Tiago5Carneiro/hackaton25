import React from 'react';
import '../styles/authStyles.css';

const AuthCard = ({ title, children }) => {
  return (
    <div className="auth-card">
      <h2 className="auth-card-title">{title}</h2>
      <div className="auth-card-content">
        {children}
      </div>
    </div>
  );
};

export default AuthCard;