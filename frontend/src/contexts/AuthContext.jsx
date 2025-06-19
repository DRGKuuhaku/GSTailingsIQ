import React, { createContext, useContext, useEffect, useState } from 'react';
import { tailingsIQApi } from '../api/tailingsIQApi';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(localStorage.getItem('tailingsiq_token'));

  useEffect(() => {
    // Check if user is already authenticated on app load
    if (token) {
      validateToken();
    } else {
      setLoading(false);
    }
  }, [token]);

  const validateToken = async () => {
    try {
      // Set the token in the API client
      tailingsIQApi.defaults.headers.common['Authorization'] = `Bearer ${token}`;

      // Validate token by fetching current user
      const response = await tailingsIQApi.get('/auth/me');
      setCurrentUser(response.data);
    } catch (error) {
      console.error('Token validation failed:', error);
      // Clear invalid token
      localStorage.removeItem('tailingsiq_token');
      setToken(null);
      setCurrentUser(null);
      delete tailingsIQApi.defaults.headers.common['Authorization'];
    } finally {
      setLoading(false);
    }
  };

  const login = async (username, password) => {
    try {
      const response = await tailingsIQApi.post('/auth/login', {
        username,
        password
      });

      const { access_token, user } = response.data;

      // Store token
      localStorage.setItem('tailingsiq_token', access_token);
      setToken(access_token);

      // Set token in API client
      tailingsIQApi.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

      // Set current user
      setCurrentUser(user);

      return { success: true };
    } catch (error) {
      console.error('Login failed:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed'
      };
    }
  };

  const logout = async () => {
    try {
      // Call logout endpoint to invalidate token on server
      if (token) {
        await tailingsIQApi.post('/auth/logout');
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local state regardless of server response
      localStorage.removeItem('tailingsiq_token');
      setToken(null);
      setCurrentUser(null);
      delete tailingsIQApi.defaults.headers.common['Authorization'];
    }
  };

  const updateProfile = async (profileData) => {
    try {
      const response = await tailingsIQApi.put('/auth/profile', profileData);
      setCurrentUser(response.data);
      return { success: true };
    } catch (error) {
      console.error('Profile update failed:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Profile update failed'
      };
    }
  };

  const changePassword = async (currentPassword, newPassword) => {
    try {
      await tailingsIQApi.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword
      });
      return { success: true };
    } catch (error) {
      console.error('Password change failed:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Password change failed'
      };
    }
  };

  const requestPasswordReset = async (email) => {
    try {
      await tailingsIQApi.post('/auth/request-password-reset', { email });
      return { success: true };
    } catch (error) {
      console.error('Password reset request failed:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Password reset request failed'
      };
    }
  };

  const resetPassword = async (token, newPassword) => {
    try {
      await tailingsIQApi.post('/auth/reset-password', {
        token,
        new_password: newPassword
      });
      return { success: true };
    } catch (error) {
      console.error('Password reset failed:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Password reset failed'
      };
    }
  };

  const value = {
    currentUser,
    loading,
    login,
    logout,
    updateProfile,
    changePassword,
    requestPasswordReset,
    resetPassword,
    isAuthenticated: !!currentUser,
    token
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
