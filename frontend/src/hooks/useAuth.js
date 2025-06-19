import { useState, useCallback } from 'react';
import tailingsIQApi from '../api/tailingsIQApi';

export const useAuth = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const handleError = useCallback((err) => {
    const errorMessage = err.response?.data?.detail || err.message || 'Authentication failed';
    setError(errorMessage);
    console.error('Auth Error:', err);
  }, []);

  const login = useCallback(async (username, password) => {
    setLoading(true);
    setError(null);

    try {
      const response = await tailingsIQApi.auth.login({ username, password });
      return response.data;
    } catch (err) {
      handleError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  const logout = useCallback(async () => {
    setLoading(true);

    try {
      await tailingsIQApi.auth.logout();
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshToken = useCallback(async () => {
    try {
      const response = await tailingsIQApi.auth.refreshToken();
      return response.data;
    } catch (err) {
      handleError(err);
      throw err;
    }
  }, [handleError]);

  const changePassword = useCallback(async (currentPassword, newPassword) => {
    setLoading(true);
    setError(null);

    try {
      const response = await tailingsIQApi.auth.changePassword({
        current_password: currentPassword,
        new_password: newPassword
      });
      return response.data;
    } catch (err) {
      handleError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  const updateProfile = useCallback(async (profileData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await tailingsIQApi.auth.updateProfile(profileData);
      return response.data;
    } catch (err) {
      handleError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  const requestPasswordReset = useCallback(async (email) => {
    setLoading(true);
    setError(null);

    try {
      const response = await tailingsIQApi.auth.requestPasswordReset({ email });
      return response.data;
    } catch (err) {
      handleError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  const resetPassword = useCallback(async (token, newPassword) => {
    setLoading(true);
    setError(null);

    try {
      const response = await tailingsIQApi.auth.resetPassword({
        token,
        new_password: newPassword
      });
      return response.data;
    } catch (err) {
      handleError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  return {
    loading,
    error,
    clearError,
    login,
    logout,
    refreshToken,
    changePassword,
    updateProfile,
    requestPasswordReset,
    resetPassword,
  };
};

export default useAuth;
