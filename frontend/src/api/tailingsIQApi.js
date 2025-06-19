import axios from 'axios';

// Create API client
export const tailingsIQApi = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
tailingsIQApi.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('tailingsiq_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
tailingsIQApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('tailingsiq_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// User Management API
export const users = {
  // Get all users
  getUsers: async (params = {}) => {
    const response = await tailingsIQApi.get('/admin/users', { params });
    return response.data;
  },

  // Get user by ID
  getUser: async (userId) => {
    const response = await tailingsIQApi.get(`/admin/users/${userId}`);
    return response.data;
  },

  // Create new user
  createUser: async (userData) => {
    const response = await tailingsIQApi.post('/admin/users', userData);
    return response.data;
  },

  // Update user
  updateUser: async (userId, userData) => {
    const response = await tailingsIQApi.put(`/admin/users/${userId}`, userData);
    return response.data;
  },

  // Delete user
  deleteUser: async (userId) => {
    const response = await tailingsIQApi.delete(`/admin/users/${userId}`);
    return response.data;
  },

  // Reset user password
  resetPassword: async (userId) => {
    const response = await tailingsIQApi.post(`/admin/users/${userId}/reset-password`);
    return response.data;
  }
};

// Authentication API
export const auth = {
  // Login
  login: async (username, password) => {
    const response = await tailingsIQApi.post('/auth/login', {
      username,
      password
    });
    return response.data;
  },

  // Logout
  logout: async () => {
    const response = await tailingsIQApi.post('/auth/logout');
    return response.data;
  },

  // Get current user
  getCurrentUser: async () => {
    const response = await tailingsIQApi.get('/auth/me');
    return response.data;
  },

  // Update profile
  updateProfile: async (profileData) => {
    const response = await tailingsIQApi.put('/auth/profile', profileData);
    return response.data;
  },

  // Change password
  changePassword: async (currentPassword, newPassword) => {
    const response = await tailingsIQApi.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword
    });
    return response.data;
  },

  // Request password reset
  requestPasswordReset: async (email) => {
    const response = await tailingsIQApi.post('/auth/request-password-reset', {
      email
    });
    return response.data;
  },

  // Reset password
  resetPassword: async (token, newPassword) => {
    const response = await tailingsIQApi.post('/auth/reset-password', {
      token,
      new_password: newPassword
    });
    return response.data;
  }
};

// Documents API
export const documents = {
  // Get all documents
  getDocuments: async (params = {}) => {
    const response = await tailingsIQApi.get('/documents', { params });
    return response.data;
  },

  // Upload document
  uploadDocument: async (formData) => {
    const response = await tailingsIQApi.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Delete document
  deleteDocument: async (documentId) => {
    const response = await tailingsIQApi.delete(`/documents/${documentId}`);
    return response.data;
  }
};

// Monitoring API
export const monitoring = {
  // Get monitoring data
  getMonitoringData: async (params = {}) => {
    const response = await tailingsIQApi.get('/monitoring', { params });
    return response.data;
  },

  // Get facility status
  getFacilityStatus: async (facilityId) => {
    const response = await tailingsIQApi.get(`/monitoring/facility/${facilityId}/status`);
    return response.data;
  }
};

// AI Query API
export const aiQuery = {
  // Submit query
  submitQuery: async (query, context = {}) => {
    const response = await tailingsIQApi.post('/query/submit', {
      query,
      context
    });
    return response.data;
  },

  // Get query history
  getQueryHistory: async (params = {}) => {
    const response = await tailingsIQApi.get('/query/history', { params });
    return response.data;
  }
};

// Compliance API
export const compliance = {
  // Get compliance data
  getComplianceData: async (params = {}) => {
    const response = await tailingsIQApi.get('/compliance', { params });
    return response.data;
  },

  // Update compliance status
  updateComplianceStatus: async (complianceId, status) => {
    const response = await tailingsIQApi.put(`/compliance/${complianceId}/status`, {
      status
    });
    return response.data;
  }
};

// Synthetic Data API
export const syntheticData = {
  // Get datasets
  getDatasets: async (params = {}) => {
    const response = await tailingsIQApi.get('/synthetic-data/datasets', { params });
    return response.data;
  },

  // Generate synthetic data
  generateData: async (dataType, count, options = {}) => {
    const response = await tailingsIQApi.post('/synthetic-data/generate', {
      data_type: dataType,
      count,
      ...options
    });
    return response.data;
  },

  // Preview synthetic data
  previewData: async (dataType, count = 10) => {
    const response = await tailingsIQApi.get(`/synthetic-data/preview/${dataType}`, {
      params: { count }
    });
    return response.data;
  },

  // Export dataset
  exportDataset: async (datasetId, format = 'json') => {
    const response = await tailingsIQApi.get(`/synthetic-data/export/${datasetId}`, {
      params: { format }
    });
    return response.data;
  }
};

export default tailingsIQApi;
