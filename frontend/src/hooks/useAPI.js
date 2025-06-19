import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import tailingsIQApi from '../api/tailingsIQApi';

export const useAPI = () => {
  const queryClient = useQueryClient();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const handleError = useCallback((err) => {
    const errorMessage = err.response?.data?.detail || err.message || 'An unexpected error occurred';
    setError(errorMessage);
    console.error('API Error:', err);
  }, []);

  // Generic API call hook
  const apiCall = useCallback(async (apiFunction, ...args) => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiFunction(...args);
      return result;
    } catch (err) {
      handleError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  // Documents hooks
  const useDocuments = (params = {}) => {
    return useQuery(
      ['documents', params],
      () => tailingsIQApi.documents.getDocuments(params),
      {
        onError: handleError,
        staleTime: 5 * 60 * 1000, // 5 minutes
      }
    );
  };

  const useUploadDocument = () => {
    return useMutation(
      ({ file, metadata }) => tailingsIQApi.documents.uploadDocument(file, metadata),
      {
        onSuccess: () => {
          queryClient.invalidateQueries(['documents']);
        },
        onError: handleError,
      }
    );
  };

  // Monitoring hooks
  const useMonitoringStations = (facilityId) => {
    return useQuery(
      ['monitoring-stations', facilityId],
      () => tailingsIQApi.monitoring.getStations(facilityId),
      {
        enabled: !!facilityId,
        onError: handleError,
      }
    );
  };

  const useMonitoringReadings = (stationId, params = {}) => {
    return useQuery(
      ['monitoring-readings', stationId, params],
      () => tailingsIQApi.monitoring.getReadings(stationId, params),
      {
        enabled: !!stationId,
        onError: handleError,
        refetchInterval: 30000, // Refetch every 30 seconds for real-time data
      }
    );
  };

  // AI Query hooks
  const useAIQuery = () => {
    return useMutation(
      (query) => tailingsIQApi.query.submitQuery(query),
      {
        onError: handleError,
      }
    );
  };

  // Compliance hooks
  const useComplianceRequirements = (facilityId) => {
    return useQuery(
      ['compliance-requirements', facilityId],
      () => tailingsIQApi.compliance.getRequirements(facilityId),
      {
        enabled: !!facilityId,
        onError: handleError,
      }
    );
  };

  const useComplianceAssessments = (facilityId) => {
    return useQuery(
      ['compliance-assessments', facilityId],
      () => tailingsIQApi.compliance.getAssessments(facilityId),
      {
        enabled: !!facilityId,
        onError: handleError,
      }
    );
  };

  // User management hooks
  const useUsers = () => {
    return useQuery(
      ['users'],
      () => tailingsIQApi.users.getUsers(),
      {
        onError: handleError,
      }
    );
  };

  const useCreateUser = () => {
    return useMutation(
      (userData) => tailingsIQApi.users.createUser(userData),
      {
        onSuccess: () => {
          queryClient.invalidateQueries(['users']);
        },
        onError: handleError,
      }
    );
  };

  const useUpdateUser = () => {
    return useMutation(
      ({ id, userData }) => tailingsIQApi.users.updateUser(id, userData),
      {
        onSuccess: () => {
          queryClient.invalidateQueries(['users']);
        },
        onError: handleError,
      }
    );
  };

  const useDeleteUser = () => {
    return useMutation(
      (userId) => tailingsIQApi.users.deleteUser(userId),
      {
        onSuccess: () => {
          queryClient.invalidateQueries(['users']);
        },
        onError: handleError,
      }
    );
  };

  // Synthetic data hooks
  const useSyntheticDatasets = () => {
    return useQuery(
      ['synthetic-datasets'],
      () => tailingsIQApi.syntheticData.getDatasets(),
      {
        onError: handleError,
      }
    );
  };

  const useGenerateSyntheticData = () => {
    return useMutation(
      (params) => tailingsIQApi.syntheticData.generateData(params),
      {
        onSuccess: () => {
          queryClient.invalidateQueries(['synthetic-datasets']);
        },
        onError: handleError,
      }
    );
  };

  return {
    loading,
    error,
    clearError,
    apiCall,

    // Document hooks
    useDocuments,
    useUploadDocument,

    // Monitoring hooks
    useMonitoringStations,
    useMonitoringReadings,

    // AI Query hooks
    useAIQuery,

    // Compliance hooks
    useComplianceRequirements,
    useComplianceAssessments,

    // User management hooks
    useUsers,
    useCreateUser,
    useUpdateUser,
    useDeleteUser,

    // Synthetic data hooks
    useSyntheticDatasets,
    useGenerateSyntheticData,
  };
};

export default useAPI;
