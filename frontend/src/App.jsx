import React, { Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';

import { useAuth } from './contexts/AuthContext';
import { useUserRole } from './contexts/UserRoleContext';
import Header from './components/common/Header';
import Sidebar from './components/common/Sidebar';
import LoadingSpinner from './components/common/LoadingSpinner';

// Lazy load pages for better performance
const DashboardPage = React.lazy(() => import('./pages/DashboardPage'));
const DocumentsPage = React.lazy(() => import('./pages/DocumentsPage'));
const MonitoringPage = React.lazy(() => import('./pages/MonitoringPage'));
const AIQueryPage = React.lazy(() => import('./pages/AIQueryPage'));
const CompliancePage = React.lazy(() => import('./pages/CompliancePage'));
const UsersPage = React.lazy(() => import('./pages/UsersPage'));
const LoginPage = React.lazy(() => import('./pages/LoginPage'));

const ProtectedRoute = ({ children, requiredPermission }) => {
  const { currentUser, loading } = useAuth();
  const { hasPermission } = useUserRole();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!currentUser) {
    return <Navigate to="/login" replace />;
  }

  if (requiredPermission && !hasPermission(requiredPermission)) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

const AppLayout = ({ children }) => {
  const { currentUser } = useAuth();

  if (!currentUser) {
    return children;
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Header />
      <Sidebar />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          mt: 8, // Account for header height
          ml: { xs: 0, md: 30 }, // Account for sidebar width
          p: 3,
          backgroundColor: '#f8f9fa',
          minHeight: '100vh',
        }}
      >
        <Suspense fallback={<LoadingSpinner />}>
          {children}
        </Suspense>
      </Box>
    </Box>
  );
};

function App() {
  return (
    <AppLayout>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<LoginPage />} />

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />

        <Route
          path="/documents"
          element={
            <ProtectedRoute requiredPermission="canViewDocuments">
              <DocumentsPage />
            </ProtectedRoute>
          }
        />

        <Route
          path="/monitoring"
          element={
            <ProtectedRoute requiredPermission="canViewMonitoring">
              <MonitoringPage />
            </ProtectedRoute>
          }
        />

        <Route
          path="/ai-query"
          element={
            <ProtectedRoute requiredPermission="canUseAIQuery">
              <AIQueryPage />
            </ProtectedRoute>
          }
        />

        <Route
          path="/compliance"
          element={
            <ProtectedRoute requiredPermission="canViewCompliance">
              <CompliancePage />
            </ProtectedRoute>
          }
        />

        <Route
          path="/users"
          element={
            <ProtectedRoute requiredPermission="canAccessAdminPanel">
              <UsersPage />
            </ProtectedRoute>
          }
        />

        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />

        {/* Catch all - redirect to dashboard */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </AppLayout>
  );
}

export default App;
