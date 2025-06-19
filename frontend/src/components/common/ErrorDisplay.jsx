import React from 'react';
import { 
  Alert, 
  AlertTitle, 
  Box, 
  Button, 
  Typography,
  Paper 
} from '@mui/material';
import { Refresh, Home } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const ErrorDisplay = ({ 
  error, 
  title = 'An Error Occurred',
  message,
  showHomeButton = true,
  showRefreshButton = true,
  onRetry 
}) => {
  const navigate = useNavigate();

  const errorMessage = message || error?.message || 'Something went wrong. Please try again.';

  const handleRetry = () => {
    if (onRetry) {
      onRetry();
    } else {
      window.location.reload();
    }
  };

  const handleGoHome = () => {
    navigate('/dashboard');
  };

  return (
    <Box 
      display="flex" 
      justifyContent="center" 
      alignItems="center" 
      minHeight={400}
      p={3}
    >
      <Paper elevation={2} sx={{ p: 4, maxWidth: 600, width: '100%' }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>{title}</AlertTitle>
          {errorMessage}
        </Alert>

        {error?.stack && process.env.NODE_ENV === 'development' && (
          <Box sx={{ mt: 2, mb: 3 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Stack Trace (Development Only):
            </Typography>
            <Box
              component="pre"
              sx={{
                fontSize: '0.75rem',
                backgroundColor: '#f5f5f5',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                maxHeight: 200,
              }}
            >
              {error.stack}
            </Box>
          </Box>
        )}

        <Box display="flex" gap={2} justifyContent="center">
          {showRefreshButton && (
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleRetry}
            >
              Try Again
            </Button>
          )}

          {showHomeButton && (
            <Button
              variant="contained"
              startIcon={<Home />}
              onClick={handleGoHome}
            >
              Go to Dashboard
            </Button>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default ErrorDisplay;
