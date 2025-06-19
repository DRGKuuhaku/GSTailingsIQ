import React from 'react';
import {
  Box, Typography, Grid, Card, CardContent, Button,
  List, ListItem, ListItemIcon, ListItemText
} from '@mui/material';
import {
  Storage, SmartToy, BarChart, Link, CheckCircle, Person,
  PlayArrow, TrendingUp, Assignment, LocationOn, Phone, Email
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import TailingsIQLogo from '../components/common/TailingsIQLogo';

const DashboardPage = () => {
  const navigate = useNavigate();

  return (
    <Box>
      {/* Hero Section */}
      <Box 
        sx={{ 
          background: 'linear-gradient(135deg, #006039 0%, #004d2e 100%)',
          color: '#ffffff',
          textAlign: 'center',
          py: { xs: 6, md: 8 },
          borderRadius: 2,
          mb: 4
        }}
      >
        <Typography variant="h6" sx={{ 
          fontSize: '1rem',
          fontWeight: 300,
          letterSpacing: '2px',
          textTransform: 'uppercase',
          mb: 2,
          opacity: 0.9
        }}>
          TailingsIQ
        </Typography>
        <Typography variant="h3" component="h1" sx={{ 
          fontWeight: 700,
          mb: 2,
          fontSize: { xs: '2rem', md: '3rem' }
        }}>
          AI-Driven Knowledge Base for TSF Management
        </Typography>
        <Typography sx={{ 
          fontSize: '1.1rem',
          fontWeight: 400,
          fontStyle: 'italic',
          maxWidth: 600,
          mx: 'auto',
          mb: 4,
          opacity: 0.9,
          lineHeight: 1.6
        }}>
          Revolutionize how you manage risks, monitor performance, and ensure compliance with industry standards
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
          <Button 
            variant="contained" 
            size="large"
            onClick={() => navigate('/monitoring')}
            sx={{ 
              px: 4, py: 1.5,
              backgroundColor: '#ffffff',
              color: '#006039',
              fontWeight: 600,
              '&:hover': { 
                backgroundColor: '#f0f0f0',
                transform: 'translateY(-2px)'
              }
            }}
          >
            View Demo
          </Button>
          <Button 
            variant="contained" 
            size="large"
            onClick={() => navigate('/ai-query')}
            sx={{ 
              px: 4, py: 1.5,
              backgroundColor: '#d4af37',
              color: '#006039',
              fontWeight: 600,
              '&:hover': { 
                backgroundColor: '#c09c30',
                transform: 'translateY(-2px)'
              }
            }}
          >
            Try AI Chat
          </Button>
        </Box>
      </Box>

      {/* Key Features Section */}
      <Typography variant="h4" sx={{ 
        fontWeight: 700,
        mb: 4,
        textAlign: 'center',
        position: 'relative',
        '&:after': {
          content: '""',
          position: 'absolute',
          left: '50%',
          bottom: -10,
          width: 50,
          height: 3,
          backgroundColor: '#006039',
          transform: 'translateX(-50%)'
        }
      }}>
        Key Features
      </Typography>

      <Grid container spacing={4} sx={{ mb: 6 }}>
        {[
          {
            icon: <Storage sx={{ fontSize: 48, color: '#006039' }} />,
            title: 'Secure Document Storage',
            description: 'AWS S3 integration for scalable, secure storage of all TSF-related documents with automated metadata extraction.'
          },
          {
            icon: <SmartToy sx={{ fontSize: 48, color: '#006039' }} />,
            title: 'AI Query Engine',
            description: 'Natural language processing to understand and respond to user queries with context-aware answers and visualizations.'
          },
          {
            icon: <BarChart sx={{ fontSize: 48, color: '#006039' }} />,
            title: 'Data Visualization',
            description: 'Interactive dashboards for monitoring data, compliance status, and risk assessments with real-time updates.'
          },
          {
            icon: <Link sx={{ fontSize: 48, color: '#006039' }} />,
            title: 'Integration Framework',
            description: 'Connections to external data sources including weather stations, monitoring systems, and regulatory repositories.'
          },
          {
            icon: <CheckCircle sx={{ fontSize: 48, color: '#006039' }} />,
            title: 'Compliance Monitoring',
            description: 'Automated tracking of GISTM requirements and other regulatory standards with compliance status dashboards.'
          },
          {
            icon: <Person sx={{ fontSize: 48, color: '#006039' }} />,
            title: 'User-Friendly Interface',
            description: 'Intuitive web-based interface for document upload, search, and querying with role-based access control.'
          }
        ].map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card sx={{ 
              height: '100%',
              border: '1px solid #dee2e6',
              transition: 'all 0.3s ease',
              '&:hover': {
                boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)',
                transform: 'translateY(-5px)'
              }
            }}>
              <CardContent sx={{ p: 4, textAlign: 'center' }}>
                <Box sx={{ mb: 3 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Intelligent Dashboard Section */}
      <Box sx={{ backgroundColor: '#f8f9fa', borderRadius: 2, p: 4, mb: 6 }}>
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h4" sx={{ 
              fontWeight: 700,
              mb: 3,
              position: 'relative',
              '&:after': {
                content: '""',
                position: 'absolute',
                left: 0,
                bottom: -10,
                width: 50,
                height: 3,
                backgroundColor: '#006039'
              }
            }}>
              Intelligent Dashboard
            </Typography>
            <Typography variant="h6" sx={{ mb: 3, color: 'text.secondary' }}>
              Comprehensive visualization of all your tailings facility data in one place
            </Typography>
            <List>
              {[
                { icon: <PlayArrow sx={{ color: '#006039' }} />, text: 'Real-time monitoring data visualization' },
                { icon: <CheckCircle sx={{ color: '#006039' }} />, text: 'Compliance status tracking with alerts' },
                { icon: <TrendingUp sx={{ color: '#006039' }} />, text: 'Risk assessment and management tools' },
                { icon: <Assignment sx={{ color: '#006039' }} />, text: 'Document repository with intelligent search' },
                { icon: <Person sx={{ color: '#006039' }} />, text: 'Customizable views for different stakeholders' }
              ].map((item, index) => (
                <ListItem key={index} sx={{ px: 0 }}>
                  <ListItemIcon>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItem>
              ))}
            </List>
            <Button 
              variant="contained" 
              size="large"
              onClick={() => navigate('/ai-query')}
              sx={{ 
                mt: 3, px: 4, py: 1.5,
                backgroundColor: '#d4af37',
                color: '#006039',
                fontWeight: 600,
                '&:hover': { 
                  backgroundColor: '#c09c30',
                  transform: 'translateY(-2px)'
                }
              }}
            >
              Try the AI Chat
            </Button>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ 
              textAlign: 'center',
              p: 4,
              backgroundColor: '#ffffff',
              borderRadius: 2,
              border: '2px dashed #dee2e6'
            }}>
              <BarChart sx={{ fontSize: 120, color: '#006039', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                Interactive Dashboard Preview
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Real-time data visualization and analytics
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Footer Section */}
      <Box sx={{ 
        backgroundColor: '#006039',
        color: '#ffffff',
        borderRadius: 2,
        p: 4
      }}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <TailingsIQLogo sx={{ mb: 2 }} />
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
              Revolutionizing tailings storage facility management with AI-driven knowledge and analytics.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Quick Links
            </Typography>
            <List sx={{ p: 0 }}>
              {[
                { text: 'Features', path: '/dashboard' },
                { text: 'AI Query', path: '/ai-query' },
                { text: 'Compliance', path: '/compliance' },
                { text: 'Monitoring', path: '/monitoring' }
              ].map((link, index) => (
                <ListItem key={index} sx={{ p: 0 }}>
                  <Button 
                    color="inherit" 
                    sx={{ color: 'rgba(255, 255, 255, 0.8)', justifyContent: 'flex-start' }}
                    onClick={() => navigate(link.path)}
                  >
                    {link.text}
                  </Button>
                </ListItem>
              ))}
            </List>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Connect With Us
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 1 }}>
              <LocationOn sx={{ verticalAlign: 'middle', mr: 1, fontSize: 'small' }} />
              125 St Georges Tce, Level 11, Perth, WA 6000
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 1 }}>
              <Phone sx={{ verticalAlign: 'middle', mr: 1, fontSize: 'small' }} />
              1300 852 216
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
              <Email sx={{ verticalAlign: 'middle', mr: 1, fontSize: 'small' }} />
              tailingsiq@geotesta.com.au
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default DashboardPage;
