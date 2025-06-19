import React, { useState } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  IconButton,
  useTheme,
  useMediaQuery,
  Collapse,
} from '@mui/material';
import {
  Dashboard,
  Description,
  Monitoring,
  SmartToy,
  Assessment,
  People,
  Settings,
  ChevronLeft,
  ChevronRight,
  ExpandLess,
  ExpandMore,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useUserRole } from '../../contexts/UserRoleContext';

const DRAWER_WIDTH = 240;
const DRAWER_WIDTH_COLLAPSED = 60;

const navigationItems = [
  {
    text: 'Dashboard',
    icon: <Dashboard />,
    path: '/dashboard',
    permission: null,
  },
  {
    text: 'Documents',
    icon: <Description />,
    path: '/documents',
    permission: 'canViewDocuments',
  },
  {
    text: 'Monitoring',
    icon: <Monitoring />,
    path: '/monitoring',
    permission: 'canViewMonitoring',
  },
  {
    text: 'AI Query',
    icon: <SmartToy />,
    path: '/ai-query',
    permission: 'canUseAIQuery',
  },
  {
    text: 'Compliance',
    icon: <Assessment />,
    path: '/compliance',
    permission: 'canViewCompliance',
  },
];

const adminItems = [
  {
    text: 'User Management',
    icon: <People />,
    path: '/users',
    permission: 'canAccessAdminPanel',
  },
  {
    text: 'System Settings',
    icon: <Settings />,
    path: '/settings',
    permission: 'canAccessAdminPanel',
  },
];

const Sidebar = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const { hasPermission } = useUserRole();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const [collapsed, setCollapsed] = useState(false);
  const [adminExpanded, setAdminExpanded] = useState(false);

  const drawerWidth = collapsed ? DRAWER_WIDTH_COLLAPSED : DRAWER_WIDTH;

  const handleToggleCollapse = () => {
    setCollapsed(!collapsed);
  };

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleAdminToggle = () => {
    setAdminExpanded(!adminExpanded);
  };

  const isActive = (path) => location.pathname === path;

  const hasAdminPermissions = hasPermission('canAccessAdminPanel');

  if (isMobile) {
    return null; // Handle mobile navigation differently
  }

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#006039',
          color: '#ffffff',
          transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
          overflowX: 'hidden',
          mt: 8, // Account for header height
        },
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
        <IconButton onClick={handleToggleCollapse} sx={{ color: '#ffffff' }}>
          {collapsed ? <ChevronRight /> : <ChevronLeft />}
        </IconButton>
      </Box>

      <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.12)' }} />

      <List>
        {navigationItems.map((item) => {
          if (item.permission && !hasPermission(item.permission)) {
            return null;
          }

          return (
            <ListItem key={item.text} disablePadding>
              <ListItemButton
                onClick={() => handleNavigation(item.path)}
                selected={isActive(item.path)}
                sx={{
                  minHeight: 48,
                  justifyContent: collapsed ? 'center' : 'initial',
                  px: 2.5,
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(212, 175, 55, 0.2)',
                    '&:hover': {
                      backgroundColor: 'rgba(212, 175, 55, 0.3)',
                    },
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 0,
                    mr: collapsed ? 'auto' : 3,
                    justifyContent: 'center',
                    color: isActive(item.path) ? '#d4af37' : '#ffffff',
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  sx={{
                    opacity: collapsed ? 0 : 1,
                    color: isActive(item.path) ? '#d4af37' : '#ffffff',
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      {hasAdminPermissions && (
        <>
          <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.12)', my: 1 }} />

          <List>
            <ListItem disablePadding>
              <ListItemButton
                onClick={handleAdminToggle}
                sx={{
                  minHeight: 48,
                  justifyContent: collapsed ? 'center' : 'initial',
                  px: 2.5,
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 0,
                    mr: collapsed ? 'auto' : 3,
                    justifyContent: 'center',
                    color: '#ffffff',
                  }}
                >
                  <Settings />
                </ListItemIcon>
                <ListItemText
                  primary="Administration"
                  sx={{ opacity: collapsed ? 0 : 1, color: '#ffffff' }}
                />
                {!collapsed && (adminExpanded ? <ExpandLess /> : <ExpandMore />)}
              </ListItemButton>
            </ListItem>

            <Collapse in={adminExpanded && !collapsed} timeout="auto" unmountOnExit>
              <List component="div" disablePadding>
                {adminItems.map((item) => (
                  <ListItem key={item.text} disablePadding>
                    <ListItemButton
                      onClick={() => handleNavigation(item.path)}
                      selected={isActive(item.path)}
                      sx={{
                        pl: 4,
                        '&.Mui-selected': {
                          backgroundColor: 'rgba(212, 175, 55, 0.2)',
                        },
                        '&:hover': {
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        },
                      }}
                    >
                      <ListItemIcon sx={{ color: isActive(item.path) ? '#d4af37' : '#ffffff' }}>
                        {item.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={item.text}
                        sx={{ color: isActive(item.path) ? '#d4af37' : '#ffffff' }}
                      />
                    </ListItemButton>
                  </ListItem>
                ))}
              </List>
            </Collapse>
          </List>
        </>
      )}
    </Drawer>
  );
};

export default Sidebar;
