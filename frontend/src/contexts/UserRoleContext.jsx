import React, { createContext, useContext, useEffect, useState } from 'react';
import { useAuth } from './AuthContext';

// User roles enum
export const USER_ROLES = {
  SUPER_ADMIN: 'super_admin',
  ADMIN: 'admin',
  ENGINEER_OF_RECORD: 'engineer_of_record',
  TSF_OPERATOR: 'tsf_operator',
  REGULATOR: 'regulator',
  MANAGEMENT: 'management',
  CONSULTANT: 'consultant',
  VIEWER: 'viewer'
};

// Role hierarchy for permission checking
const ROLE_HIERARCHY = {
  [USER_ROLES.SUPER_ADMIN]: 8,
  [USER_ROLES.ADMIN]: 7,
  [USER_ROLES.ENGINEER_OF_RECORD]: 6,
  [USER_ROLES.MANAGEMENT]: 5,
  [USER_ROLES.TSF_OPERATOR]: 4,
  [USER_ROLES.REGULATOR]: 3,
  [USER_ROLES.CONSULTANT]: 2,
  [USER_ROLES.VIEWER]: 1
};

// Permission definitions
const PERMISSIONS = {
  // Admin permissions
  canAccessAdminPanel: [USER_ROLES.SUPER_ADMIN, USER_ROLES.ADMIN],
  canManageUsers: [USER_ROLES.SUPER_ADMIN, USER_ROLES.ADMIN],
  canDeleteUsers: [USER_ROLES.SUPER_ADMIN],
  canResetPasswords: [USER_ROLES.SUPER_ADMIN, USER_ROLES.ADMIN],

  // Document permissions
  canUploadDocuments: [
    USER_ROLES.SUPER_ADMIN, 
    USER_ROLES.ADMIN, 
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.TSF_OPERATOR,
    USER_ROLES.MANAGEMENT
  ],
  canDeleteDocuments: [
    USER_ROLES.SUPER_ADMIN, 
    USER_ROLES.ADMIN, 
    USER_ROLES.ENGINEER_OF_RECORD
  ],
  canViewAllDocuments: [
    USER_ROLES.SUPER_ADMIN, 
    USER_ROLES.ADMIN, 
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.REGULATOR,
    USER_ROLES.MANAGEMENT
  ],

  // Monitoring permissions
  canViewMonitoring: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.TSF_OPERATOR,
    USER_ROLES.REGULATOR,
    USER_ROLES.MANAGEMENT,
    USER_ROLES.CONSULTANT,
    USER_ROLES.VIEWER
  ],
  canEditMonitoring: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.TSF_OPERATOR
  ],

  // Compliance permissions
  canViewCompliance: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.REGULATOR,
    USER_ROLES.MANAGEMENT
  ],
  canEditCompliance: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD
  ],

  // AI Query permissions
  canUseAIQuery: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.TSF_OPERATOR,
    USER_ROLES.REGULATOR,
    USER_ROLES.MANAGEMENT,
    USER_ROLES.CONSULTANT
  ],

  // Synthetic data permissions
  canGenerateSyntheticData: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD
  ],
  canViewSyntheticData: [
    USER_ROLES.SUPER_ADMIN,
    USER_ROLES.ADMIN,
    USER_ROLES.ENGINEER_OF_RECORD,
    USER_ROLES.TSF_OPERATOR
  ]
};

const UserRoleContext = createContext();

export const UserRoleProvider = ({ children }) => {
  const { currentUser } = useAuth();
  const [userRole, setUserRole] = useState(null);
  const [roleLevel, setRoleLevel] = useState(0);

  useEffect(() => {
    if (currentUser?.role) {
      setUserRole(currentUser.role);
      setRoleLevel(ROLE_HIERARCHY[currentUser.role] || 0);
    } else {
      setUserRole(null);
      setRoleLevel(0);
    }
  }, [currentUser]);

  // Check if user has a specific permission
  const hasPermission = (permission) => {
    if (!userRole || !PERMISSIONS[permission]) {
      return false;
    }
    return PERMISSIONS[permission].includes(userRole);
  };

  // Check if user has at least a certain role level
  const hasRoleLevel = (requiredLevel) => {
    return roleLevel >= requiredLevel;
  };

  // Check if user has higher role than another user
  const hasHigherRoleThan = (otherUserRole) => {
    const otherRoleLevel = ROLE_HIERARCHY[otherUserRole] || 0;
    return roleLevel > otherRoleLevel;
  };

  // Get role display name
  const getRoleDisplayName = (role = userRole) => {
    const roleNames = {
      [USER_ROLES.SUPER_ADMIN]: 'Super Administrator',
      [USER_ROLES.ADMIN]: 'Administrator',
      [USER_ROLES.ENGINEER_OF_RECORD]: 'Engineer of Record',
      [USER_ROLES.TSF_OPERATOR]: 'TSF Operator',
      [USER_ROLES.REGULATOR]: 'Regulator',
      [USER_ROLES.MANAGEMENT]: 'Management',
      [USER_ROLES.CONSULTANT]: 'Consultant',
      [USER_ROLES.VIEWER]: 'Viewer'
    };
    return roleNames[role] || 'Unknown Role';
  };

  // Get role color for UI display
  const getRoleColor = (role = userRole) => {
    const roleColors = {
      [USER_ROLES.SUPER_ADMIN]: '#d32f2f', // Red
      [USER_ROLES.ADMIN]: '#f57c00', // Orange
      [USER_ROLES.ENGINEER_OF_RECORD]: '#006039', // Green (primary)
      [USER_ROLES.TSF_OPERATOR]: '#1976d2', // Blue
      [USER_ROLES.REGULATOR]: '#7b1fa2', // Purple
      [USER_ROLES.MANAGEMENT]: '#388e3c', // Dark Green
      [USER_ROLES.CONSULTANT]: '#0288d1', // Light Blue
      [USER_ROLES.VIEWER]: '#616161' // Grey
    };
    return roleColors[role] || '#616161';
  };

  // Check if user can access specific facilities
  const canAccessFacility = (facilityId) => {
    if (!currentUser?.facilities_access) {
      return hasPermission('canViewAllDocuments'); // Admin roles can access all
    }
    return currentUser.facilities_access.includes(facilityId) || 
           hasPermission('canViewAllDocuments');
  };

  // Get available roles for user creation (can only create users with equal or lower roles)
  const getAvailableRoles = () => {
    return Object.entries(USER_ROLES)
      .filter(([_, roleValue]) => (ROLE_HIERARCHY[roleValue] || 0) <= roleLevel)
      .map(([key, value]) => ({
        value,
        label: getRoleDisplayName(value),
        color: getRoleColor(value)
      }));
  };

  const contextValue = {
    userRole,
    roleLevel,
    hasPermission,
    hasRoleLevel,
    hasHigherRoleThan,
    getRoleDisplayName,
    getRoleColor,
    canAccessFacility,
    getAvailableRoles,
    USER_ROLES,
    PERMISSIONS
  };

  return (
    <UserRoleContext.Provider value={contextValue}>
      {children}
    </UserRoleContext.Provider>
  );
};

export const useUserRole = () => {
  const context = useContext(UserRoleContext);
  if (!context) {
    throw new Error('useUserRole must be used within a UserRoleProvider');
  }
  return context;
};

export default UserRoleContext;
