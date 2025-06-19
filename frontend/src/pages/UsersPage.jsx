import React, { useState } from 'react';
import {
  Box, Typography, Button, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  MenuItem, Select, InputLabel, FormControl, CircularProgress,
  Alert, AlertTitle, Chip
} from '@mui/material';
import { Add, Edit, Delete, AdminPanelSettings } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { users } from '../api/tailingsIQApi';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorDisplay from '../components/common/ErrorDisplay';
import { useUserRole, USER_ROLES } from '../contexts/UserRoleContext';

const UsersPage = () => {
  const queryClient = useQueryClient();
  const { hasPermission } = useUserRole();
  const isAdmin = hasPermission('canAccessAdminPanel');

  const [openDialog, setOpenDialog] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [formState, setFormState] = useState({
    username: '',
    email: '',
    password: '',
    firstName: '',
    lastName: '',
    role: USER_ROLES.ENGINEER_OF_RECORD,
    organization: '',
    position: '',
    phone: '',
    licenseNumber: '',
  });
  const [formErrors, setFormErrors] = useState({});

  const { data: usersData, isLoading, isError, error } = useQuery(
    'users', 
    users.getUsers,
    { enabled: isAdmin }
  );

  const createUserMutation = useMutation(users.createUser, {
    onSuccess: () => {
      queryClient.invalidateQueries('users');
      handleCloseDialog();
    },
    onError: (err) => {
      setFormErrors({ submit: err.response?.data?.detail || 'Error creating user' });
    },
  });

  const updateUserMutation = useMutation(
    ({ id, data }) => users.updateUser(id, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('users');
        handleCloseDialog();
      },
      onError: (err) => {
        setFormErrors({ submit: err.response?.data?.detail || 'Error updating user' });
      },
    }
  );

  const deleteUserMutation = useMutation(users.deleteUser, {
    onSuccess: () => {
      queryClient.invalidateQueries('users');
    },
    onError: (err) => {
      console.error('Error deleting user:', err);
    },
  });

  const handleOpenCreate = () => {
    setEditingUser(null);
    setFormState({
      username: '',
      email: '',
      password: '',
      firstName: '',
      lastName: '',
      role: USER_ROLES.ENGINEER_OF_RECORD,
      organization: '',
      position: '',
      phone: '',