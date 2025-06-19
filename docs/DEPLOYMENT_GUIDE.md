# TailingsIQ Deployment Guide

## File Organization

The TailingsIQ application has been extracted into the following deployment-ready files:

### Backend Files (FastAPI/Python)

#### Core Application
- `backend_app_main.py` - Main FastAPI application entry point
- `backend_app_core_config.py` - Configuration settings with environment variables
- `backend_Dockerfile` - Docker container configuration for backend
- `backend_requirements.txt` - Python dependencies
- `app_core_config.py` - Alternative config file
- `requirements.txt` - Alternative requirements file

#### User Management System ✨ NEW
- `models_user.py` - User database models and Pydantic schemas
- `services_user_service.py` - User service with authentication and security
- `api_auth.py` - Authentication API routes (login, logout, password management)
- `api_admin_users.py` - Admin user management API routes

#### Synthetic Data Framework
- `synthetic_data_models.py` - Database models for synthetic data
- `synthetic_data_generator.py` - Core data generation engine
- `synthetic_data.py` - API routes for synthetic data management
- `synthetic_data_service.py` - Business logic integration layer
- `generate_synthetic_data.py` - Standalone command-line script
- `SYNTHETIC_DATA_README.md` - Synthetic data documentation

### Frontend Files (React/Material-UI)

#### Core Components
- `package.json` - NPM dependencies with Material-UI, Chart.js, React Router
- `frontend_src_styles_App.css` - Complete CSS with Manus design (Rolex Green #006039, Gold #d4af37)
- `frontend_src_styles_theme.js` - Material-UI theme configuration
- `src_components_common_TailingsIQLogo.jsx` - Logo component with "TailingsIQ" branding
- `src_components_common_Header.jsx` - Navigation header with authentication
- `frontend_Dockerfile` - Production Docker container for frontend

#### User Management Frontend ✨ NEW
- `pages_UsersPage.jsx` - Complete user management interface
- `contexts_AuthContext.jsx` - Authentication context provider
- `contexts_UserRoleContext.jsx` - Role-based access control context
- `api_tailingsIQApi.js` - Complete API client with user management

### Configuration Files
- `.env.example` - Environment variables template
- `.env.development` - Development environment settings
- `.env.production` - Production environment settings
- `alembic.ini` - Database migration configuration

### Deployment Files
- `docker-compose.yml` - Complete local development stack (PostgreSQL, Redis, Elasticsearch)
- `nginx.conf` - Web server configuration
- `railway.json` - Railway platform deployment
- `vercel.json` - Vercel frontend deployment
- `railway-deployment.md` - Step-by-step deployment guide

### Documentation
- `README.md` - Project documentation
- `DEPLOYMENT_GUIDE.md` - Comprehensive deployment instructions
- `.gitignore` - Version control exclusions

## User Management System Features ✨

### Backend Capabilities
- ✅ **Role-Based Access Control** - 8 user roles from Super Admin to Viewer
- ✅ **Secure Authentication** - JWT tokens with bcrypt password hashing
- ✅ **Account Security** - Failed login tracking, account lockout protection
- ✅ **Password Management** - Change password, reset password functionality
- ✅ **Audit Logging** - Complete user action tracking for compliance
- ✅ **User CRUD Operations** - Full admin interface for user management
- ✅ **Profile Management** - Users can update their own profiles

### Frontend Features
- ✅ **Complete User Interface** - Full-featured user management page
- ✅ **Role-Based UI** - Different interface elements for different roles
- ✅ **Authentication Flow** - Login, logout, password management
- ✅ **Permission System** - Granular permissions for different features
- ✅ **Responsive Design** - Mobile-friendly user management

### User Roles Hierarchy

1. **Super Admin** - Full system access, can manage all users
2. **Admin** - System administration, user management (except Super Admins)
3. **Engineer of Record** - Full technical access, document and compliance management
4. **Management** - Business oversight, reporting access
5. **TSF Operator** - Operational monitoring and basic document access
6. **Regulator** - Compliance and regulatory document access
7. **Consultant** - Limited technical access for analysis
8. **Viewer** - Read-only access to basic information

## Deployment Instructions

### Backend Deployment

1. **Setup Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values including:
   # - DATABASE_URL
   # - SECRET_KEY (generate a secure 256-bit key)
   # - OPENAI_API_KEY
   # - REDIS_URL
   # - ELASTICSEARCH_URL
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Setup**
   ```bash
   # Initialize database migrations
   alembic upgrade head

   # The application will create a default super admin user:
   # Username: superadmin
   # Password: ChangeMe123! (CHANGE IMMEDIATELY)
   ```

4. **Run with Docker**
   ```bash
   docker build -f backend_Dockerfile -t tailingsiq-backend .
   docker run -p 8000:8000 tailingsiq-backend
   ```

5. **Deploy to Railway**
   - Connect your GitHub repository to Railway
   - Use the railway.json configuration
   - Set environment variables in Railway dashboard
   - The app will auto-deploy on git push

### Frontend Deployment

1. **Install Dependencies**
   ```bash
   npm install
   # Use the dependencies from package.json
   ```

2. **Environment Setup**
   ```bash
   # Development
   cp .env.development .env.local

   # Production
   cp .env.production .env.local

   # Set REACT_APP_API_URL to your backend URL
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

4. **Deploy to Vercel**
   - Connect repository to Vercel
   - Use vercel.json configuration
   - Set environment variables in Vercel dashboard
   - Auto-deploy on git push

### Complete Project Structure

To recreate the full project structure, organize files as follows:

```
tailingsiq/
├── backend/
│   ├── app/
│   │   ├── main.py (from backend_app_main.py)
│   │   ├── core/
│   │   │   └── config.py (from backend_app_core_config.py)
│   │   ├── models/
│   │   │   ├── user.py (from models_user.py) ✨
│   │   │   └── synthetic_data_models.py
│   │   ├── services/
│   │   │   ├── user_service.py (from services_user_service.py) ✨
│   │   │   ├── synthetic_data_generator.py
│   │   │   └── synthetic_data_service.py
│   │   └── api/
│   │       ├── auth.py (from api_auth.py) ✨
│   │       ├── synthetic_data.py
│   │       └── admin/
│   │           └── users.py (from api_admin_users.py) ✨
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   ├── alembic.ini
│   └── generate_synthetic_data.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── common/
│   │   │       ├── TailingsIQLogo.jsx
│   │   │       └── Header.jsx
│   │   ├── pages/
│   │   │   └── UsersPage.jsx (from pages_UsersPage.jsx) ✨
│   │   ├── contexts/
│   │   │   ├── AuthContext.jsx (from contexts_AuthContext.jsx) ✨
│   │   │   └── UserRoleContext.jsx (from contexts_UserRoleContext.jsx) ✨
│   │   ├── api/
│   │   │   └── tailingsIQApi.js (from api_tailingsIQApi.js) ✨
│   │   └── styles/
│   │       ├── App.css
│   │       └── theme.js
│   ├── package.json
│   ├── .env.development
│   └── .env.production
├── deployment/
│   ├── docker-compose.yml
│   ├── nginx.conf
│   ├── railway.json
│   └── vercel.json
└── README.md
```

## User Management Quick Start

### 1. First Login
```
URL: http://localhost:3000/login
Username: superadmin
Password: ChangeMe123!

⚠️ IMPORTANT: Change this password immediately after first login!
```

### 2. Create Your First Users
1. Navigate to Admin → Users
2. Click "Add User"
3. Fill in user details and assign appropriate role
4. User will receive login credentials

### 3. Role Assignment Guide
- **Super Admin**: Only for system administrators
- **Admin**: For IT staff managing users and system
- **Engineer of Record**: For qualified engineers managing TSF
- **TSF Operator**: For daily operations staff
- **Management**: For business oversight and reporting
- **Regulator**: For regulatory access and compliance
- **Consultant**: For external technical consultants
- **Viewer**: For read-only access

## Security Configuration

### 1. JWT Token Security
```python
# In .env file
SECRET_KEY=your-256-bit-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 2. Password Policy
- Minimum 8 characters
- Account lockout after 5 failed attempts
- 30-minute lockout duration
- Password change tracking

### 3. Audit Logging
All user actions are logged including:
- Login/logout events
- User creation/modification
- Password changes
- Permission changes
- Document access

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `GET /api/v1/auth/me` - Get current user
- `PUT /api/v1/auth/profile` - Update profile
- `POST /api/v1/auth/change-password` - Change password
- `POST /api/v1/auth/request-password-reset` - Request password reset
- `POST /api/v1/auth/reset-password` - Reset password

### User Management (Admin only)
- `GET /api/v1/admin/users` - List users
- `POST /api/v1/admin/users` - Create user
- `GET /api/v1/admin/users/{id}` - Get user
- `PUT /api/v1/admin/users/{id}` - Update user
- `DELETE /api/v1/admin/users/{id}` - Delete user
- `POST /api/v1/admin/users/{id}/reset-password` - Reset user password

## File Count Summary

**Total Deployment Files: 38**
- Backend: 15 files (including 4 new user management files)
- Frontend: 12 files (including 4 new user management files)
- Configuration: 7 files
- Documentation: 4 files

## Next Steps

1. **Download all files** from the generated CDN links
2. **Set up your environment variables** using the `.env.example` template
3. **Configure your database** (PostgreSQL recommended)
4. **Set up external services** (Redis, Elasticsearch, AI APIs)
5. **Deploy backend** to Railway or your preferred platform
6. **Deploy frontend** to Vercel or your preferred platform
7. **Create your first admin user** and configure user roles
8. **Test the complete application** including user management

## Security Checklist

- [ ] Change default super admin password
- [ ] Generate secure SECRET_KEY for JWT tokens
- [ ] Configure proper CORS origins for production
- [ ] Set up SSL certificates (HTTPS)
- [ ] Configure proper database security
- [ ] Set up regular database backups
- [ ] Configure logging and monitoring
- [ ] Review and test user permissions
- [ ] Set up regular security updates

## Support

The user management system is now fully integrated with:
- ✅ Complete authentication flow
- ✅ Role-based access control
- ✅ User CRUD operations
- ✅ Password management
- ✅ Audit logging
- ✅ Frontend integration
- ✅ API documentation

All components work together to provide a complete, production-ready user management system for the TailingsIQ application.
