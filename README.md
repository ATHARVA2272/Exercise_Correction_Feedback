# Exercise Correction Feedback

A comprehensive exercise detection and correction system that provides real-time feedback on exercise form and technique.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project combines computer vision and machine learning to analyze exercise movements and provide real-time feedback for form correction. The system consists of a FastAPI backend for processing and a React frontend for user interaction.

## ✨ Features

- Real-time exercise detection and analysis
- Form correction feedback
- User-friendly web interface
- RESTful API for exercise data processing
- Support for multiple exercise types

## 🔧 Prerequisites

Before running this application, make sure you have the following installed:

### Backend Requirements
- Python 3.8 or higher
- pip (Python package installer)

### Frontend Requirements
- Node.js 16.0 or higher
- npm (Node package manager)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ATHARVA2272/Exercise_Correction_Feedback.git
cd Exercise_Correction_Feedback
```

### 2. Backend Setup
```bash
cd Backend
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd Frontend/kineticsGuide
npm install
```

## 🏃‍♂️ Running the Application

### Starting the Backend Server

1. Navigate to the Backend directory:
```bash
cd Backend
```

2. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The backend server will start running on `http://localhost:8000`

- API documentation will be available at: `http://localhost:8000/docs`
- Alternative API docs at: `http://localhost:8000/redoc`

### Starting the Frontend Application

1. Open a new terminal and navigate to the Frontend directory:
```bash
cd Frontend/kineticsGuide
```

2. Start the development server:
```bash
npm run dev
```

The frontend application will start running on `http://localhost:3000` (or the next available port)

## 📁 Project Structure

```
Exercise_Correction_Feedback/
├── Backend/
│   ├── main.py              # FastAPI main application
│   ├── requirements.txt     # Python dependencies
│   └── ...                  # Other backend files
├── Frontend/
│   └── kineticsGuide/
│       ├── package.json     # Node.js dependencies
│       ├── src/             # React source files
│       └── ...              # Other frontend files
└── README.md               # Project documentation
```

## 🔌 API Documentation

Once the backend is running, you can access the interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🛠️ Development

### Backend Development
- The backend uses FastAPI with automatic reload enabled
- Make changes to Python files and the server will automatically restart
- Follow PEP 8 style guidelines for Python code

### Frontend Development
- The frontend uses React with hot reload
- Changes to React components will automatically refresh in the browser
- Follow React best practices and ESLint recommendations

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**: If you get port conflicts, you can specify different ports:
   ```bash
   # Backend on different port
   uvicorn main:app --reload --port 8001
   
   # Frontend will automatically use next available port
   ```

2. **Module not found errors**: Make sure all dependencies are installed:
   ```bash
   # Reinstall backend dependencies
   cd Backend && pip install -r requirements.txt
   
   # Reinstall frontend dependencies
   cd Frontend/kineticsGuide && npm install
   ```

3. **CORS issues**: If you encounter CORS errors, ensure the backend CORS settings allow your frontend URL.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **ATHARVA2272** - *Initial work* - [GitHub Profile](https://github.com/ATHARVA2272)

## 🙏 Acknowledgments

- Thanks to all contributors who helped build this project
- Special thanks to the open-source community for the tools and libraries used

---

**Happy exercising with proper form! 💪**
