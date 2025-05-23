# Basketball Play Analysis App

A modern web application for analyzing basketball plays using computer vision and machine learning.

## Features

- Video upload and processing
- Real-time player tracking
- Play pattern recognition
- Interactive play diagrams
- Statistical analysis
- User authentication and management

## Tech Stack

### Backend
- Python 3.8+
- FastAPI
- PyTorch
- YOLOv8
- DeepSORT
- OpenPose
- MMAction2

### Frontend
- React
- TypeScript
- Tailwind CSS
- Three.js (for 3D play diagrams)

### Infrastructure
- Docker
- Kubernetes
- PostgreSQL
- Redis
- AWS S3

## Project Structure

```
.
├── src/
│   ├── api/            # API endpoints and routes
│   ├── models/         # ML models and model utilities
│   ├── services/       # Business logic and services
│   └── utils/          # Utility functions and helpers
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── docs/
│   ├── api/           # API documentation
│   └── user/          # User documentation
├── config/            # Configuration files
├── frontend/          # Frontend application
└── docker/            # Docker configuration
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/basketball-play-analysis.git
cd basketball-play-analysis
```

2. Set up the Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the development servers:
```bash
# Backend
uvicorn src.api.main:app --reload

# Frontend
cd frontend
npm run dev
```

### Docker Deployment

1. Build the Docker images:
```bash
docker-compose build
```

2. Start the services:
```bash
docker-compose up -d
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Code Style

We use:
- Black for Python code formatting
- ESLint for JavaScript/TypeScript
- Prettier for code formatting

```bash
# Format Python code
black src/

# Format frontend code
cd frontend
npm run format
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 for object detection
- DeepSORT for object tracking
- OpenPose for pose estimation
- MMAction2 for action recognition 