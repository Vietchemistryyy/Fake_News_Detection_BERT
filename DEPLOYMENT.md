# Fake News Detection System

Full-stack fake news detection system using **BERT (RoBERTa)** + **OpenAI GPT-3.5-turbo** for cross-verification.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Frontend (Next.js + React)                    │
│  • User interface at http://localhost:3000              │
│  • Real-time analysis with Tailwind CSS                 │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
                     ↓
┌─────────────────────────────────────────────────────────┐
│         Backend (FastAPI + Uvicorn)                     │
│  • API at http://localhost:8000                         │
│  • BERT model inference (GPU-accelerated)               │
│  • OpenAI verification (optional)                       │
└─────────────────────────────────────────────────────────┘
```

## Features

- **BERT Model Inference**: Fine-tuned RoBERTa for fake news detection
- **OpenAI Verification**: Cross-check with GPT-3.5-turbo (optional)
- **Dual-Model Verdict**: Combined confidence from both models
- **Temperature Scaling**: Calibrated confidence scores
- **MC Dropout**: Uncertainty estimation during inference
- **Batch Processing**: Handle multiple texts at once
- **REST API**: Full OpenAPI documentation at `/docs`
- **Modern UI**: Responsive Next.js interface with real-time updates

## Quick Start

### 1. Setup and Start

```bash
cd d:\Fake_News_Detection_BERT
python app.py
```

This will:
- ✓ Check Python dependencies (install if needed)
- ✓ Check Node.js availability
- ✓ Verify BERT model path
- ✓ Setup environment files (.env)
- ✓ Install npm packages
- ✓ Start FastAPI backend (port 8000)
- ✓ Start Next.js frontend (port 3000)

### 2. Open Browser

```
http://localhost:3000
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news: Scientists discover...",
    "verify_with_openai": false,
    "mc_dropout": false
  }'
```

Response:
```json
{
  "label": "real",
  "confidence": 0.92,
  "probabilities": {
    "real": 0.92,
    "fake": 0.08
  },
  "openai_result": null,
  "combined_result": null
}
```

### With OpenAI Verification

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news: Scientists discover...",
    "verify_with_openai": true,
    "mc_dropout": false
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Article 1 text...",
      "Article 2 text..."
    ],
    "verify_with_openai": false
  }'
```

### API Documentation

Interactive Swagger UI:
```
http://localhost:8000/docs
```

## Configuration

### Backend (.env)

Edit `api/.env`:

```env
# Server
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model
MODEL_NAME=roberta-base
MODEL_PATH=../models/roberta
TEMPERATURE=0.7
MC_DROPOUT_ENABLED=true
MC_DROPOUT_ITERATIONS=5

# OpenAI (Optional)
OPENAI_API_KEY=your-api-key-here
ENABLE_OPENAI=false
```

### Frontend (.env.local)

Edit `fe/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## OpenAI Setup (Optional)

To enable OpenAI verification:

1. Get free API key from: https://platform.openai.com/api-keys
2. Edit `api/.env`:
   ```env
   OPENAI_API_KEY=sk-...
   ENABLE_OPENAI=true
   ```
3. Restart the API

## Command Line Options

```bash
# Run full setup and start both services
python app.py

# Start only backend
python app.py --backend-only

# Start only frontend
python app.py --frontend-only

# Run setup only
python app.py --setup-only

# Skip setup and start directly
python app.py --skip-setup
```

## Project Structure

```
Fake_News_Detection_BERT/
├── api/                      # FastAPI backend
│   ├── main.py              # Main FastAPI app
│   ├── config.py            # Configuration
│   ├── model_loader.py      # BERT model management
│   ├── openai_verifier.py   # OpenAI integration
│   ├── utils.py             # Utilities
│   ├── requirements.txt     # Python dependencies
│   ├── .env                 # Environment config
│   └── api.log              # Log file
│
├── fe/                       # Next.js frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── index.js     # Landing page
│   │   │   ├── detector.js  # Detection page
│   │   │   ├── _app.js      # App wrapper
│   │   │   └── _document.js # Document wrapper
│   │   └── styles/
│   │       └── globals.css  # Tailwind styles
│   ├── package.json         # Dependencies
│   ├── next.config.js       # Next.js config
│   ├── tailwind.config.js   # Tailwind config
│   ├── .env.local           # Environment config
│   └── .next/               # Build output
│
├── src/                      # Training code (unchanged)
├── models/                   # Model checkpoints
├── data/                     # Dataset
├── notebooks/                # Jupyter notebooks
│
├── app.py                    # Orchestrator script
└── requirements-dev.txt      # Dev dependencies
```

## Model Details

- **Base Model**: RoBERTa (roberta-base or fine-tuned checkpoint)
- **Task**: Binary classification (Real/Fake news)
- **Accuracy**: ~95% on test set
- **Inference**: GPU-accelerated with CPU fallback
- **Temperature**: 0.7 (configurable)
- **MC Dropout**: 5 iterations (configurable)

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <PID> /F
```

### Model Not Found

The API will automatically download `roberta-base` from HuggingFace if local model not found. First inference may take longer.

### OpenAI API Errors

- Check API key is valid
- Check rate limits
- Verify OPENAI_API_KEY in `.env` is set correctly
- Try with ENABLE_OPENAI=false first

### Frontend Won't Load

```bash
# Clear npm cache
cd fe
npm cache clean --force
npm install

# Rebuild
npm run build
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA/GPU drivers installed for faster inference
2. **Batch Processing**: Use `/predict-batch` for multiple texts
3. **Caching**: Results are computed fresh each time
4. **Timeout**: Increase OPENAI_TIMEOUT if getting timeout errors

## Development

### Run Backend Only

```bash
cd api
python -m uvicorn main:app --reload
```

### Run Frontend Only

```bash
cd fe
npm run dev
```

### Run Tests

```bash
# Test API endpoints
curl http://localhost:8000/health

# Test model info
curl http://localhost:8000/models/info
```

## Deployment

### Docker

Create `Dockerfile` and `docker-compose.yml` for containerized deployment.

### Production

1. Set `DEBUG=false` in `.env`
2. Use production-grade ASGI server (Gunicorn)
3. Add authentication/rate limiting
4. Enable HTTPS
5. Use environment variables for secrets

## License

MIT License

## Support

For issues or questions, check:
- API docs: http://localhost:8000/docs
- Frontend logs: Browser console (F12)
- Backend logs: `api/api.log` or terminal output
