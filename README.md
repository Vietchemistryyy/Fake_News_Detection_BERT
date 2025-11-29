# Fake News Detection with BERT

A fake news detection system that actually works. Built with BERT models for both English and Vietnamese news articles.

[Watch the demo](https://youtu.be/9RmSw6CzblE) if you want to see it in action.

## What is this?

Basically, you paste in a news article and the system tells you if it's likely fake or real. It uses two fine-tuned BERT models:
- RoBERTa for English (92% accuracy)
- PhoBERT for Vietnamese (92% accuracy)

There's also an optional Groq AI integration for cross-verification, which is completely free.

## Features

**For users:**
- Detect fake news in English or Vietnamese
- See confidence scores and detailed breakdowns
- Track your analysis history
- Get AI cross-verification (optional)

**For admins:**
- User management dashboard
- Search and filter users
- View system statistics
- Monitor all queries

## Tech Stack

- **Backend:** FastAPI + PyTorch + Transformers
- **Frontend:** Next.js + Tailwind CSS
- **Database:** MongoDB
- **Models:** RoBERTa (EN), PhoBERT (VI)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Vietchemistryyy/Fake_News_Detection_BERT.git
cd Fake_News_Detection_BERT

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd fe && npm install && cd ..
```

### 2. Download Models

```bash
python download_models.py
```

This downloads the fine-tuned models (~1GB total). They'll go into `models/BERT/` and `models/PhoBERT/`.

### 3. Setup Environment

Create `api/.env`:

```env
API_HOST=0.0.0.0
API_PORT=8000
MONGODB_URL=mongodb://localhost:27017/
MONGODB_DB_NAME=fake_news_detection
SECRET_KEY=your-secret-key-here
MODEL_PATH=../models/BERT
PHOBERT_MODEL_PATH=../models/PhoBERT
CORS_ORIGINS=http://localhost:3000
```

Generate a secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Optional - Add Groq AI (free):**
Get an API key from [console.groq.com](https://console.groq.com/) and add:
```env
ENABLE_GROQ=true
GROQ_API_KEY=your-key-here
```

Create `fe/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Start MongoDB

```bash
# Windows
net start MongoDB

# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### 5. Run the App

Open two terminals:

```bash
# Terminal 1 - Backend
cd api
python main.py
```

```bash
# Terminal 2 - Frontend
cd fe
npm run dev
```

Visit http://localhost:3000

## Project Structure

```
├── api/              # FastAPI backend
├── fe/               # Next.js frontend
├── models/           # Trained BERT models
│   ├── BERT/        # RoBERTa (English)
│   └── PhoBERT/     # PhoBERT (Vietnamese)
├── tests/            # Test scripts
└── notebooks/        # Training notebooks
```

## How to Use

**As a regular user:**
1. Register an account
2. Login
3. Select language (English or Vietnamese)
4. Paste your news article (10-5000 characters)
5. Click analyze
6. Check your history anytime

**As an admin:**
- Login with admin credentials (default: admin/123456)
- Access the admin dashboard
- Manage users, view stats, monitor queries

## API Endpoints

Full docs at http://localhost:8000/docs when running.

**Auth:**
- `POST /auth/register` - Create account
- `POST /auth/login` - Login
- `GET /auth/me` - Get current user

**Detection:**
- `POST /predict` - Analyze news article
- `POST /predict-batch` - Batch analysis

**History:**
- `GET /history` - Your query history
- `GET /history/stats` - Your statistics

**Admin:**
- `GET /admin/users` - List all users
- `GET /admin/stats` - System stats
- `PUT /admin/users/{id}` - Update user
- `DELETE /admin/users/{id}` - Delete user

## Fine-tuning PhoBERT

If you want to train your own Vietnamese model:

1. Open `notebooks/05_PHOBERT_VFND_COLAB.txt`
2. Copy the code into Google Colab (free T4 GPU!)
3. Run it (takes 1-2 hours)
4. Download the trained model
5. Update your `.env` file

## Testing

```bash
# Test the system
python tests/test_system.py

# Test MongoDB connection
python tests/test_mongodb.py

# Test API endpoints
python tests/test_api.py
```

## Common Issues

**Backend won't start:**
- Make sure MongoDB is running
- Check if port 8000 is free
- Look at `api/api.log` for errors

**Frontend won't start:**
- Check if port 3000 is available
- Try `npm install` again
- Verify `.env.local` exists

**Models not loading:**
- Confirm models are in `models/BERT/` and `models/PhoBERT/`
- Check if all files downloaded correctly
- Make sure you have enough RAM (8GB minimum)

**Predictions failing:**
- Verify MongoDB is connected
- Check if you're logged in
- Make sure the text is between 10-5000 characters

## Requirements

- Python 3.12+
- Node.js 18+
- MongoDB 4.4+
- 8GB RAM minimum (16GB recommended)
- ~5GB disk space for models

GPU is optional but speeds things up.

## Contributing

Feel free to open issues or submit PRs. The code could definitely use some improvements.

## License

MIT License - do whatever you want with it.

## Credits

Built by [@Vietchemistryyy](https://github.com/Vietchemistryyy)

Thanks to:
- Hugging Face for the Transformers library
- VinAI for PhoBERT
- FastAPI and Next.js teams
- Groq for free AI API access

---

If this helped you, give it a star ⭐

Made with ❤️ and lots of coffee
