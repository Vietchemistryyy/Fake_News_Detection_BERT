# 🚀 Quick Start Guide

Get the Fake News Detection System running in 10 minutes!

## ⚡ Prerequisites

- Python 3.12+
- Node.js 18+
- MongoDB 4.4+
- Git

## 📦 Installation Steps

### Step 1: Clone Repository (1 min)

```bash
git clone https://github.com/Vietchemistryyy/Fake_News_Detection_BERT.git
cd Fake_News_Detection_BERT
```

### Step 2: Install Dependencies (2 min)

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd fe
npm install
cd ..
```

### Step 3: Download Models (5-10 min)

```bash
python download_models.py
```

**What this does:**
- Downloads RoBERTa model (~480 MB) for English
- Downloads PhoBERT model (~517 MB) for Vietnamese
- Total: ~1 GB

**Models from:**
- https://huggingface.co/Vietchemistryyy/fake-news-roberta-english
- https://huggingface.co/Vietchemistryyy/fake-news-phobert-vietnamese

### Step 4: Start MongoDB (30 sec)

```bash
# Windows
net start MongoDB

# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### Step 5: Configure Environment (1 min)

**Backend (api/.env):**
```bash
# Copy example
copy api\.env.example api\.env

# Or create manually with:
API_HOST=0.0.0.0
API_PORT=8000
MONGODB_URL=mongodb://localhost:27017/
SECRET_KEY=your-secret-key-here
```

Generate SECRET_KEY:
```bash
openssl rand -hex 32
```

**Frontend (fe/.env.local):**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Step 6: Run the System (30 sec)

**Terminal 1 - Backend:**
```bash
cd api
python main.py
```

Wait for:
```
✓ Connected to MongoDB
✓ Admin user initialized
✓ English model loaded
✓ Vietnamese model loaded
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend:**
```bash
cd fe
npm run dev
```

Wait for:
```
ready - started server on 0.0.0.0:3000
```

### Step 7: Access the System

Open browser:
- **Web App:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs

**Default Admin:**
- Username: `admin`
- Password: `123456`

## ✅ Verify Installation

### Test 1: Check API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": {"en": true, "vi": true},
  "database_connected": true
}
```

### Test 2: Test Models
```bash
python test_phobert_model.py
```

### Test 3: Use Web Interface
1. Go to http://localhost:3000
2. Click "Login to Start"
3. Login with admin/123456
4. Try analyzing some news!

## 🐛 Troubleshooting

### Models not downloading?
```bash
# Check internet connection
# Try manual download from Hugging Face
# Or use base models (lower accuracy)
```

### MongoDB not starting?
```bash
# Windows: Check Services
# Install MongoDB if not installed
# Check port 27017 is available
```

### Port already in use?
```bash
# Backend (8000): Change API_PORT in api/.env
# Frontend (3000): Change port in fe/package.json
```

### Import errors?
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📚 Next Steps

- Read full [README.md](README.md)
- Check [TEST_CHECKLIST_V1.0.md](TEST_CHECKLIST_V1.0.md)
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Visit [API Documentation](http://localhost:8000/docs)

## 🎯 Usage

### As Regular User:
1. Register account
2. Login
3. Select language (EN/VI)
4. Paste news content
5. Click "Analyze"
6. View results

### As Admin:
1. Login with admin/123456
2. View system statistics
3. Manage users
4. Monitor queries

## 💡 Tips

- Use GPU for faster inference
- Enable AI verification for cross-checking
- Check query history for past analyses
- Change admin password in production

## 🆘 Need Help?

- Check [README.md](README.md) for detailed docs
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Visit API docs at http://localhost:8000/docs
- Create issue on GitHub

---

**Made with ❤️ by Nguyen Quoc Viet**

© 2025 Nguyen Quoc Viet. All rights reserved.
