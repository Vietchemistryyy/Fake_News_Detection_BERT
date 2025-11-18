# 🛡️ Fake News Detection System v1.0

> AI-Powered Multi-Language News Verification System

Hệ thống phát hiện tin giả toàn diện sử dụng BERT models, hỗ trợ đa ngôn ngữ (English/Vietnamese) với admin dashboard và query history tracking.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Tính năng chính

### 🤖 AI Models
- **RoBERTa** (English) - Fine-tuned for fake news detection (92%+ accuracy)
- **PhoBERT** (Vietnamese) - Fine-tuned on Vietnamese news dataset (92%+ accuracy)
- **AI Verification** - Optional cross-verification with Gemini & Groq (100% FREE)
- **MC Dropout** - Uncertainty estimation for predictions

### 👥 User Management
- **Authentication** - JWT-based secure authentication
- **User Roles** - Regular users and admin roles
- **Query History** - Track all predictions with statistics
- **Personal Dashboard** - View your analysis history

### 🛡️ Admin Dashboard
- **System Statistics** - Monitor total users, queries, and predictions
- **User Management** - CRUD operations for users (Create, Read, Update, Delete)
- **Query Monitoring** - View all system queries in real-time
- **Role Management** - Promote/demote users to admin

### 🌍 Multi-Language Support
- English news detection using RoBERTa
- Vietnamese news detection using PhoBERT
- Automatic language detection
- Separate models for optimal accuracy

---

## Tech Stack

**Backend:** FastAPI, PyTorch, Transformers, MongoDB  
**Frontend:** Next.js, Tailwind CSS  
**Models:** RoBERTa (English), PhoBERT (Vietnamese)

---

## Cấu trúc

```
├── api/              # Backend
├── fe/               # Frontend
├── models/BERT/      # Trained models
├── tests/            # Test scripts
└── requirements.txt
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Vietchemistryyy/Fake_News_Detection_BERT.git
cd Fake_News_Detection_BERT
```

### 2. Install Dependencies

```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd fe && npm install && cd ..
```

### 3. Download Models

**Option A: Download from Hugging Face (Recommended)**

```bash
# Download fine-tuned models (~1GB, 5-10 minutes)
python download_models.py
```

This will download:
- **RoBERTa** (English) - 92%+ accuracy
- **PhoBERT** (Vietnamese) - 92%+ accuracy

**Option B: Manual Download**

Visit Hugging Face and download manually:
- [RoBERTa Model](https://huggingface.co/Vietchemistryyy/fake-news-roberta-english)
- [PhoBERT Model](https://huggingface.co/Vietchemistryyy/fake-news-phobert-vietnamese)

Extract to:
- `models/BERT/`
- `models/PhoBERT/`

### 4. Setup MongoDB

```bash
# Windows
net start MongoDB

# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### 5. Configure

**api/.env:**
```bash
API_HOST=0.0.0.0
API_PORT=8000
MONGODB_URL=mongodb://localhost:27017/
MONGODB_DB_NAME=fake_news_detection
SECRET_KEY=your-secret-key-here
MODEL_PATH=../models/BERT
CORS_ORIGINS=http://localhost:3000
```

Generate SECRET_KEY: `openssl rand -hex 32`

**fe/.env.local:**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 6. Run

```bash
# Terminal 1 - Backend
cd api && python main.py

# Terminal 2 - Frontend
cd fe && npm run dev
```

### 7. Access

- Web: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Fine-tuning PhoBERT với VFND Dataset

### 🚀 Google Colab (FREE GPU T4!)

1. Mở `notebooks/05_PHOBERT_VFND_COLAB.txt`
2. Copy từng cell vào Google Colab
3. Chạy tuần tự (1-2 giờ)
4. Download model từ Google Drive về `models/PhoBERT_VFND/`
5. Update `api/.env`: `PHOBERT_MODEL_PATH=../models/PhoBERT_VFND`
6. Restart API: `cd api && python main.py`

---

## Testing

```bash
python tests/test_system.py
python tests/test_mongodb.py
python tests/test_api.py
```

---

## 📖 Usage Guide

### Regular User Flow
1. **Register** at http://localhost:3000
2. **Login** with your credentials
3. **Select language** (English or Vietnamese)
4. **Paste news content** (10-5000 characters)
5. **Analyze** and get instant results
6. **View history** to track your queries

### Admin Flow
1. **Login** with admin credentials (admin/123456)
2. **Monitor system** statistics
3. **Manage users** - view, update roles, delete
4. **View all queries** across the system
5. **System administration** tasks

---

## 🔌 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `GET /auth/me` - Get current user info

### Prediction
- `POST /predict` - Analyze news (supports EN/VI)
- `POST /predict-batch` - Batch prediction

### History
- `GET /history` - Get user query history
- `GET /history/stats` - Get user statistics

### Admin (Admin only)
- `GET /admin/users` - Get all users
- `GET /admin/stats` - Get system statistics
- `GET /admin/queries` - Get all queries
- `PUT /admin/users/{id}` - Update user
- `DELETE /admin/users/{id}` - Delete user

**Full API Documentation:** http://localhost:8000/docs

---

## 🧪 Testing

### Automated Tests
```bash
# Test admin functionality
python test_admin.py

# Test PhoBERT model
python test_phobert_model.py

# Test history API
python quick_test.py
```

### Manual Testing
Follow the checklist in `TEST_CHECKLIST_V1.0.md`

---

## 💡 Tips & Best Practices

- **Security:** Generate strong SECRET_KEY: `openssl rand -hex 32`
- **AI Verification:** Enable Gemini (FREE) in `.env` for cross-verification
- **Database:** Use MongoDB Compass to visualize data
- **Logs:** Check `api/api.log` for debugging
- **Performance:** Use GPU for faster inference
- **Backup:** Regularly backup MongoDB database

---

## 🐛 Troubleshooting

### Common Issues

**Backend won't start:**
- Check if MongoDB is running: `net start MongoDB` (Windows)
- Verify port 8000 is available
- Check `api/api.log` for errors

**Frontend won't start:**
- Verify port 3000 is available
- Run `npm install` in `fe/` directory
- Check `.env.local` configuration

**Predictions fail:**
- Ensure models are in `models/BERT/` and `models/PhoBERT/`
- Check if all model files are present (config.json, model files, tokenizer files)
- Verify GPU/CPU availability

**History not working:**
- Ensure MongoDB is running and connected
- Check if user is logged in
- Verify token is valid

See `TROUBLESHOOTING.md` for detailed solutions.

---

## 📊 System Requirements

- **Python:** 3.12+
- **Node.js:** 18+
- **MongoDB:** 4.4+
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB for models and data
- **GPU:** Optional (CUDA-compatible for faster inference)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Nguyen Quoc Viet**

© 2025 Nguyen Quoc Viet. All rights reserved.

---

## 🙏 Acknowledgments

- **Hugging Face** for Transformers library
- **VinAI** for PhoBERT model
- **FastAPI** for the amazing web framework
- **Next.js** for the frontend framework
- **MongoDB** for the database

---

## 📞 Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check the documentation at http://localhost:8000/docs
- Review `TEST_CHECKLIST_V1.0.md` for testing guidelines

---

**🛡️ Fighting fake news with AI - One prediction at a time**
