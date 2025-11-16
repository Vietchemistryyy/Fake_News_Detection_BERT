# Fake News Detection System

Hệ thống phát hiện tin giả sử dụng BERT + AI Verification, hỗ trợ đa ngôn ngữ (English/Vietnamese).

---

## Giới thiệu

Dự án phát hiện tin giả toàn diện với khả năng phân tích tin tức bằng tiếng Anh và tiếng Việt.

**Tính năng:**
- Đa ngôn ngữ (English/Vietnamese)
- BERT Models (RoBERTa + PhoBERT)
- Authentication (JWT)
- Query History
- MC Dropout
- AI Verification (Gemini/Groq/OpenAI)

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

### 1. Clone & Install

```bash
git clone <repo-url>
cd Fake_News_Detection_BERT
pip install -r requirements.txt
cd fe && npm install && cd ..
```

### 2. Setup MongoDB

```bash
# Windows
net start MongoDB

# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### 3. Configure

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

### 4. Run

```bash
# Terminal 1 - Backend
cd api && python main.py

# Terminal 2 - Frontend
cd fe && npm run dev
```

### 5. Access

- Web: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Testing

```bash
python tests/test_system.py
python tests/test_mongodb.py
python tests/test_api.py
```

---

## Usage

1. Register/Login tại http://localhost:3000
2. Chọn ngôn ngữ (EN/VI)
3. Paste nội dung tin tức
4. Click "Analyze"
5. Xem kết quả: Real/Fake + Confidence

---

## API Endpoints

- `POST /auth/register` - Đăng ký
- `POST /auth/login` - Đăng nhập
- `POST /predict` - Phân tích tin tức
- `GET /history` - Lịch sử
- `GET /health` - Health check

Docs: http://localhost:8000/docs

---

## Tips

- Generate strong SECRET_KEY: `openssl rand -hex 32`
- Enable AI verification (Gemini FREE) trong `.env`
- Dùng MongoDB Compass để xem database
- Check logs: `tail -f api/api.log`

---

## Issues?

- Xem [tests/README.md](tests/README.md) cho troubleshooting
- Check [API Docs](http://localhost:8000/docs)
- Tạo issue trên GitHub

---

## Contributing

Fork → Branch → Commit → Push → Pull Request

---

## License

MIT License

---

**Made with love for fighting fake news**
