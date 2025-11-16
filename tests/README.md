# Tests

Các test scripts để kiểm tra hệ thống sau khi clone về.

## Danh sách Tests

### 1. test_system.py
**Mục đích:** Test toàn bộ hệ thống backend (imports, config, model, API)

**Chạy:**
```bash
python tests/test_system.py
```

**Kiểm tra:**
- Imports (FastAPI, PyTorch, Transformers)
- Configuration (paths, settings)
- Model files (config.json, pytorch_model.bin, etc.)
- Model loading (RoBERTa)
- Predictions (3 test cases)
- API structure (routes)

**Kết quả mong đợi:** 6/6 tests passed

---

### 2. test_model.py
**Mục đích:** Test model inference chi tiết

**Chạy:**
```bash
python tests/test_model.py
```

**Kiểm tra:**
- Model loading
- Tokenization
- Inference
- Output format
- Confidence scores
- Multiple test cases

---

### 3. test_mongodb.py
**Mục đích:** Test MongoDB connection và operations

**Chạy:**
```bash
python tests/test_mongodb.py
```

**Kiểm tra:**
- MongoDB connection
- User CRUD operations
- Query CRUD operations
- Password hashing/verification
- Database cleanup

**Yêu cầu:** MongoDB phải đang chạy

---

### 4. test_api.py
**Mục đích:** Test API endpoints (integration test)

**Chạy:**
```bash
# Terminal 1: Start API
cd api
python main.py

# Terminal 2: Run test
python tests/test_api.py
```

**Kiểm tra:**
- Health check endpoint
- User registration
- User login
- Prediction (English + Vietnamese)
- Query history
- Statistics

**Yêu cầu:** API và MongoDB phải đang chạy

---

## Quick Test All

Chạy tất cả tests theo thứ tự:

```bash
# 1. Test system (không cần MongoDB/API)
python tests/test_system.py

# 2. Test model
python tests/test_model.py

# 3. Start MongoDB (nếu chưa chạy)
# Windows: net start MongoDB
# macOS: brew services start mongodb-community
# Linux: sudo systemctl start mongod

# 4. Test MongoDB
python tests/test_mongodb.py

# 5. Start API (terminal riêng)
cd api
python main.py

# 6. Test API (terminal khác)
python tests/test_api.py
```

---

## Troubleshooting

### Test system failed

**Lỗi:** Import errors
```bash
# Cài dependencies
pip install -r requirements.txt
```

**Lỗi:** Model files not found
```bash
# Check model files
ls models/BERT/
# Cần có: config.json, pytorch_model.bin, vocab.json, merges.txt
```

---

### Test MongoDB failed

**Lỗi:** Connection refused
```bash
# Check MongoDB status
# Windows:
sc query MongoDB

# macOS/Linux:
sudo systemctl status mongod
```

**Giải pháp:** Start MongoDB
```bash
# Windows (as Administrator):
net start MongoDB

# macOS:
brew services start mongodb-community

# Linux:
sudo systemctl start mongod
```

---

### Test API failed

**Lỗi:** API not responding
```bash
# Make sure API is running
cd api
python main.py

# Check if port 8000 is available
# Windows:
netstat -ano | findstr :8000

# macOS/Linux:
lsof -i :8000
```

**Lỗi:** 401 Unauthorized
- Token có thể đã expire
- Chạy lại test để generate token mới

---

## Expected Results

Nếu tất cả setup đúng:

```
test_system.py    → 6/6 tests passed
test_model.py     → All predictions successful
test_mongodb.py   → 3/3 tests passed
test_api.py       → 6/6 tests passed
```

---

## Tips

1. **Chạy test_system.py trước** - Không cần MongoDB/API, test nhanh nhất
2. **MongoDB phải chạy** trước khi test MongoDB và API
3. **API phải chạy** trước khi test API
4. **Dùng virtual environment** để tránh conflict dependencies
5. **Check logs** nếu test fail: `api/api.log`

---

## Notes

- Tests tự động cleanup data (xóa test users/queries)
- test_api.py tạo random username để tránh conflict
- Tất cả tests có thể chạy nhiều lần
- Tests không ảnh hưởng đến production data

---

## Recommended Testing Flow

**Sau khi clone repository:**

1. Install dependencies: `pip install -r requirements.txt`
2. Test system: `python tests/test_system.py`
3. Start MongoDB
4. Test MongoDB: `python tests/test_mongodb.py`
5. Start API: `cd api && python main.py`
6. Test API: `python tests/test_api.py`

**Nếu tất cả pass → Hệ thống sẵn sàng!**
