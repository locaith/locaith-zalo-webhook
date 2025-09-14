# Locaith Zalo Webhook FastAPI App

🚀 **FastAPI webhook application for Zalo OA integration with Gemini AI**

## 📋 Mô tả

Ứng dụng webhook FastAPI để tích hợp Zalo Official Account với Google Gemini AI, cho phép chatbot tự động trả lời tin nhắn từ người dùng.

## 🛠️ Công nghệ sử dụng

- **FastAPI** - Web framework hiện đại cho Python
- **Google Gemini AI** - AI model để xử lý và trả lời tin nhắn
- **Zalo API** - Tích hợp với Zalo Official Account
- **Docker** - Containerization
- **Render.com** - Cloud deployment platform

## 📁 Cấu trúc project

```
├── main.py              # FastAPI application chính
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── .gitignore          # Git ignore rules
├── .env.example        # Environment variables template
└── README.md           # Documentation
```

## 🚀 Deploy lên Render.com

### Bước 1: Chuẩn bị Repository

1. **Push code lên GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Bước 2: Deploy trên Render

1. **Truy cập Render.com:**
   - Đăng nhập vào [Render.com](https://render.com)
   - Click "New" → "Web Service"

2. **Kết nối Repository:**
   - Chọn "Connect a repository"
   - Authorize GitHub và chọn repository của bạn

3. **Cấu hình Deploy:**
   ```
   Name: locaith-zalo-webhook
   Environment: Docker
   Region: Singapore (gần Việt Nam nhất)
   Branch: main
   ```

4. **Environment Variables:**
   Thêm các biến môi trường sau trong Render dashboard:
   ```
   ZALO_APP_ID=your_zalo_app_id
   ZALO_APP_SECRET=your_zalo_app_secret
   ZALO_ACCESS_TOKEN=your_zalo_access_token
   GEMINI_API_KEY=your_gemini_api_key
   PORT=10000
   ```

5. **Deploy:**
   - Click "Create Web Service"
   - Render sẽ tự động build và deploy app

### Bước 3: Cấu hình Webhook

1. **Lấy URL:**
   Sau khi deploy thành công, bạn sẽ có URL:
   ```
   https://your-app-name.onrender.com
   ```

2. **Webhook Endpoint:**
   ```
   https://your-app-name.onrender.com/zalo/webhook
   ```

3. **Cấu hình trong Zalo Developer:**
   - Truy cập [Zalo Developer Console](https://developers.zalo.me)
   - Vào OA của bạn → Webhook
   - Paste URL webhook vào
   - Test webhook

## 🧪 Test Webhook

```bash
curl -X POST "https://your-app-name.onrender.com/zalo/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "sender": {"id": "USER_TEST_1"},
    "message": {"text": "xin chào"}
  }'
```

## 📝 Environment Variables

| Variable | Mô tả | Bắt buộc |
|----------|-------|----------|
| `ZALO_APP_ID` | Zalo App ID từ Developer Console | ✅ |
| `ZALO_APP_SECRET` | Zalo App Secret | ✅ |
| `ZALO_ACCESS_TOKEN` | Zalo Access Token | ✅ |
| `GEMINI_API_KEY` | Google Gemini API Key | ✅ |
| `PORT` | Port cho ứng dụng (mặc định: 8080) | ❌ |

## 🔧 Local Development

1. **Clone repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Tạo virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # hoặc
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Tạo file .env:**
   ```bash
   cp .env.example .env
   # Điền thông tin API keys vào file .env
   ```

5. **Chạy ứng dụng:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```

## 💰 Chi phí Render.com

- **Free Tier:** 
  - 750 giờ/tháng miễn phí
  - App sẽ sleep sau 15 phút không hoạt động
  - Phù hợp cho testing và development

- **Paid Plan ($7/tháng):**
  - Không sleep
  - Phù hợp cho production

## 🔒 Bảo mật

- ✅ File `.env` đã được loại trừ khỏi Git
- ✅ API keys được lưu trữ an toàn trong Environment Variables
- ✅ Không commit sensitive data lên repository

## 📞 Hỗ trợ

Nếu gặp vấn đề trong quá trình deploy:

1. Kiểm tra logs trong Render dashboard
2. Verify environment variables đã được set đúng
3. Test webhook endpoint bằng curl
4. Kiểm tra Zalo webhook configuration

## 📄 License

MIT License - Locaith Solutions

---

**Developed by Locaith Team** 🚀