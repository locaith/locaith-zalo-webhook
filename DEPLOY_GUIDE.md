# 🚀 Hướng dẫn Deploy lên Fly.io

## ✅ Đã hoàn thành:
- ✅ Dockerfile đã được tạo
- ✅ fly.toml đã được cấu hình

## 📋 Các bước tiếp theo:

### 1. Cài đặt flyctl (Windows)
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**Sau khi cài xong, mở terminal mới và chạy:**
```powershell
flyctl version
flyctl auth signup   # hoặc flyctl auth login (nếu có tài khoản)
```

### 2. Khởi tạo app trên Fly
```powershell
flyctl launch
```

**Lựa chọn:**
- App name: `locaith-webhook` (hoặc tên khác)
- Region: `sin` (Singapore) hoặc `hkg`
- "Would you like to allocate a Postgres database?" → **No**
- "Deploy now?" → **No** (để set secrets trước)

### 3. Thiết lập biến môi trường (secrets)
```powershell
flyctl secrets set ZALO_OA_TOKEN=xxxxx
flyctl secrets set ZALO_APP_SECRET=xxxxx
flyctl secrets set ENABLE_APPSECRET_PROOF=false
flyctl secrets set GEMINI_API_KEY=xxxxx
flyctl secrets set ENABLE_CORS=false
flyctl secrets set ALLOWED_ORIGINS=*
flyctl secrets set ADMIN_ALERT_USER_ID=
flyctl secrets set MAX_UPLOAD_MB=25
```

### 4. Cấu hình máy ảo (free-friendly)
```powershell
flyctl scale vm shared-cpu-1x
flyctl scale memory 256
```

### 5. Deploy
```powershell
flyctl deploy
flyctl logs
```

### 6. Kiểm tra
- Health check: `https://<app-name>.fly.dev/health`
- Swagger docs: `https://<app-name>.fly.dev/docs`

## 🔗 Webhook URL cho Zalo
Sau khi deploy thành công, webhook URL sẽ là:
```
https://<app-name>.fly.dev/webhook
```

## 💰 Chi phí
Free credit $5/tháng đủ cho 1 máy nhỏ với cấu hình này.

## 🔧 Troubleshooting
- Nếu build lỗi: kiểm tra requirements.txt
- Nếu app không start: xem logs với `flyctl logs`
- Nếu cần restart: `flyctl apps restart <app-name>`