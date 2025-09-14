# main.py
import os, time, hmac, hashlib, json, re, io
from typing import Dict, Any, List, Optional

import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import google.generativeai as genai
from pypdf import PdfReader
from docx import Document as Docx
import faiss
from bs4 import BeautifulSoup

# =================== ENV & INIT ===================
load_dotenv()

ZALO_OA_TOKEN        = os.getenv("ZALO_OA_TOKEN")
ZALO_APP_SECRET      = os.getenv("ZALO_APP_SECRET", "")
ZALO_VERIFY_FILE     = os.getenv("ZALO_VERIFY_FILE")
ENABLE_APPSECRET     = os.getenv("ENABLE_APPSECRET_PROOF", "false").lower() == "true"

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
ENABLE_CORS          = os.getenv("ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS_STR  = os.getenv("ALLOWED_ORIGINS", "*")
ADMIN_ALERT_USER_ID  = os.getenv("ADMIN_ALERT_USER_ID", "")
MAX_UPLOAD_MB        = int(os.getenv("MAX_UPLOAD_MB", "25"))

# Zalo Domain Verification
VERIFY_DIR = "verify"

assert ZALO_OA_TOKEN and GEMINI_API_KEY, "Thiếu ZALO_OA_TOKEN hoặc GEMINI_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO   = "gemini-2.5-pro"
MODEL_EMBED = "text-embedding-004"    # kích thước 768

app = FastAPI(
    title="Locaith AI - Zalo Webhook",
    description="AI-powered Zalo OA webhook for Locaith services",
    version="1.0.0"
)

# Startup event to ensure proper initialization
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Locaith AI Zalo Webhook starting up...")
    print(f"✅ FAISS index initialized with {faiss_index.ntotal} vectors")
    print(f"✅ Knowledge base has {len(kb_chunks)} chunks")
    print("✅ Application ready to receive webhooks")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Locaith AI Zalo Webhook shutting down...")

if ENABLE_CORS:
    origins = [o.strip() for o in (ALLOWED_ORIGINS_STR or "*").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# =================== STORES (DEMO) ===================
rate_bucket: Dict[str, List[float]] = {}    # user_id -> timestamps (30s window)
warn_count: Dict[str, int] = {}             # user_id -> warnings count
soft_ban_until: Dict[str, float] = {}       # user_id -> unix time
session: Dict[str, Dict[str, Any]] = {}     # user_id -> {consented, profile, ...}

MAX_MSG_PER_30S  = 6
BAN_DURATION_SEC = 24 * 3600

# --- FAISS in-memory (RAG demo) ---
faiss_index = faiss.IndexFlatIP(768)    # inner product (cosine sau khi normalize)
kb_chunks: List[str] = []               # lưu chunk text
kb_meta:   List[dict] = []              # meta mỗi chunk


# =================== ZALO HELPERS ===================
def _appsecret_proof(access_token: str, app_secret: str) -> str:
    return hmac.new(app_secret.encode(), access_token.encode(), hashlib.sha256).hexdigest()

def zalo_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json", "access_token": ZALO_OA_TOKEN}
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        h["appsecret_proof"] = _appsecret_proof(ZALO_OA_TOKEN, ZALO_APP_SECRET)
    return h

def zalo_send_text(user_id: str, text: str) -> dict:
    """Gửi tin nhắn text qua Zalo OA API v3.0"""
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    payload = {"recipient": {"user_id": user_id}, "message": {"text": text}}
    r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
    if r.status_code >= 400:
        print("Send error:", r.text)
    return r.json() if r.text else {}

def zalo_send_image(user_id: str, image_url: str, message: str = "") -> dict:
    """Gửi tin nhắn hình ảnh qua Zalo OA API v3.0"""
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    payload = {
        "recipient": {"user_id": user_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "media",
                    "elements": [{
                        "media_type": "image",
                        "url": image_url,
                        "title": message
                    }]
                }
            }
        }
    }
    r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
    if r.status_code >= 400:
        print("Send image error:", r.text)
    return r.json() if r.text else {}

def zalo_get_profile(user_id: str) -> Dict[str, Any]:
    """Lấy profile cơ bản nếu user đã quan tâm OA."""
    url = "https://openapi.zalo.me/v2.0/oa/getprofile"
    payload = {"user_id": user_id}
    r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
    if r.status_code == 200:
        return r.json().get("data", {})
    print("Get profile failed:", r.text)
    return {}


# =================== SPAM & SESSION ===================
def is_spamming(user_id: str) -> bool:
    now = time.time()
    if user_id in soft_ban_until and now < soft_ban_until[user_id]:
        return True
    bucket = rate_bucket.setdefault(user_id, [])
    bucket.append(now)
    rate_bucket[user_id] = [t for t in bucket if now - t <= 30]
    return len(rate_bucket[user_id]) > MAX_MSG_PER_30S

def escalate_spam(user_id: str) -> str:
    c = warn_count.get(user_id, 0) + 1
    warn_count[user_id] = c
    if c == 1:
        return ("Em phát hiện tần suất tin nhắn hơi dày. "
                "Mình hạ nhịp một xíu nha ạ—lần sau em buộc phải tạm khóa 24 giờ.")
    else:
        soft_ban_until[user_id] = time.time() + BAN_DURATION_SEC
        if ADMIN_ALERT_USER_ID:
            zalo_send_text(ADMIN_ALERT_USER_ID, f"[SPAM] User {user_id} bị tạm khóa 24h.")
        return ("Do tái phạm spam nên tài khoản đã bị tạm khóa tương tác 24 giờ. "
                "Nếu có nhu cầu gấp vui lòng liên hệ CSKH Locaith AI.")

def ensure_session(user_id: str) -> Dict[str, Any]:
    return session.setdefault(user_id, {
        "consented": False,
        "profile": None,
        "product": None,     # "chatbot" | "website" | "landing"
        "info": {},          # name/phone/email/domain/logo/notes
        "history": [],       # [{"role": "user|assistant", "text": str, "ts": float}]
        "image_buffer": []   # các NOTE ảnh nội bộ chưa tiêu thụ
    })

def push_history(user_id: str, role: str, text: str):
    s = ensure_session(user_id)
    s["history"].append({"role": role, "text": text, "ts": time.time()})
    if len(s["history"]) > 12:
        s["history"] = s["history"][-12:]

def short_context(user_id: str, k: int = 6) -> str:
    s = ensure_session(user_id)
    rows = []
    for h in s["history"][-k:]:
        prefix = "USER:" if h["role"] == "user" else "ASSISTANT:"
        rows.append(f"{prefix} {h['text']}")
    return "\n".join(rows)


# =================== PROMPTS ===================
def system_prompt(profile: Dict[str, Any]) -> str:
    dn = profile.get("display_name") or "bạn"
    return f"""
Bạn là Trợ lý CSKH của Locaith AI (locaith.ai). Giọng điệu: ấm áp, tự nhiên, linh hoạt như người thật.
Có thể trò chuyện & tư vấn ngoài sản phẩm (du lịch/đời sống nhẹ nhàng) giống phong cách chủ OA.

NGUYÊN TẮC:
- Chỉ dùng thông tin cá nhân khi người dùng đã đồng ý; cần dữ liệu liên hệ thì xin rõ ràng (Họ tên, SĐT, Email).
- Người mới: giới thiệu ngắn và hỏi quan tâm: Chatbot AI hay Website (Website hoàn chỉnh / Landing page).
- Chatbot AI:
  • Cơ Bản: ~6.000.000đ (một lần) — UI chatbot frontend, tích hợp website/fanpage, hỗ trợ thiết lập ban đầu, tuỳ chỉnh dữ liệu cơ bản.
  • Nâng Cao: ~2.000.000đ/tháng — khuyến nghị <5.000 người dùng/tháng, cung cấp API key miễn phí, báo cáo/analytics chi tiết.
  • Doanh Nghiệp: tuỳ chỉnh, không giới hạn người dùng, hỗ trợ 24/7, đào tạo nhân viên.
  • Nền tảng huấn luyện & nhúng: phechat.com (khách tự huấn luyện và gắn vào website).
  • Quy trình: xin thông tin liên hệ → yêu cầu tài liệu tri thức (PDF/DOC/URL) để train.
- Website:
  • Website Hoàn Chỉnh: ~2.000.000đ, code thuần (không WordPress), triển khai nhanh ~2 ngày, deploy GitHub + Vercel.
  • Landing Page: ~500.000đ, ~1 ngày, tối ưu chuyển đổi, form liên hệ.
  • Hỏi domain (đã có/chưa), xin logo/brand guideline, nội dung chính.
- Soạn thảo văn bản theo quy phạm/nghị định: mạnh; có thể đưa khung, checklist, mẫu.
- Chống spam: cảnh cáo 1 lần; tái phạm → tạm khóa 24h.
- Khi thông tin giá/ưu đãi có thể thay đổi, dùng "ước lượng/tham khảo" và đề nghị kết nối nhân sự.

PHONG CÁCH:
- Thân thiện, mạch lạc, câu ngắn; dùng bullet khi cần; luôn chốt bằng câu hỏi/CTA tiếp theo.
- Xưng hô linh hoạt với {dn}; nếu chưa chắc, dùng “mình/bạn” trung tính.
"""

def onboarding(profile: Dict[str, Any]) -> str:
    name = profile.get("display_name") or "bạn"
    return (f"Chào {name}! Em là Trợ lý Locaith AI 🌟\n"
            "Mình đang quan tâm Chatbot AI hay Website (Website hoàn chỉnh / Landing page)?\n"
            "Để hỗ trợ nhanh, cho em xin Họ tên, SĐT, Email nhé — đồng ý không ạ?")

def ask_contact() -> str:
    return "Cho em xin Họ tên, SĐT, Email nha (ví dụ: Nguyễn A, 09xx..., a@example.com)."

def ask_assets(product: str) -> str:
    if product == "chatbot":
        return ("Tuyệt ạ! Anh/chị vui lòng gửi **tài liệu/URL** để em huấn luyện chatbot "
                "(PDF/DOC/URL; có FAQ/kịch bản thì đính kèm luôn).")
    if product == "website":
        return ("Ok mình làm **Website code thuần**. Anh/chị đã có **domain** chưa ạ? "
                "Và cho em xin **logo/màu thương hiệu** + nội dung chính để triển khai nhé.")
    if product == "landing":
        return ("Ok **Landing page ~1 ngày**. Cho em mục tiêu chiến dịch, nội dung chính "
                "và thông tin form liên hệ mong muốn ạ.")
    return ""


# =================== IMAGE PIPELINE ===================
def try_download_image_from_event(event: dict) -> Optional[bytes]:
    """Tìm URL ảnh trong webhook (tuỳ payload)."""
    att = event.get("message", {}).get("attachments", [])
    if not att:
        return None
    for a in att:
        if a.get("type") != "image":
            continue
        payload = a.get("payload", {}) or {}
        url = payload.get("url") or payload.get("thumb") or payload.get("href")
        if not url:
            continue
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.content and len(r.content) <= 8 * 1024 * 1024:
                return r.content
        except Exception:
            continue
    return None

def analyze_image_internal(img_bytes: bytes) -> str:
    """Dùng gemini-2.5-pro tạo NOTE nội bộ 1 câu, không tiết lộ cho user."""
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return "an image (unreadable)"

    prompt = (
        "You are an INTERNAL image summarizer.\n"
        "Describe very briefly and neutrally what the image mainly shows in ONE short sentence.\n"
        "No personal/sensitive attributes. Plain text only."
    )
    model = genai.GenerativeModel(MODEL_PRO)
    resp = model.generate_content([prompt, image])
    try:
        note = resp.text.strip()
        return note[:200]
    except Exception:
        return "an image (summary failed)"


# =================== RAG: EMBED / CHUNK / INDEX ===================
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Dùng Gemini embeddings chuẩn SDK:
    genai.embed_content(model="models/text-embedding-004", content=...)
    Trả về vector float32 đã normalize (cosine).
    """
    vecs = []
    for t in texts:
        try:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=t,
                task_type="retrieval_document",  # gợi ý loại tác vụ để nhúng ổn định
            )
            # SDK thường trả về {'embedding': {'values': [..]}}
            values = res.get("embedding", {}).get("values") or res.get("embedding")
            v = np.array(values, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        except Exception as e:
            print("Embed error:", e)
            # fallback vector zero đúng kích thước 768
            vecs.append(np.zeros(768, dtype="float32"))
    return np.vstack(vecs)

def chunk_text(text: str, max_chars=1200) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    while text:
        chunk = text[:max_chars]
        cut = chunk.rfind(". ")
        if cut > 200:
            chunk = chunk[:cut+1]
        chunks.append(chunk)
        text = text[len(chunk):]
    return chunks

def extract_text_from_file(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        with io.BytesIO(data) as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        with io.BytesIO(data) as f:
            doc = Docx(f)
            return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")

def index_document(source_id: str, text: str, owner: str):
    chunks = chunk_text(text)
    if not chunks:
        return
    vecs = embed_texts(chunks)
    faiss_index.add(vecs)
    base = len(kb_chunks)
    for i, c in enumerate(chunks):
        kb_chunks.append(c)
        kb_meta.append({"source": source_id, "owner": owner, "pos": base + i})

def retrieve(query: str, top_k=5) -> List[dict]:
    if faiss_index.ntotal == 0:
        return []
    qv = embed_texts([query])
    D, I = faiss_index.search(qv, top_k)
    hits = []
    for idx in I[0]:
        if 0 <= idx < len(kb_chunks):
            hits.append({"text": kb_chunks[idx], "meta": kb_meta[idx]})
    return hits


# =================== LLM CALL (FINAL ANSWER) ===================
def call_flash_final(sys_prompt: str, user_text: str, profile: Dict[str, Any],
                     image_notes: List[str], history_str: str, rag_context: str) -> str:
    private_clause = (
        "PRIVATE IMAGE NOTES (do NOT reveal or mention they exist):\n"
        + ("\n".join(f"- {n}" for n in image_notes) if image_notes else "- (none)")
    )
    content = (
        f"{private_clause}\n\n"
        f"RECENT CONTEXT:\n{history_str or '(none)'}\n\n"
        f"RETRIEVED CONTEXT (may be empty):\n{rag_context or '(none)'}\n\n"
        f"USER MESSAGE:\n{user_text or ''}\n"
    )
    model = genai.GenerativeModel(MODEL_FLASH)
    resp = model.generate_content(
        [
            {"role": "user", "parts": [sys_prompt]},
            {"role": "user", "parts": [content]},
            {"role": "user", "parts": [
                "Answer naturally as a human CSKH. Do NOT expose private image notes."
            ]},
        ],
        generation_config={"temperature": 0.7}
    )
    try:
        return resp.text.strip()
    except Exception:
        return "Xin lỗi, hệ thống đang bận. Bạn vui lòng thử lại giúp em nhé!"


# =================== SMALL NLU ===================
def guess_product(text: str) -> str:
    t = (text or "").lower()
    if "chatbot" in t: return "chatbot"
    if "landing" in t or "landing page" in t: return "landing"
    if "website" in t: return "website"
    return ""

def parse_contact(text: str) -> Dict[str, str]:
    email = None
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    if m: email = m.group(0)
    phone = None
    m2 = re.search(r"(0|\+84)[0-9]{8,11}", (text or "").replace(" ", ""))
    if m2: phone = m2.group(0)
    return {"email": email, "phone": phone}


# =================== ROUTES ===================
@app.get("/health")
def health():
    """Health check endpoint for Render.com deployment verification"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "Locaith AI - Zalo Webhook",
        "version": "1.0.0",
        "uptime": time.time(),
        "checks": {
            "database": "ok",  # FAISS index
            "api_keys": "configured" if ZALO_OA_TOKEN and GEMINI_API_KEY else "missing",
            "kb_chunks": len(kb_chunks),
            "faiss_index": int(faiss_index.ntotal)
        }
    }

@app.get("/")
def root():
    """Root endpoint for basic service info"""
    return {
        "service": "Locaith AI - Zalo Webhook", 
        "status": "running",
        "health_check": "/health"
    }

# ---- Knowledge upload: trực tiếp ----
@app.post("/kb/url")
def kb_from_url(user_id: str = Form(...), url: str = Form(...)):
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        if not text:
            return {"ok": False, "error": "Trang không có nội dung văn bản"}
        index_document(source_id=url, text=text, owner=user_id)
        return {"ok": True}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Lỗi mạng/HTTP: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Lỗi xử lý nội dung: {e}"}

# ---- Knowledge từ URL ----
@app.post("/kb/url")
def kb_from_url(user_id: str = Form(...), url: str = Form(...)):
    try:
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
    except Exception:
        return {"ok": False, "error": "Không tải được URL"}
    index_document(source_id=url, text=text, owner=user_id)
    return {"ok": True}

# ---- Zalo Domain Verification ----
@app.get("/{verify_name}")
def serve_zalo_verify(verify_name: str):
    """
    Trả về file xác thực của Zalo nếu tên khớp.
    URL yêu cầu: https://<domain>/<VERIFY_FILE>
    """
    if ZALO_VERIFY_FILE and verify_name == ZALO_VERIFY_FILE:
        path = os.path.join(VERIFY_DIR, ZALO_VERIFY_FILE)
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Not found")

# ---- Zalo webhook ----
@app.get("/zalo/webhook")
async def webhook_verification(challenge: str = None):
    """Webhook verification endpoint for Zalo OA"""
    if challenge:
        return {"challenge": challenge}
    return {"error": "Missing challenge parameter"}

@app.post("/zalo/webhook")
async def webhook(req: Request):
    """Main webhook endpoint for processing Zalo messages"""
    # Verify signature if enabled
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        signature = req.headers.get("X-ZEvent-Signature")
        if signature:
            body = await req.body()
            expected_signature = hmac.new(
                ZALO_APP_SECRET.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                return {"status": "invalid_signature"}
    
    event = await req.json()
    
    # Log event for debugging
    print(f"Received event: {json.dumps(event, indent=2)}")
    
    # Parse event data
    try:
        event_name = event.get("event_name", "")
        user_id = event.get("sender", {}).get("id")
        user_text = event.get("message", {}).get("text", "").strip()
        
        if not user_id:
            return {"status": "no_user_id"}
            
        # Handle different event types
        if event_name == "user_send_text":
            # Text message - continue with existing logic
            pass
        elif event_name == "user_send_image":
            # Image message - extract image and continue
            user_text = user_text or "[Đã gửi hình ảnh]"
        elif event_name == "user_send_sticker":
            # Sticker message
            zalo_send_text(user_id, "Em đã nhận được sticker rồi ạ! 😊")
            return {"status": "sticker_received"}
        elif event_name == "user_send_gif":
            # GIF message
            zalo_send_text(user_id, "Em đã nhận được GIF rồi ạ! 🎬")
            return {"status": "gif_received"}
        elif event_name == "user_send_audio":
            # Audio message
            zalo_send_text(user_id, "Em đã nhận được tin nhắn âm thanh rồi ạ! 🎵")
            return {"status": "audio_received"}
        elif event_name == "user_send_video":
            # Video message
            zalo_send_text(user_id, "Em đã nhận được video rồi ạ! 🎥")
            return {"status": "video_received"}
        elif event_name == "user_send_file":
            # File message
            zalo_send_text(user_id, "Em đã nhận được file rồi ạ! 📎")
            return {"status": "file_received"}
        elif event_name == "user_send_location":
            # Location message
            zalo_send_text(user_id, "Em đã nhận được vị trí rồi ạ! 📍")
            return {"status": "location_received"}
        elif event_name == "follow":
            # User follows OA
            profile = zalo_get_profile(user_id)
            welcome_msg = f"Chào mừng {profile.get('display_name', 'bạn')} đến với Locaith AI! 🎉\n\nEm là trợ lý AI của Locaith, sẵn sàng hỗ trợ bạn về:\n• Thiết kế Website\n• Tạo Landing Page\n• Phát triển Chatbot AI\n\nHãy nhắn tin để bắt đầu nhé! 😊"
            zalo_send_text(user_id, welcome_msg)
            return {"status": "welcome_sent"}
        elif event_name == "unfollow":
            # User unfollows OA
            return {"status": "unfollowed"}
        else:
            # Unknown event type
            print(f"Unknown event type: {event_name}")
            return {"status": "unknown_event"}
            
    except Exception as e:
        print(f"Error parsing event: {e}")
        return {"status": "parse_error"}

    # Anti-spam
    if is_spamming(user_id):
        zalo_send_text(user_id, escalate_spam(user_id))
        return {"status": "spam"}

    s = ensure_session(user_id)

    # Soft-ban check
    if user_id in soft_ban_until and time.time() < soft_ban_until[user_id]:
        return {"status": "banned"}

    # Cache profile
    if not s["profile"]:
        s["profile"] = zalo_get_profile(user_id)

    # Ảnh: nếu có -> phân tích nội bộ, không trả lời về ảnh
    img_bytes = try_download_image_from_event(event)
    if img_bytes:
        note = analyze_image_internal(img_bytes)
        s["image_buffer"].append(note)
        if not user_text:
            msg = "Em đã nhận ảnh rồi ạ. Mình muốn em hỗ trợ gì từ ảnh này vậy ạ?"
            zalo_send_text(user_id, msg)
            push_history(user_id, "assistant", msg)
            return {"status": "image_received"}

    # Consent & contact
    if not s["consented"]:
        if user_text.lower() in ["đồng ý", "dong y", "ok", "oke", "yes", "y"]:
            s["consented"] = True
            zalo_send_text(user_id, "Cảm ơn ạ! " + ask_contact())
            return {"status": "consented"}
        else:
            msg = onboarding(s["profile"])
            zalo_send_text(user_id, msg)
            push_history(user_id, "assistant", msg)
            return {"status": "ask_consent"}

    if "email" not in s["info"] or "phone" not in s["info"] or "name" not in s["info"]:
        found = parse_contact(user_text)
        if found.get("email"): s["info"]["email"] = found["email"]
        if found.get("phone"): s["info"]["phone"] = found["phone"]
        if "name" not in s["info"]:
            s["info"]["name"] = s["profile"].get("display_name", "Khách")
        if not (s["info"].get("email") and s["info"].get("phone")):
            zalo_send_text(user_id, ask_contact())
            return {"status": "ask_contact_retry"}
        zalo_send_text(user_id, "Cảm ơn ạ! Anh/chị đang muốn Chatbot AI, Website hay Landing page?")
        return {"status": "ask_product"}

    # Xác định product
    if not s["product"]:
        g = guess_product(user_text)
        if g:
            s["product"] = g
            zalo_send_text(user_id, ask_assets(g))
            return {"status": "ask_assets"}
        # Tư vấn tự nhiên qua Flash với history + image notes
        history_str = short_context(user_id, 6)
        rag_ctx = ""
        if faiss_index.ntotal > 0 and user_text:
            ctx_hits = retrieve(user_text, 4)
            rag_ctx = "\n\n".join(h["text"] for h in ctx_hits)
        reply = call_flash_final(system_prompt(s["profile"]), user_text, s["profile"], s["image_buffer"], history_str, rag_ctx)
        zalo_send_text(user_id, reply)
        push_history(user_id, "user", user_text or "[image]")
        push_history(user_id, "assistant", reply)
        s["image_buffer"].clear()
        return {"status": "clarify_product"}

    # Flow theo sản phẩm
    if s["product"] in ["website", "landing"]:
        if "domain" not in s["info"]:
            low = user_text.lower()
            if "." in low or "chưa" in low or "không" in low:
                s["info"]["domain"] = user_text.strip()
            else:
                zalo_send_text(user_id, "Anh/chị đã có domain chưa ạ? (nhập domain, hoặc nói 'chưa có').")
                return {"status": "ask_domain"}
        if "logo" not in s["info"]:
            if "logo" in user_text.lower():
                s["info"]["logo"] = "sẽ gửi"
            else:
                s["info"]["logo"] = "chưa nhận"
                zalo_send_text(user_id, "Anh/chị vui lòng cung cấp logo và màu thương hiệu (gửi file sau cũng được).")

        summary = (
            "Tóm tắt:\n"
            f"- Gói: {'Website' if s['product']=='website' else 'Landing page'}\n"
            f"- Liên hệ: {s['info'].get('name')} | {s['info'].get('phone')} | {s['info'].get('email')}\n"
            f"- Domain: {s['info'].get('domain')}\n"
            f"- Logo/Brand: {s['info'].get('logo')}\n"
            "Nếu ok, em gửi hướng dẫn nộp tài liệu & timeline triển khai ạ."
        )
        zalo_send_text(user_id, summary)
        push_history(user_id, "assistant", summary)
        return {"status": "summary"}

    if s["product"] == "chatbot":
        if "kb_status" not in s["info"]:
            s["info"]["kb_status"] = "waiting"
            zalo_send_text(user_id, "Anh/chị vui lòng gửi tài liệu (PDF/DOC) hoặc URL để train chatbot nhé.")
            return {"status": "ask_kb"}
        summary = (
            "Tóm tắt đơn hàng Chatbot AI:\n"
            f"- Liên hệ: {s['info'].get('name')} | {s['info'].get('phone')} | {s['info'].get('email')}\n"
            "- Tài liệu: đang chờ bạn gửi.\n"
            "Cần tích hợp Website/Fanpage/khác không ạ?"
        )
        zalo_send_text(user_id, summary)
        push_history(user_id, "assistant", summary)
        return {"status": "summary"}

    # Fallback chung (cả đời sống)
    history_str = short_context(user_id, 6)
    rag_ctx = ""
    if faiss_index.ntotal > 0 and user_text:
        ctx_hits = retrieve(user_text, 4)
        rag_ctx = "\n\n".join(h["text"] for h in ctx_hits)
    reply = call_flash_final(system_prompt(s["profile"]), user_text, s["profile"], s["image_buffer"], history_str, rag_ctx)
    zalo_send_text(user_id, reply)
    push_history(user_id, "user", user_text or "[image]")
    push_history(user_id, "assistant", reply)
    s["image_buffer"].clear()
    return {"status": "ok"}
