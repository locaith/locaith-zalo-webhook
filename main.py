# main.py
# Locaith AI – Zalo OA CSKH Chatbot (MVP, production-ready skeleton)
# - Tự nhiên như người Việt, hỏi thông tin để xưng hô đúng
# - Hiểu nhiều loại sự kiện Zalo OA: text/image/sticker/gif/audio/video/file/location/follow/unfollow
# - Chống spam, lưu phiên theo user, nhớ ngữ cảnh ngắn
# - RAG demo (FAISS) + Serper.dev cho truy vấn realtime + Gemini 2.5 Flash để tổng hợp trả lời
# - Sửa lỗi lặp chào, không còn “bắt buộc Đồng ý”, gọn route, mã sạch dễ deploy Render

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
ZALO_VERIFY_FILE     = os.getenv("ZALO_VERIFY_FILE")           # ví dụ: "zaloac516c....html"
ENABLE_APPSECRET     = os.getenv("ENABLE_APPSECRET_PROOF", "false").lower() == "true"

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY       = os.getenv("SERPER_API_KEY")
ENABLE_CORS          = os.getenv("ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS_STR  = os.getenv("ALLOWED_ORIGINS", "*")
ADMIN_ALERT_USER_ID  = os.getenv("ADMIN_ALERT_USER_ID", "")
MAX_UPLOAD_MB        = int(os.getenv("MAX_UPLOAD_MB", "25"))

VERIFY_DIR = "verify"

assert ZALO_OA_TOKEN and GEMINI_API_KEY, "Thiếu ZALO_OA_TOKEN hoặc GEMINI_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO   = "gemini-2.5-pro"

app = FastAPI(
    title="Locaith AI - Zalo Webhook",
    description="AI-powered Zalo OA webhook for Locaith services",
    version="2.0.0"
)

if ENABLE_CORS:
    origins = [o.strip() for o in (ALLOWED_ORIGINS_STR or "*").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# =================== STORES ===================
rate_bucket: Dict[str, List[float]] = {}    # user_id -> timestamps (30s window)
warn_count: Dict[str, int] = {}             # user_id -> warnings count
soft_ban_until: Dict[str, float] = {}       # user_id -> unix time

# Per-user session
session: Dict[str, Dict[str, Any]] = {}     # user_id -> {...}

MAX_MSG_PER_30S  = 6
BAN_DURATION_SEC = 24 * 3600

# --- FAISS in-memory (RAG demo) ---
EMBED_DIM = 768
faiss_index = faiss.IndexFlatIP(EMBED_DIM)    # cosine (sau khi normalize)
kb_chunks: List[str] = []
kb_meta:   List[dict] = []

# =================== ZALO HELPERS ===================
def _appsecret_proof(access_token: str, app_secret: str) -> str:
    return hmac.new(app_secret.encode(), access_token.encode(), hashlib.sha256).hexdigest()

def zalo_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json", "access_token": ZALO_OA_TOKEN}
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        h["appsecret_proof"] = _appsecret_proof(ZALO_OA_TOKEN, ZALO_APP_SECRET)
    return h

def zalo_send_text(user_id: str, text: str) -> dict:
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    payload = {"recipient": {"user_id": user_id}, "message": {"text": text}}
    r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
    if r.status_code >= 400:
        print("Send error:", r.text)
    return r.json() if r.text else {}

def zalo_get_profile(user_id: str) -> Dict[str, Any]:
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
        return ("Mình thấy tần suất tin nhắn hơi dày đó nè. "
                "Mình xin phép giảm nhịp xíu nhé—tái phạm mình sẽ tạm khóa 24 giờ ạ.")
    else:
        soft_ban_until[user_id] = time.time() + BAN_DURATION_SEC
        if ADMIN_ALERT_USER_ID:
            zalo_send_text(ADMIN_ALERT_USER_ID, f"[SPAM] User {user_id} bị tạm khóa 24h.")
        return ("Bạn đã bị tạm khóa tương tác 24 giờ do spam. Nếu cần gấp hãy liên hệ CSKH Locaith AI giúp mình ạ.")

def ensure_session(user_id: str) -> Dict[str, Any]:
    return session.setdefault(user_id, {
        "welcomed": False,             # đã gửi lời chào chưa (chỉ 1 lần)
        "profile": None,
        "salute": None,                # xưng hô ưa thích: "anh/chị/em + Tên"
        "product": None,               # "chatbot" | "website" | "landing"
        "info": {},                    # name/phone/email/domain/logo/notes
        "history": [],                 # [{role,text,ts}]
        "image_notes": []
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

# =================== WEB SEARCH VIA SERPER ===================
def search_web(query: str, num_results: int = 3) -> str:
    if not SERPER_API_KEY:
        return ""
    try:
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
        payload = {'q': query, 'num': num_results, 'gl': 'vn', 'hl': 'vi'}
        r = requests.post('https://google.serper.dev/search', headers=headers, json=payload, timeout=12)
        if r.status_code != 200:
            return ""
        data = r.json()
        parts = []
        if 'answerBox' in data and data['answerBox'].get('answer'):
            parts.append(f"**Trả lời nhanh:** {data['answerBox']['answer']}")
        for it in (data.get('organic') or [])[:num_results]:
            title = it.get('title', '')
            snippet = it.get('snippet', '')
            link = it.get('link', '')
            parts.append(f"**{title}**\n{snippet}\nNguồn: {link}")
        return "\n\n".join(parts)
    except Exception as e:
        print("Serper error:", e)
        return ""

def should_search_web(text: str) -> bool:
    if not text:
        return False
    kws = [
        "tin tức","mới nhất","hôm nay","giá","lịch","sự kiện","thời tiết","dự báo",
        "trend","viral","trending","tuyển dụng","kết quả","tỷ giá","bitcoin","chứng khoán",
        "xe bus","metro","giờ chiếu","lịch thi","kèo","promotion","khuyến mãi","học bổng"
    ]
    t = text.lower()
    return any(k in t for k in kws)

# =================== PROMPTS ===================
def system_prompt(profile: Dict[str, Any], salute: Optional[str]) -> str:
    dn = profile.get("display_name") or "bạn"
    call = salute or dn or "bạn"
    return f"""
Bạn là Minh – trợ lý AI của Locaith AI (locaith.ai). Hãy trò chuyện tự nhiên như người Việt:
- Xưng "mình" cho bản thân; xưng với đối phương là "{call}".
- Thân thiện, ấm áp, dí dỏm nhẹ; dùng emoji vừa phải.
- Có thể tư vấn CSKH Locaith (Chatbot AI, Website/Landing) khi được hỏi.
- Khi nghi vấn cần thông tin thời gian thực, hãy tổng hợp kết quả tìm web (nếu có) vào câu trả lời.
- Không tiết lộ ghi chú nội bộ hay thông tin nhạy cảm. Tuân thủ an toàn & pháp luật.
- Kết thúc bằng một câu hỏi ngắn để giữ nhịp hội thoại.

Nếu người dùng hỏi về dịch vụ, hãy ưu tiên:
- Xin thông tin liên hệ (Họ tên, SĐT, Email) để tiện xưng hô và tư vấn.
- Sau đó hỏi nhu cầu: Chatbot AI / Website hoàn chỉnh / Landing page.
"""

def onboarding_text(profile: Dict[str, Any]) -> str:
    name = profile.get("display_name") or "bạn"
    return (f"Chào {name}! Mình là **Minh** – trợ lý AI của Locaith 🌟\n"
            "Bạn có thể tâm sự hay hỏi mình bất cứ điều gì nhé 😊\n\n"
            "Để tiện xưng hô và hỗ trợ đúng nhu cầu, cho mình xin cách xưng hô (anh/chị/em + tên) "
            "và **Họ tên, SĐT, Email** được không ạ? Ví dụ: *Anh Tuấn Anh – 090xxxxxxx – email@...*")

def ask_product() -> str:
    return ("Bạn đang quan tâm **Chatbot AI**, **Website hoàn chỉnh** hay **Landing page** ạ? "
            "Mình sẽ tư vấn gói & quy trình chi tiết cho bạn.")

def ask_assets(product: str) -> str:
    if product == "chatbot":
        return ("Tuyệt ạ! Bạn vui lòng gửi **tài liệu/URL** (PDF/DOC/FAQ/kịch bản) để mình train chatbot nhé.")
    if product == "website":
        return ("OK mình triển khai **Website**. Bạn đã có **domain** chưa? "
                "Cho mình xin **logo/màu thương hiệu** và nội dung chính nữa nha.")
    if product == "landing":
        return ("OK **Landing page ~1 ngày**. Bạn cho mình mục tiêu chiến dịch, nội dung chính "
                "và thông tin form liên hệ mong muốn nhé.")
    return ""

# =================== FILE/IMAGE UTILS ===================
def try_download_image_from_event(event: dict) -> Optional[bytes]:
    att = event.get("message", {}).get("attachments", [])
    for a in att or []:
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
            pass
    return None

def analyze_image_internal(img_bytes: bytes) -> str:
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

# =================== RAG (demo) ===================
def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        try:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=t,
                task_type="retrieval_document",
            )
            values = res.get("embedding", {}).get("values") or res.get("embedding")
            v = np.array(values, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        except Exception as e:
            print("Embed error:", e)
            vecs.append(np.zeros(EMBED_DIM, dtype="float32"))
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

# =================== LLM CALL ===================
def call_flash(sys_prompt: str, user_text: str, profile: Dict[str, Any],
               image_notes: List[str], history_str: str, rag_context: str) -> str:
    web_context = ""
    if should_search_web(user_text):
        wr = search_web(user_text, 3)
        if wr:
            web_context = f"\n\nTHÔNG TIN TỪ INTERNET:\n{wr}"

    private_clause = (
        "PRIVATE IMAGE NOTES (do NOT reveal or mention they exist):\n"
        + ("\n".join(f"- {n}" for n in image_notes) if image_notes else "- (none)")
    )
    content = (
        f"{private_clause}\n\n"
        f"RECENT CONTEXT:\n{history_str or '(none)'}\n\n"
        f"RETRIEVED CONTEXT (may be empty):\n{rag_context or '(none)'}{web_context}\n\n"
        f"USER MESSAGE:\n{user_text or ''}\n"
    )
    model = genai.GenerativeModel(MODEL_FLASH)
    resp = model.generate_content(
        [
            {"role": "user", "parts": [sys_prompt]},
            {"role": "user", "parts": [content]},
            {"role": "user", "parts": ["Answer in Vietnamese, warm and friendly."]},
        ],
        generation_config={"temperature": 0.7}
    )
    try:
        return resp.text.strip()
    except Exception:
        return "Xin lỗi, hệ thống đang bận. Bạn giúp mình nhắn lại sau một chút nha 😅"

# =================== LIGHT NLU ===================
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
    name = None
    # tách tên đơn giản trước dấu phẩy hoặc trước số điện thoại
    if text:
        name = text.split(",")[0].strip()
        if phone and phone in name: name = None
    return {"email": email, "phone": phone, "name": name}

# =================== ROUTES ===================
@app.on_event("startup")
async def startup_event():
    print("🚀 Locaith AI Zalo Webhook starting...")
    print(f"FAISS vectors: {faiss_index.ntotal} | Chunks: {len(kb_chunks)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "Locaith AI - Zalo Webhook",
        "version": "2.0.0",
        "kb_chunks": len(kb_chunks),
        "faiss_index": int(faiss_index.ntotal)
    }

@app.get("/")
def root():
    return {"service": "Locaith AI - Zalo Webhook", "status": "running", "health_check": "/health"}

# ---- Knowledge ingest from URL (one route, đã fix trùng) ----
@app.post("/kb/url")
def kb_from_url(user_id: str = Form(...), url: str = Form(...)):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        if not text:
            return {"ok": False, "error": "Trang không có nội dung văn bản"}
        index_document(source_id=url, text=text, owner=user_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"Lỗi: {e}"}

# ---- Zalo Domain Verification ----
@app.get("/{verify_name}")
def serve_zalo_verify(verify_name: str):
    if ZALO_VERIFY_FILE and verify_name == ZALO_VERIFY_FILE:
        path = os.path.join(VERIFY_DIR, ZALO_VERIFY_FILE)
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Not found")

# ---- Zalo webhook verification ----
@app.get("/zalo/webhook")
async def webhook_verification(challenge: str = None):
    if challenge:
        return {"challenge": challenge}
    return {"error": "Missing challenge parameter"}

# ---- Main webhook ----
@app.post("/zalo/webhook")
async def webhook(req: Request):
    # Verify signature nếu bật
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        signature = req.headers.get("X-ZEvent-Signature")
        if signature:
            body = await req.body()
            expected = hmac.new(ZALO_APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return {"status": "invalid_signature"}

    event = await req.json()
    print("EVENT:", json.dumps(event, ensure_ascii=False))

    event_name = event.get("event_name", "")
    user_id = event.get("sender", {}).get("id")
    if not user_id:
        return {"status": "no_user_id"}

    # Chặn spam
    if is_spamming(user_id):
        zalo_send_text(user_id, escalate_spam(user_id))
        return {"status": "spam"}

    s = ensure_session(user_id)

    # Cache profile
    if not s["profile"]:
        s["profile"] = zalo_get_profile(user_id)

    # Soft-ban
    if user_id in soft_ban_until and time.time() < soft_ban_until[user_id]:
        return {"status": "banned"}

    # ===== Handle event types =====
    user_text = (event.get("message", {}) or {}).get("text", "") or ""
    lower = user_text.lower().strip()

    if event_name == "follow":
        # Chỉ chào 1 lần khi follow
        msg = onboarding_text(s["profile"])
        zalo_send_text(user_id, msg)
        s["welcomed"] = True
        push_history(user_id, "assistant", msg)
        return {"status": "welcome_sent"}

    if event_name == "unfollow":
        return {"status": "unfollowed"}

    # ảnh: lưu NOTE nội bộ (không lộ ra ngoài)
    img_bytes = try_download_image_from_event(event) if event_name == "user_send_image" else None
    if img_bytes:
        note = analyze_image_internal(img_bytes)
        s["image_notes"].append(note)
        if not user_text:
            zalo_send_text(user_id, "Mình nhận được ảnh rồi nè. Bạn muốn mình hỗ trợ gì từ ảnh này không ạ?")
            push_history(user_id, "assistant", "Đã nhận ảnh.")
            return {"status": "image_received"}

    # Sticker/GIF/Audio/Video/File/Location – phản hồi nhẹ, không phá luồng
    quick_ack = {
        "user_send_sticker": "Mình nhận được sticker rồi nè! 😊",
        "user_send_gif": "GIF xịn quá! 🎬",
        "user_send_audio": "Mình đã nhận voice của bạn nha 🎵",
        "user_send_video": "Video đã tới! 🎥",
        "user_send_file": "Mình đã nhận file rồi nha 📎",
        "user_send_location": "Đã nhận vị trí của bạn 📍",
    }
    if event_name in quick_ack and not user_text:
        zalo_send_text(user_id, quick_ack[event_name])
        return {"status": f"{event_name}_ack"}

    # ===== Greeting (không lặp) & thu thập thông tin để xưng hô =====
    if not s["welcomed"]:
        msg = onboarding_text(s["profile"])
        zalo_send_text(user_id, msg)
        s["welcomed"] = True
        push_history(user_id, "assistant", msg)
        # tiếp tục xử lý nội dung user_text bên dưới (không ép 'đồng ý')
        # => khắc phục lỗi lặp “1 câu duy nhất”
    
    # Parse thông tin liên hệ/cách xưng hô nếu người dùng gửi
    if user_text:
        found = parse_contact(user_text)
        for k in ("email", "phone", "name"):
            if found.get(k):
                s["info"][k] = found[k]
        # cách xưng hô: bắt các cụm "anh/chị/em + tên"
        m_salute = re.search(r"\b(anh|chị|em)\s+[A-Za-zÀ-ỹĐđ][\wÀ-ỹĐđ\s]*", user_text, flags=re.IGNORECASE)
        if m_salute:
            s["salute"] = m_salute.group(0).strip()

    # Nếu chưa đủ contact → nhắc nhẹ nhưng không cản trở hội thoại
    need_contact = not (s["info"].get("phone") and s["info"].get("email"))
    contact_hint = ("\n\nĐể mình tư vấn sát hơn, "
                    "bạn gửi giúp **Họ tên, SĐT, Email** nha (ví dụ: *Anh Nam – 09xx – a@b.com*).") if need_contact else ""

    # ===== Nếu người dùng hỏi về sản phẩm → flow CSKH =====
    prod_guess = guess_product(user_text)
    if prod_guess and not s["product"]:
        s["product"] = prod_guess
        zalo_send_text(user_id, ask_assets(prod_guess) + contact_hint)
        push_history(user_id, "assistant", ask_assets(prod_guess))
        return {"status": "ask_assets"}

    # Nếu đã có product → tiếp tục thu thập tài nguyên/tóm tắt
    if s["product"] in ("website", "landing"):
        if "domain" not in s["info"]:
            low = lower
            if "." in low or "chưa" in low or "không" in low:
                s["info"]["domain"] = user_text.strip()
            else:
                zalo_send_text(user_id, "Bạn đã có **domain** chưa ạ? (nhập domain hoặc nói *chưa có*).")
                return {"status": "ask_domain"}
        if "logo" not in s["info"]:
            s["info"]["logo"] = "chưa nhận"
            zalo_send_text(user_id, "Bạn gửi giúp **logo/màu thương hiệu** nha (gửi sau cũng được).")

        summary = (
            "✅ Tóm tắt yêu cầu:\n"
            f"- Gói: {'Website' if s['product']=='website' else 'Landing page'}\n"
            f"- Liên hệ: {s['info'].get('name', s['profile'].get('display_name',''))} | {s['info'].get('phone','?')} | {s['info'].get('email','?')}\n"
            f"- Domain: {s['info'].get('domain','?')}\n"
            f"- Logo/Brand: {s['info'].get('logo')}\n"
            "Nếu OK, mình gửi hướng dẫn nộp tài liệu & timeline triển khai nhé."
        )
        zalo_send_text(user_id, summary)
        push_history(user_id, "assistant", summary)
        return {"status": "summary"}

    if s["product"] == "chatbot":
        if "kb_status" not in s["info"]:
            s["info"]["kb_status"] = "waiting"
            zalo_send_text(user_id, "Bạn gửi **tài liệu (PDF/DOC)** hoặc **URL** để mình train chatbot nha." + contact_hint)
            return {"status": "ask_kb"}
        summary = (
            "✅ Tóm tắt đơn hàng Chatbot AI:\n"
            f"- Liên hệ: {s['info'].get('name', s['profile'].get('display_name',''))} | {s['info'].get('phone','?')} | {s['info'].get('email','?')}\n"
            "- Tài liệu: đang chờ bạn gửi.\n"
            "Bạn cần tích hợp Website/Fanpage v.v… không ạ?"
        )
        zalo_send_text(user_id, summary)
        push_history(user_id, "assistant", summary)
        return {"status": "summary"}

    # ===== Trả lời hội thoại tự nhiên (đời sống/khác) =====
    history_str = short_context(user_id, 6)
    rag_ctx = ""
    if faiss_index.ntotal > 0 and user_text:
        ctx_hits = retrieve(user_text, 4)
        rag_ctx = "\n\n".join(h["text"] for h in ctx_hits)

    reply = call_flash(
        system_prompt(s["profile"], s["salute"]),
        user_text,
        s["profile"],
        s["image_notes"],
        history_str,
        rag_ctx
    )
    # đính kèm nhắc contact nếu còn thiếu (không lặp lại nếu đã có đủ)
    reply_out = reply + (contact_hint if contact_hint and "Họ tên" not in reply else "")
    zalo_send_text(user_id, reply_out)

    push_history(user_id, "user", user_text or "[non-text]")
    push_history(user_id, "assistant", reply_out)
    s["image_notes"].clear()
    return {"status": "ok"}
