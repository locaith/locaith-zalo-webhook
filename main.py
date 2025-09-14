# main.py
# Locaith AI – Zalo OA CSKH Chatbot (Agent Architecture)
# - Planner (gemini-2.5-flash) định tuyến: reply trực tiếp / empathy / sales / web / vision
# - Vision (gemini-2.5-pro) OCR + mô tả ảnh
# - Web (Serper.dev) khi cần realtime
# - Dedupe event để tránh "double", chống spam, nhớ ngắn hạn từng user.
# - Lời thoại tự nhiên, sạch (không markdown đậm/nghiêng), chỉ gợi ý Locaith khi có tín hiệu.

import os, time, hmac, hashlib, json, re, io
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dotenv import load_dotenv
from PIL import Image

import google.generativeai as genai

# =================== ENV & INIT ===================
load_dotenv()

ZALO_OA_TOKEN     = os.getenv("ZALO_OA_TOKEN", "")
ZALO_APP_SECRET   = os.getenv("ZALO_APP_SECRET", "")
ENABLE_APPSECRET  = os.getenv("ENABLE_APPSECRET_PROOF", "false").lower() == "true"

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY", "")
ZALO_VERIFY_FILE  = os.getenv("ZALO_VERIFY_FILE")  # ví dụ "zalo123abc.html"
VERIFY_DIR        = "verify"

EMOJI_ENABLED     = os.getenv("EMOJI_ENABLED", "true").lower() == "true"
MAX_MSG_PER_30S   = int(os.getenv("MAX_MSG_PER_30S", "6"))
BAN_DURATION_SEC  = int(os.getenv("BAN_DURATION_SEC", str(24*3600)))
HISTORY_TURNS     = int(os.getenv("HISTORY_TURNS", "12"))

assert ZALO_OA_TOKEN and GEMINI_API_KEY, "Thiếu ZALO_OA_TOKEN hoặc GEMINI_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
MODEL_PLANNER   = "gemini-2.5-flash"
MODEL_RESPONDER = "gemini-2.5-flash"
MODEL_VISION    = "gemini-2.5-pro"

app = FastAPI(title="Locaith AI – Zalo OA", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =================== STATE ===================
# anti-spam
_rate: Dict[str, List[float]] = {}
_warn: Dict[str, int] = {}
_ban_until: Dict[str, float] = {}

# dedupe (giữ 500 event gần nhất)
_processed: Dict[str, float] = {}

# per-user session
_session: Dict[str, Dict[str, Any]] = {}  # user_id -> state


# =================== UTILITIES ===================
def _appsecret_proof(access_token: str, app_secret: str) -> str:
    return hmac.new(app_secret.encode(), access_token.encode(), hashlib.sha256).hexdigest()

def zalo_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json", "access_token": ZALO_OA_TOKEN}
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        h["appsecret_proof"] = _appsecret_proof(ZALO_OA_TOKEN, ZALO_APP_SECRET)
    return h

def zalo_send_text(user_id: str, text: str) -> dict:
    # Không markdown, không ký tự lạ; cắt gọn phòng length limit
    clean = (text or "").strip()
    if len(clean) > 4000:
        clean = clean[:3990] + "..."
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    payload = {"recipient": {"user_id": user_id}, "message": {"text": clean}}
    try:
        r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
        return r.json() if r.text else {}
    except Exception as e:
        print("Send error:", e)
        return {}

def zalo_get_profile(user_id: str) -> Dict[str, Any]:
    url = "https://openapi.zalo.me/v2.0/oa/getprofile"
    payload = {"user_id": user_id}
    try:
        r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
        return r.json().get("data", {}) if r.status_code == 200 else {}
    except Exception as e:
        print("Get profile error:", e)
        return {}

def emoji(s: str) -> str:
    return s if EMOJI_ENABLED else ""

def is_spamming(uid: str) -> bool:
    now = time.time()
    if uid in _ban_until and now < _ban_until[uid]:
        return True
    bucket = _rate.setdefault(uid, [])
    bucket.append(now)
    _rate[uid] = [t for t in bucket if now - t <= 30]
    return len(_rate[uid]) > MAX_MSG_PER_30S

def escalate_spam(uid: str) -> str:
    c = _warn.get(uid, 0) + 1
    _warn[uid] = c
    if c == 1:
        return "Tần suất tin nhắn hơi dày. Mình xin phép giảm nhịp một chút nhé. Nếu lặp lại mình sẽ tạm khóa 24 giờ."
    _ban_until[uid] = time.time() + BAN_DURATION_SEC
    return "Bạn đã bị tạm khóa tương tác 24 giờ do gửi quá nhiều tin nhắn trong thời gian ngắn."

def ensure_session(uid: str) -> Dict[str, Any]:
    return _session.setdefault(uid, {
        "welcomed": False,
        "profile": None,
        "salute": None,     # cách xưng hô do user cung cấp: "anh Tuấn", "chị Linh"...
        "history": [],      # [{role,text,ts}]
        "notes": [],        # ghi chú từ ảnh/voice (nội bộ, không lộ)
        "last_seen": time.time(),
    })

def push_history(uid: str, role: str, text: str):
    s = ensure_session(uid)
    s["history"].append({"role": role, "text": text, "ts": time.time()})
    if len(s["history"]) > HISTORY_TURNS:
        s["history"] = s["history"][-HISTORY_TURNS:]

def recent_context(uid: str, k: int = 8) -> str:
    s = ensure_session(uid)
    out = []
    for h in s["history"][-k:]:
        prefix = "USER:" if h["role"] == "user" else "ASSISTANT:"
        out.append(f"{prefix} {h['text']}")
    return "\n".join(out)

def already_processed(event_id: str) -> bool:
    if not event_id:
        return False
    now = time.time()
    if event_id in _processed:
        return True
    _processed[event_id] = now
    # dọn bớt
    if len(_processed) > 500:
        # xóa event cũ nhất
        oldest = sorted(_processed.items(), key=lambda x: x[1])[:50]
        for k, _ in oldest:
            _processed.pop(k, None)
    return False

# =================== SERPER (WEB) ===================
def web_search(query: str, n: int = 3) -> str:
    if not SERPER_API_KEY:
        return ""
    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": n, "gl": "vn", "hl": "vi"}
        r = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=12)
        if r.status_code != 200:
            return ""
        data = r.json()
        lines = []
        if "answerBox" in data and data["answerBox"].get("answer"):
            lines.append(f"Trả lời nhanh: {data['answerBox']['answer']}")
        for it in (data.get("organic") or [])[:n]:
            t = it.get("title", "")
            s = it.get("snippet", "")
            u = it.get("link", "")
            if t or s:
                lines.append(f"- {t}. {s} (Nguồn: {u})")
        return "\n".join(lines)
    except Exception as e:
        print("Serper error:", e)
        return ""

# =================== AGENTS ===================
def planner(profile: Dict[str, Any], salute: Optional[str], text: str, has_image: bool, event_name: str) -> Dict[str, Any]:
    """
    Quyết định: reply trực tiếp hay cần web/vision/empathy/sales.
    Trả về dict:
      {intent, need_web, need_empathy, need_sales, concise}
    """
    # Quy tắc nhanh trước cho các intent rõ ràng
    t = (text or "").lower()
    if has_image:
        return {"intent": "VISION", "need_web": False, "need_empathy": False, "need_sales": False, "concise": True}
    if event_name == "user_send_sticker":
        return {"intent": "STICKER", "need_web": False, "need_empathy": True, "need_sales": False, "concise": True}
    if any(k in t for k in ["giá", "bảng giá", "bao nhiêu", "triển khai", "website", "landing", "chatbot", "locaith"]):
        return {"intent": "SALES", "need_web": False, "need_empathy": False, "need_sales": True, "concise": False}
    # heuristic cần thông tin realtime
    realtime_kw = ["hôm nay", "mới nhất", "tin tức", "giá", "tỷ giá", "lịch", "thời tiết", "kết quả", "promote", "khuyến mãi", "tuyển dụng"]
    need_web = any(k in t for k in realtime_kw)
    # Nhẹ nhàng đồng cảm khi người dùng nói về cuộc sống, sức khỏe, cảm xúc
    empathy_kw = ["mệt", "buồn", "lo", "căng thẳng", "khó chịu", "con mình", "gia đình", "áp lực"]
    need_empathy = any(k in t for k in empathy_kw)
    return {"intent": "GENERAL", "need_web": need_web, "need_empathy": need_empathy, "need_sales": False, "concise": False}

def agent_vision(image_bytes: bytes) -> str:
    """
    OCR + mô tả ảnh ngắn gọn. Không lộ đây là ghi chú nội bộ.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "Ảnh không đọc được."
    prompt = (
        "Hãy đọc nội dung xuất hiện trong ảnh (OCR) và mô tả ngắn gọn ảnh đang nói về điều gì. "
        "Chỉ trả về văn bản súc tích, không thêm nhận xét thừa."
    )
    model = genai.GenerativeModel(MODEL_VISION)
    resp = model.generate_content([prompt, image])
    try:
        return resp.text.strip()
    except Exception:
        return "Không trích xuất được nội dung từ ảnh."

def agent_web(query: str) -> str:
    result = web_search(query, 3)
    return result or ""

def agent_responder(system_note: str, user_text: str, ctx: str, web_ctx: str, vision_ctx: str, mode: str, profile: Dict[str, Any], salute: Optional[str]) -> str:
    """
    mode: GENERAL | EMPATHY | SALES | STICKER | VISION
    """
    dn = profile.get("display_name") or "bạn"
    call = salute or dn
    style_rules = (
        "Bạn là một người Việt nói chuyện tự nhiên, dùng lời giản dị, không định dạng markdown, không ký tự đặc biệt."
        " Xưng hô thân thiện: xưng 'mình' và gọi đối phương là '" + call + "'."
        " Tránh liệt kê khô khan; ưu tiên một đoạn văn rõ ràng, cuối cùng có một câu hỏi ngắn để giữ nhịp hội thoại."
    )
    loc_hint = (
        "Khi và chỉ khi người dùng hỏi về Locaith hoặc sản phẩm liên quan (Chatbot AI, Website, Landing page),"
        " hãy gợi ý rất nhẹ nhàng rằng Locaith có thể hỗ trợ. Nếu người dùng hỏi cụ thể, hãy tư vấn theo ngôn ngữ đời thường."
    )
    mode_hint = {
        "GENERAL": "Trả lời câu hỏi hoặc trò chuyện bình thường.",
        "EMPATHY": "Ưu tiên lắng nghe và đồng cảm, hỏi mở và giúp người dùng gỡ rối.",
        "SALES": "Khám phá nhu cầu, hỏi ngắn gọn bối cảnh. Không bán hàng khiên cưỡng. Chỉ đề cập Locaith khi phù hợp.",
        "STICKER": "Phản hồi ngắn gọn, thân thiện khi người dùng gửi sticker.",
        "VISION": "Giải thích ngắn gọn dựa trên phần vision_ctx.",
    }.get(mode, "Trả lời tự nhiên.")
    web_part = f"\n\nThông tin từ internet:\n{web_ctx}" if web_ctx else ""
    vision_part = f"\n\nThông tin rút ra từ ảnh:\n{vision_ctx}" if vision_ctx else ""

    content = (
        f"{style_rules}\n{loc_hint}\nChế độ: {mode_hint}\n\n"
        f"Ngữ cảnh gần đây:\n{ctx or '(trống)'}\n"
        f"{web_part}{vision_part}\n\n"
        f"Tin nhắn của người dùng:\n{user_text}"
    )
    model = genai.GenerativeModel(MODEL_RESPONDER)
    resp = model.generate_content(
        [{"role": "user", "parts": [system_note]},
         {"role": "user", "parts": [content]}],
        generation_config={"temperature": 0.6}
    )
    try:
        return resp.text.strip()
    except Exception:
        return "Xin lỗi, mình đang hơi bận. Bạn nhắn lại giúp mình sau một lát nhé."

# =================== EVENT HELPERS ===================
def parse_salute(text: str) -> Optional[str]:
    # bắt các cụm "anh|chị|em + Tên"
    m = re.search(r"\b(anh|chị|em)\s+[A-Za-zÀ-ỹĐđ][\wÀ-ỹĐđ\s]*", text or "", flags=re.IGNORECASE)
    return m.group(0).strip() if m else None

def extract_event_id(evt: dict) -> str:
    return evt.get("event_id") or evt.get("timestamp", "") + "_" + evt.get("event_name", "")

def get_text(evt: dict) -> str:
    return ((evt.get("message") or {}).get("text") or "").strip()

def get_image_bytes(evt: dict) -> Optional[bytes]:
    att = (evt.get("message") or {}).get("attachments") or []
    for a in att:
        if a.get("type") == "image":
            url = (a.get("payload") or {}).get("url") or (a.get("payload") or {}).get("href")
            if not url:
                continue
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and r.content:
                    return r.content
            except Exception:
                pass
    return None

# =================== PROMPTS (system note) ===================
def system_note(profile: Dict[str, Any], salute: Optional[str]) -> str:
    name = profile.get("display_name") or "bạn"
    who = salute or name
    return (
        "Bạn là một trợ lý hội thoại tiếng Việt, nói chuyện tự nhiên như người thật, lịch sự và kín đáo."
        f" Gọi đối phương là '{who}'. Không dùng ký tự lạ, không định dạng đậm/nghiêng."
        " Tôn trọng quyền riêng tư. Trả lời gọn, ấm và có cảm xúc vừa phải."
    )

def welcome_line(profile: Dict[str, Any]) -> str:
    name = profile.get("display_name") or "bạn"
    w = f"Chào {name}. Rất vui được trò chuyện cùng bạn."
    if EMOJI_ENABLED:
        w += " 🙂"
    return w

# =================== ROUTES ===================
@app.on_event("startup")
async def on_start():
    print("Locaith AI – Zalo webhook (Agent) started.")

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "ts": time.time()}

@app.get("/{verify_name}")
def zalo_verify(verify_name: str):
    if ZALO_VERIFY_FILE and verify_name == ZALO_VERIFY_FILE:
        path = os.path.join(VERIFY_DIR, ZALO_VERIFY_FILE)
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404)

@app.get("/zalo/webhook")
def webhook_verify(challenge: str = ""):
    return {"challenge": challenge} if challenge else {"error": "missing challenge"}

@app.post("/zalo/webhook")
async def webhook(req: Request):
    # verify signature nếu bật
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        try:
            body = await req.body()
            signature = req.headers.get("X-ZEvent-Signature", "")
            expected = hmac.new(ZALO_APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return {"status": "invalid_signature"}
        except Exception:
            pass

    event = await req.json()
    print("EVENT:", json.dumps(event, ensure_ascii=False))

    event_id = extract_event_id(event)
    if already_processed(event_id):
        return {"status": "duplicate_ignored"}

    event_name = event.get("event_name", "")
    user_id = (event.get("sender") or {}).get("id")
    if not user_id:
        return {"status": "no_user"}

    # anti-spam
    if is_spamming(user_id):
        zalo_send_text(user_id, escalate_spam(user_id))
        return {"status": "spam"}

    s = ensure_session(user_id)
    if not s["profile"]:
        s["profile"] = zalo_get_profile(user_id)

    # follow/unfollow
    if event_name == "follow":
        if not s["welcomed"]:
            msg = welcome_line(s["profile"])
            zalo_send_text(user_id, msg)
            s["welcomed"] = True
            push_history(user_id, "assistant", msg)
        return {"status": "ok"}

    if user_id in _ban_until and time.time() < _ban_until[user_id]:
        return {"status": "banned"}

    # ===== gather input =====
    text = get_text(event)
    salute = parse_salute(text) or s.get("salute")
    if salute and s.get("salute") != salute:
        s["salute"] = salute

    img_bytes = get_image_bytes(event) if event_name == "user_send_image" else None
    has_image = bool(img_bytes)

    # Nếu là các event không có text
    if event_name in ["user_send_sticker", "user_send_gif", "user_send_audio", "user_send_video", "user_send_file", "user_send_location"] and not text:
        # phản hồi một câu ngắn, không salesy
        short = {
            "user_send_sticker": "Mình nhận được sticker rồi.",
            "user_send_gif": "Mình nhận được ảnh động rồi.",
            "user_send_audio": "Mình đã nhận voice.",
            "user_send_video": "Mình đã nhận video.",
            "user_send_file": "Mình đã nhận file.",
            "user_send_location": "Mình đã nhận vị trí.",
        }[event_name]
        zalo_send_text(user_id, short)
        push_history(user_id, "assistant", short)
        return {"status": "ok"}

    # ===== planner decides =====
    plan = planner(s["profile"] or {}, s.get("salute"), text, has_image, event_name)
    mode = "GENERAL"
    web_ctx = ""
    vision_ctx = ""

    if has_image:
        vision_ctx = agent_vision(img_bytes)
        s["notes"].append(vision_ctx)
        mode = "VISION"

    if plan.get("need_web") and text:
        web_ctx = agent_web(text)

    # empathy or sales mode
    if plan.get("need_sales"):
        mode = "SALES"
    elif plan.get("need_empathy"):
        mode = "EMPATHY"
    elif event_name == "user_send_sticker":
        mode = "STICKER"

    # first-time welcome on first meaningful message
    if not s["welcomed"]:
        w = welcome_line(s["profile"])
        push_history(user_id, "assistant", w)
        zalo_send_text(user_id, w)
        s["welcomed"] = True
        # Không gửi thêm gì nữa trong event này để tránh double nếu chỉ là lời chào trống
        # nhưng nếu user có text thật thì vẫn tiếp tục trả lời dưới đây.

    # ===== compose final reply =====
    push_history(user_id, "user", text or "[non-text]")

    final = agent_responder(
        system_note(s["profile"] or {}, s.get("salute")),
        text,
        recent_context(user_id, 8),
        web_ctx,
        vision_ctx,
        mode,
        s["profile"] or {},
        s.get("salute"),
    )

    zalo_send_text(user_id, final)
    push_history(user_id, "assistant", final)

    # clear one-time notes
    s["notes"].clear()
    s["last_seen"] = time.time()
    return {"status": "ok"}

# ---- Optional: simple KB ingest via URL (RAG có thể thêm sau) ----
@app.post("/kb/url")
def kb_url(user_id: str = Form(...), url: str = Form(...)):
    # placeholder an toàn; có thể mở rộng RAG sau
    return {"ok": True, "note": "Endpoint placeholder. Bạn có thể mở rộng RAG nếu cần."}
