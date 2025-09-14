# main.py
# Locaith AI – Zalo OA Chatbot (Agent Planner)
# - Fix duplicate greeting; natural VN tone; no markdown
# - Agents: planner (flash), responder (flash), vision (pro), sticker-mood (pro), web (serper.dev)
# - Sticker => đoán cảm xúc; Image => OCR/mô tả; Web => gold/coin/stock/forex/news
# - Dedupe event; anti-spam; per-user short memory

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

app = FastAPI(title="Locaith AI – Zalo OA", version="3.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =================== STATE ===================
_rate: Dict[str, List[float]] = {}
_warn: Dict[str, int] = {}
_ban_until: Dict[str, float] = {}
_processed: Dict[str, float] = {}  # dedupe
_session: Dict[str, Dict[str, Any]] = {}    # per-user

# =================== UTILITIES ===================
def _appsecret_proof(access_token: str, app_secret: str) -> str:
    return hmac.new(app_secret.encode(), access_token.encode(), hashlib.sha256).hexdigest()

def zalo_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json", "access_token": ZALO_OA_TOKEN}
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        h["appsecret_proof"] = _appsecret_proof(ZALO_OA_TOKEN, ZALO_APP_SECRET)
    return h

def zalo_send_text(user_id: str, text: str) -> dict:
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
        return "Tin nhắn hơi dày, mình xin phép giảm nhịp một chút nhé. Nếu lặp lại mình sẽ tạm khóa 24 giờ."
    _ban_until[uid] = time.time() + BAN_DURATION_SEC
    return "Bạn đã bị tạm khóa tương tác 24 giờ do gửi quá nhiều tin trong thời gian ngắn."

def ensure_session(uid: str) -> Dict[str, Any]:
    return _session.setdefault(uid, {
        "welcomed": False,
        "profile": None,
        "salute": None,   # cách xưng hô do user cung cấp
        "history": [],    # [{role,text,ts}]
        "notes": [],      # ảnh/voice notes (nội bộ)
        "last_seen": time.time(),
    })

def push_history(uid: str, role: str, text: str):
    s = ensure_session(uid)
    s["history"].append({"role": role, "text": text, "ts": time.time()})
    if len(s["history"]) > HISTORY_TURNS:
        s["history"] = s["history"][-HISTORY_TURNS:]

def recent_context(uid: str, k: int = 8) -> str:
    s = ensure_session(uid)
    return "\n".join(
        ("USER: " if h["role"] == "user" else "ASSISTANT: ") + h["text"]
        for h in s["history"][-k:]
    )

def already_processed(event_id: str) -> bool:
    if not event_id:
        return False
    if event_id in _processed:
        return True
    _processed[event_id] = time.time()
    if len(_processed) > 500:
        for k, _ in sorted(_processed.items(), key=lambda x: x[1])[:50]:
            _processed.pop(k, None)
    return False

# =================== EXTRACTORS ===================
def parse_salute(text: str) -> Optional[str]:
    m = re.search(r"\b(anh|chị|em)\s+[A-Za-zÀ-ỹĐđ][\wÀ-ỹĐđ\s]*", text or "", flags=re.IGNORECASE)
    return m.group(0).strip() if m else None

def extract_event_id(evt: dict) -> str:
    return evt.get("event_id") or f"{evt.get('timestamp','')}_{evt.get('event_name','')}"

def get_text(evt: dict) -> str:
    return ((evt.get("message") or {}).get("text") or "").strip()

def get_image_or_sticker_bytes(evt: dict) -> Optional[bytes]:
    att = (evt.get("message") or {}).get("attachments") or []
    for a in att:
        if a.get("type") in ("image", "sticker"):
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

# =================== SERPER (WEB) ===================
def serper_search(q: str, n: int = 3) -> str:
    if not SERPER_API_KEY:
        return ""
    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": q, "num": n, "gl": "vn", "hl": "vi"}
        r = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=12)
        if r.status_code != 200:
            return ""
        data = r.json()
        lines = []
        if "answerBox" in data and data["answerBox"].get("answer"):
            lines.append(f"Trả lời nhanh: {data['answerBox']['answer']}")
        for it in (data.get("organic") or [])[:n]:
            t = it.get("title", "") or ""
            s = it.get("snippet", "") or ""
            u = it.get("link", "") or ""
            if t or s:
                lines.append(f"- {t}. {s} (Nguồn: {u})")
        return "\n".join(lines)
    except Exception as e:
        print("Serper error:", e)
        return ""

def build_query(text: str) -> Optional[str]:
    t = (text or "").lower()
    # gold
    if "giá vàng" in t or "sjc" in t:
        return "giá vàng SJC hôm nay"
    # forex
    mfx = re.search(r"tỷ giá\s+([a-z]{3})/([a-z]{3})", t)
    if mfx:
        return f"tỷ giá {mfx.group(1).upper()}/{mfx.group(2).upper()} hôm nay"
    if "tỷ giá usd" in t or "usd" in t and "tỷ giá" in t:
        return "tỷ giá USD VND hôm nay"
    # crypto
    if "bitcoin" in t or re.search(r"\bbtc\b", t):
        return "giá BTC hôm nay"
    if "ethereum" in t or re.search(r"\beth\b", t):
        return "giá ETH hôm nay"
    mcoin = re.search(r"giá\s+([a-z0-9]{2,10})\b", t)
    if mcoin:
        return f"giá {mcoin.group(1).upper()} hôm nay"
    # stock (VN tickers 3 chữ cái hoa)
    mt = re.search(r"\b([A-Z]{3,4})\b", text)
    if mt and mt.group(1).isupper():
        tk = mt.group(1)
        return f"giá cổ phiếu {tk} hôm nay"
    # generic news/price
    if any(k in t for k in ["giá", "hôm nay", "mới nhất", "tin tức", "kết quả", "thời tiết"]):
        return text
    return None

# =================== AGENTS ===================
def planner(text: str, has_image: bool, event_name: str) -> Dict[str, Any]:
    t = (text or "").lower()
    if has_image:
        return {"mode": "VISION", "need_web": False, "need_empathy": False, "need_sales": False}
    if event_name == "user_send_sticker":
        return {"mode": "STICKER", "need_web": False, "need_empathy": True, "need_sales": False}
    # sales cue
    if any(k in t for k in ["locaith", "chatbot", "website", "landing", "giải pháp", "triển khai", "báo giá"]):
        return {"mode": "SALES", "need_web": False, "need_empathy": False, "need_sales": True}
    # realtime cue
    need_web = build_query(t) is not None
    # empathy cue
    empathy_kw = ["mệt", "buồn", "lo", "chán", "khó chịu", "áp lực", "con mình", "gia đình", "căng thẳng"]
    need_empathy = any(k in t for k in empathy_kw)
    return {"mode": "GENERAL", "need_web": need_web, "need_empathy": need_empathy, "need_sales": False}

def agent_vision_summary(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "Ảnh không đọc được."
    prompt = (
        "Hãy đọc nội dung có trong ảnh (OCR nếu có chữ) và tóm tắt ngắn gọn ảnh nói về điều gì. "
        "Chỉ trả văn bản súc tích."
    )
    model = genai.GenerativeModel(MODEL_VISION)
    resp = model.generate_content([prompt, image])
    try:
        return resp.text.strip()
    except Exception:
        return "Không trích xuất được nội dung từ ảnh."

def agent_sticker_mood(image_bytes: Optional[bytes]) -> str:
    # Nếu Zalo cung cấp ảnh của sticker -> dùng thị giác để suy đoán cảm xúc
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt = (
                "Đây là ảnh sticker. Dựa vào nét mặt/cử chỉ, hãy suy đoán cảm xúc chính của sticker "
                "(một trong các nhóm: vui, buồn, sốc, giận, dỗi, chán, phấn khích, bình thản, yêu thương). "
                "Trả về đúng một nhãn tiếng Việt."
            )
            model = genai.GenerativeModel(MODEL_VISION)
            resp = model.generate_content([prompt, image])
            mood = (resp.text or "").strip().lower()
            return mood[:40]
        except Exception:
            pass
    # fallback
    return "khó đoán"

def agent_web_answer(text: str) -> str:
    q = build_query(text)
    if not q:
        return ""
    return serper_search(q, 3) or ""

def system_note(profile: Dict[str, Any], salute: Optional[str], avoid_greeting: bool) -> str:
    name = (profile or {}).get("display_name") or "bạn"
    who = salute or name
    note = (
        "Bạn là một trợ lý hội thoại tiếng Việt, nói chuyện tự nhiên như người thật, kín đáo và lịch sự. "
        f"Gọi đối phương là '{who}'. Không dùng ký tự lạ hay định dạng đậm/nghiêng. "
        "Chốt một câu hỏi ngắn cuối câu để giữ nhịp hội thoại."
    )
    if avoid_greeting:
        note += " Không mở đầu bằng câu chào; đi thẳng vào nội dung."
    return note

def agent_responder(profile: Dict[str, Any], salute: Optional[str], user_text: str,
                    ctx: str, web_ctx: str, vision_ctx: str, mood_ctx: str,
                    mode: str, avoid_greeting: bool) -> str:
    style = (
        "Phong cách: tự nhiên, giản dị, không liệt kê quá khô, dùng câu ngắn dễ hiểu."
        " Chỉ gợi ý giải pháp Locaith khi người dùng chủ động hỏi hoặc có tín hiệu rõ ràng."
    )
    mode_hint = {
        "GENERAL": "Trả lời hoặc trò chuyện bình thường.",
        "EMPATHY": "Ưu tiên lắng nghe và đồng cảm.",
        "SALES": "Khám phá nhu cầu, hỏi bối cảnh ngắn gọn; không bán hàng khiên cưỡng.",
        "STICKER": "Phản hồi dựa trên cảm xúc ước lượng từ sticker.",
        "VISION": "Giải thích dựa trên thông tin từ ảnh.",
    }.get(mode, "Trò chuyện tự nhiên.")
    web_part = f"\n\nThông tin từ internet:\n{web_ctx}" if web_ctx else ""
    vision_part = f"\n\nThông tin rút ra từ ảnh:\n{vision_ctx}" if vision_ctx else ""
    mood_part = f"\n\nTâm trạng ước lượng từ sticker: {mood_ctx}" if mood_ctx else ""

    content = (
        f"{style}\nChế độ: {mode_hint}\n\n"
        f"Ngữ cảnh gần đây:\n{ctx or '(trống)'}{web_part}{vision_part}{mood_part}\n\n"
        f"Tin nhắn của người dùng:\n{user_text}"
    )
    model = genai.GenerativeModel(MODEL_RESPONDER)
    resp = model.generate_content(
        [{"role": "user", "parts": [system_note(profile, salute, avoid_greeting)]},
         {"role": "user", "parts": [content]}],
        generation_config={"temperature": 0.6}
    )
    try:
        return resp.text.strip()
    except Exception:
        return "Xin lỗi, mình đang hơi bận. Bạn nhắn lại giúp mình sau một lát nhé."

# =================== WELCOME ===================
def welcome_line(profile: Dict[str, Any]) -> str:
    name = (profile or {}).get("display_name") or "bạn"
    w = f"Chào {name}. Rất vui được trò chuyện cùng bạn."
    if EMOJI_ENABLED:
        w += " " + emoji("🙂")
    return w

# =================== ROUTES ===================
@app.on_event("startup")
async def on_start():
    print("Locaith AI – Zalo webhook (Agent) started 3.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.1.0", "ts": time.time()}

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
    # verify signature (optional)
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
            zalo_send_text(user_id, "Bạn muốn mình giúp gì hay có điều gì muốn chia sẻ không?")
            push_history(user_id, "assistant", msg)
            push_history(user_id, "assistant", "Hôm nay bạn muốn mình giúp điều gì?")
            s["welcomed"] = True
            avoid_greeting = True
        return {"status": "ok"}

    if user_id in _ban_until and time.time() < _ban_until[user_id]:
        return {"status": "banned"}

    # gather input
    text = get_text(event)
    salute = parse_salute(text) or s.get("salute")
    if salute and s.get("salute") != salute:
        s["salute"] = salute

    img_bytes = get_image_or_sticker_bytes(event) if event_name in ["user_send_image","user_send_sticker"] else None
    has_image = bool(img_bytes) and event_name == "user_send_image"

    # some non-text-only events
    if event_name in ["user_send_gif", "user_send_audio", "user_send_video", "user_send_file", "user_send_location"] and not text:
        short = {
            "user_send_gif": "Mình đã nhận ảnh động.",
            "user_send_audio": "Mình đã nhận voice.",
            "user_send_video": "Mình đã nhận video.",
            "user_send_file": "Mình đã nhận file.",
            "user_send_location": "Mình đã nhận vị trí.",
        }[event_name]
        zalo_send_text(user_id, short)
        push_history(user_id, "assistant", short)
        return {"status": "ok"}

    # planner
    plan = planner(text, has_image, event_name)
    mode = plan["mode"]
    web_ctx = ""
    vision_ctx = ""
    mood_ctx = ""

    # sticker mood
    if event_name == "user_send_sticker":
        mood_ctx = agent_sticker_mood(img_bytes)
        mode = "EMPATHY"  # chuyển sang thấu hiểu

    # image OCR/summary
    justWelcomed = False
    if has_image:
        vision_ctx = agent_vision_summary(img_bytes)
        mode = "VISION"

    # realtime search
    if plan.get("need_web") and text:
        web_ctx = agent_web_answer(text)

    # first-time welcome on first meaningful message
    avoid_greeting = False
    if not s["welcomed"]:
        msg = welcome_line(s["profile"])
        zalo_send_text(user_id, msg)             # gửi 1 lần duy nhất
        push_history(user_id, "assistant", msg)
        s["welcomed"] = True
        justWelcomed = True
        avoid_greeting = True                    # cấm responder lặp “Chào …”

    # build final reply
    if text or vision_ctx or mood_ctx:
        push_history(user_id, "user", text or "[non-text]")

        final = agent_responder(
            s["profile"], s.get("salute"), text,
            recent_context(user_id, 8), web_ctx, vision_ctx, mood_ctx,
            mode, avoid_greeting
        )

        # Nếu chỉ là lời chào ngắn của người dùng và vừa gửi welcome,
        # chỉnh final thành một câu hỏi mở, không lặp lại chào
        if justWelcomed and re.fullmatch(r"(chào|xin chào|alo|hi|hello)[!.\s]*", (text or "").lower()):
            final = "Hôm nay bạn muốn mình giúp điều gì?"

        zalo_send_text(user_id, final)
        push_history(user_id, "assistant", final)

    s["notes"].clear()
    s["last_seen"] = time.time()
    return {"status": "ok"}

# ---- Optional: simple endpoint để tích hợp URL tài liệu sau này ----
@app.post("/kb/url")
def kb_url(user_id: str = Form(...), url: str = Form(...)):
    return {"ok": True, "note": "Placeholder ingest. Có thể mở rộng RAG sau."}
