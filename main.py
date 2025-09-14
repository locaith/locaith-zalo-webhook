# main.py
# Locaith AI – Zalo OA Chatbot (v3.4.0)
# - Tự động chia nhỏ tin dài, gửi nối tiếp (chunked send)
# - Dedupe chắc hơn bằng message.msg_id
# - Giữ toàn bộ Multi-Agent + Crypto TA snapshot + OpenWeather + Vision + Sticker mood + Brand guard + Runtime Constitution

import os, time, hmac, hashlib, json, re, io, math
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
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

ZALO_VERIFY_FILE  = os.getenv("ZALO_VERIFY_FILE")
VERIFY_DIR        = "verify"

EMOJI_ENABLED     = os.getenv("EMOJI_ENABLED", "true").lower() == "true"
MAX_MSG_PER_30S   = int(os.getenv("MAX_MSG_PER_30S", "6"))
BAN_DURATION_SEC  = int(os.getenv("BAN_DURATION_SEC", str(24*3600)))
HISTORY_TURNS     = int(os.getenv("HISTORY_TURNS", "12"))

# ====== NGƯỠNG CHIA TIN NHẮN (có thể chỉnh qua ENV) ======
ZALO_CHUNK_LIMIT  = int(os.getenv("ZALO_CHUNK_LIMIT", "900"))  # an toàn ~900 ký tự
ZALO_CHUNK_PAUSE  = float(os.getenv("ZALO_CHUNK_PAUSE", "0.25"))  # nghỉ giữa các phần

PROMPT_CONFIG_PATH = os.getenv("PROMPT_CONFIG_PATH", "").strip()

# Brand guard
BRAND_NAME     = "Locaith AI"
BRAND_DEVLINE  = "được đội ngũ founder của Locaith phát triển."
BRAND_OFFERING = "các giải pháp Chatbot AI và Website (website hoàn chỉnh hoặc landing page)."
BRAND_INTRO    = f"{BRAND_NAME} là một startup Việt, {BRAND_DEVLINE} Chúng mình cung cấp {BRAND_OFFERING}"

assert ZALO_OA_TOKEN and GEMINI_API_KEY, "Thiếu ZALO_OA_TOKEN hoặc GEMINI_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
MODEL_PLANNER   = "gemini-2.5-flash"
MODEL_RESPONDER = "gemini-2.5-flash"
MODEL_VISION    = "gemini-2.5-pro"

app = FastAPI(title="Locaith AI – Zalo OA", version="3.4.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =================== STATE ===================
_rate: Dict[str, List[float]] = {}
_warn: Dict[str, int] = {}
_ban_until: Dict[str, float] = {}
_processed: Dict[str, float] = {}  # dedupe
_session: Dict[str, Dict[str, Any]] = {}

# =================== UTILITIES ===================
def _appsecret_proof(access_token: str, app_secret: str) -> str:
    return hmac.new(app_secret.encode(), access_token.encode(), hashlib.sha256).hexdigest()

def zalo_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json", "access_token": ZALO_OA_TOKEN}
    if ENABLE_APPSECRET and ZALO_APP_SECRET:
        h["appsecret_proof"] = _appsecret_proof(ZALO_OA_TOKEN, ZALO_APP_SECRET)
    return h

# ---- Low-level: gửi 1 tin nhắn đơn ----
def _zalo_send_text_once(user_id: str, text: str) -> dict:
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    payload = {"recipient": {"user_id": user_id}, "message": {"text": text}}
    try:
        r = requests.post(url, headers=zalo_headers(), json=payload, timeout=15)
        return r.json() if r.text else {}
    except Exception as e:
        print("Send error:", e)
        return {}

# ---- Split & send: tự động chia nhỏ tin dài, đánh số (1/3)… ----
def _smart_split(text: str, limit: int) -> List[str]:
    s = (text or "").strip()
    if len(s) <= limit:
        return [s]
    # tách theo đoạn trước
    parts: List[str] = []
    paras = re.split(r"\n{2,}", s)
    buf = ""
    def flush():
        nonlocal buf
        if buf.strip():
            parts.append(buf.strip())
            buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) > limit:
            # tách tiếp theo câu
            sentences = re.split(r"(?<=[\.\!\?\…;:])\s+", p)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) > limit:
                    # cuối cùng: cắt cứng
                    for i in range(0, len(sent), limit):
                        chunk = sent[i:i+limit]
                        if len(buf) + len(chunk) + 1 > limit:
                            flush()
                        buf += ((" " if buf else "") + chunk)
                else:
                    if len(buf) + len(sent) + 1 > limit:
                        flush()
                    buf += ((" " if buf else "") + sent)
            flush()
        else:
            if len(buf) + len(p) + 2 > limit:
                flush()
            buf += (("\n\n" if buf else "") + p)
    flush()
    return parts

def zalo_send_text(user_id: str, text: str) -> None:
    clean = (text or "").strip()
    if not clean:
        return
    chunks = _smart_split(clean, ZALO_CHUNK_LIMIT)
    total = len(chunks)
    for idx, ch in enumerate(chunks, 1):
        prefix = f"({idx}/{total}) " if total > 1 else ""
        _zalo_send_text_once(user_id, prefix + ch)
        if idx < total:
            time.sleep(ZALO_CHUNK_PAUSE)

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
        "welcomed": False, "profile": None, "salute": None,
        "history": [], "notes": [], "last_seen": time.time(),
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

# =================== DEDUPE CHẮC HƠN ===================
def extract_event_id(evt: dict) -> str:
    # Ưu tiên msg_id của Zalo; fallback timestamp+event_name+hash(text)
    mid = (evt.get("message") or {}).get("msg_id")
    if mid:
        return f"msg_{mid}"
    ts = str(evt.get("timestamp", ""))
    ev = evt.get("event_name", "")
    txt = ((evt.get("message") or {}).get("text") or "")
    hx = hashlib.sha256((ts + ev + txt).encode("utf-8")).hexdigest()[:16]
    return f"{ts}_{ev}_{hx}"

def already_processed(event_id: str) -> bool:
    if not event_id:
        return False
    if event_id in _processed:
        return True
    _processed[event_id] = time.time()
    if len(_processed) > 800:
        for k, _ in sorted(_processed.items(), key=lambda x: x[1])[:200]:
            _processed.pop(k, None)
    return False

# =================== RUNTIME CONSTITUTION ===================
def load_runtime_prompt() -> Dict[str, Any]:
    cfg = {"raw_text": "", "json": None}
    p = PROMPT_CONFIG_PATH
    if not p:
        return cfg
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = f.read()
        try:
            cfg["json"] = json.loads(data)
        except Exception:
            cfg["raw_text"] = data
    except Exception as e:
        print("Cannot load prompt file:", e)
    return cfg

RUNTIME_PROMPT = load_runtime_prompt()

# =================== EXTRACTORS ===================
def parse_salute(text: str) -> Optional[str]:
    m = re.search(r"\b(anh|chị|em)\s+[A-Za-zÀ-ỹĐđ][\wÀ-ỹĐđ\s]*", text or "", flags=re.IGNORECASE)
    return m.group(0).strip() if m else None

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

def build_crypto_queries(text: str) -> List[str]:
    t = (text or "").lower()
    symbol = None
    m = re.search(r"\b([A-Z]{2,5})\b", text)
    if m: symbol = m.group(1)
    if "bitcoin" in t or re.search(r"\bbtc\b", t): symbol = "BTC"
    if "ethereum" in t or re.search(r"\beth\b", t): symbol = "ETH"
    if "wld" in t: symbol = "WLD"
    qs = []
    if symbol:
        qs += [
            f"giá {symbol} hôm nay",
            f"{symbol} technical analysis RSI MACD today",
            f"{symbol} order book sentiment news today",
            f"{symbol} khối lượng giao dịch hôm nay",
        ]
    else:
        qs += ["giá crypto hôm nay", "bitcoin price today"]
    return qs

def summarize_crypto_from_web(text: str) -> str:
    queries = build_crypto_queries(text)
    blocks = []
    for q in queries:
        res = serper_search(q, 3)
        if res:
            blocks.append(f"• Query: {q}\n{res}")
    return "\n\n".join(blocks)

# =================== OPENWEATHER ===================
def get_weather_snapshot(text: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "Chưa cấu hình OpenWeather API key."
    # Tìm city
    city = None
    m = re.search(r"(thời tiết|nhiệt độ|mưa)\s+(ở|tại)\s+([^?.,!]+)", text.lower())
    if m: city = m.group(3).strip()
    if not city:
        mc = re.search(r"(?:ở|tại)\s+([^?.,!]+)$", text.lower())
        if mc: city = mc.group(1).strip()
    if not city:
        mc = re.search(r"(thời tiết|nhiệt độ)\s+([^?.,!]+)$", text.lower())
        if mc: city = mc.group(2).strip()

    city_q = city or "Hanoi"
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_q, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "vi"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return f"Không tìm được thời tiết cho “{city_q}”."

        d = r.json()
        name  = d.get("name", city_q)
        desc  = d.get("weather", [{}])[0].get("description", "")
        main  = d.get("main", {}) or {}
        windd = d.get("wind", {}) or {}

        temp  = main.get("temp", None)
        feels = main.get("feels_like", None)
        hum   = main.get("humidity", None)
        wind  = windd.get("speed", None)

        line = f"Thời tiết {name}: {desc}. Nhiệt độ {temp}°C (cảm giác {feels}°C), độ ẩm {hum}%, gió {wind} m/s."
        advice = _weather_advice(desc, float(temp or 0), float(feels or 0), int(hum or 0), float(wind or 0))
        if advice:
            line += f"\n{advice}"
        return line
    except Exception as e:
        print("OpenWeather error:", e)
        return "Mình không lấy được thông tin thời tiết lúc này."

def _weather_advice(desc: str, temp: float, feels: float, hum: int, wind: float) -> str:
    desc_l = (desc or "").lower()
    tips = []
    if any(k in desc_l for k in ["mưa", "dông", "giông", "mưa rào"]):
        tips.append("mang áo mưa/áo khoác chống nước")
    if any(k in desc_l for k in ["nắng", "nắng nóng", "nắng gắt", "quang mây"]) and (temp is not None and temp >= 34):
        tips.append("uống đủ nước, hạn chế ra nắng gắt buổi trưa")
    if temp is not None and temp <= 20:
        tips.append("mặc ấm khi ra ngoài")
    if hum is not None and hum >= 85:
        tips.append("cẩn thận đường trơn, mang khăn giấy/áo mưa mỏng")
    if wind is not None and wind >= 10:  # ~36 km/h
        tips.append("chú ý gió mạnh khi di chuyển bằng xe máy")
    return ("Lời khuyên: " + "; ".join(tips)) if tips else ""

# =================== AGENTS ===================
def planner(text: str, has_image: bool, event_name: str) -> Dict[str, Any]:
    t = (text or "").lower()
    if has_image:
        return {"mode": "VISION", "need_web": False, "need_empathy": False, "need_sales": False, "need_weather": False}
    if event_name == "user_send_sticker":
        return {"mode": "STICKER", "need_web": False, "need_empathy": True, "need_sales": False, "need_weather": False}
    if any(k in t for k in ["công ty gì", "công ty bạn", "bạn là ai", "ai phát triển", "thuộc công ty nào", "locaith là gì", "locaith ai là gì"]):
        return {"mode": "BRAND", "need_web": False, "need_empathy": False, "need_sales": False, "need_weather": False}
    if any(k in t for k in ["thời tiết", "nhiệt độ", "mưa", "độ ẩm"]):
        return {"mode": "WEATHER", "need_web": False, "need_empathy": False, "need_sales": False, "need_weather": True}
    if any(k in t for k in ["crypto", "coin", "btc", "eth", "wld", "phân tích kỹ thuật", "rsi", "macd", "kháng cự", "hỗ trợ"]):
        return {"mode": "CRYPTO_TA", "need_web": True, "need_empathy": False, "need_sales": False, "need_weather": False}
    if any(k in t for k in ["locaith", "chatbot", "website", "landing", "giải pháp", "triển khai", "báo giá"]):
        return {"mode": "SALES", "need_web": False, "need_empathy": False, "need_sales": True, "need_weather": False}
    need_web = any(k in t for k in ["giá", "hôm nay", "mới nhất", "tin tức", "kết quả"])
    empathy_kw = ["mệt","buồn","lo","chán","khó chịu","áp lực","con mình","gia đình","căng thẳng"]
    need_empathy = any(k in t for k in empathy_kw)
    return {"mode": "GENERAL", "need_web": need_web, "need_empathy": need_empathy, "need_sales": False, "need_weather": False}

def agent_vision_summary(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "Ảnh không đọc được."
    prompt = (
        "Nếu ảnh là biểu đồ tài chính, hãy tóm tắt rất ngắn (khung thời gian nếu nhận ra, xu hướng gần đây, mức hỗ trợ/kháng cự nổi bật) "
        "và lưu ý đây không phải lời khuyên. Nếu là ảnh thường, mô tả súc tích nội dung chính hoặc OCR chữ quan trọng."
    )
    model = genai.GenerativeModel(MODEL_VISION)
    resp = model.generate_content([prompt, image])
    try:
        return resp.text.strip()
    except Exception:
        return "Không trích xuất được nội dung từ ảnh."

def agent_sticker_mood(image_bytes: Optional[bytes]) -> str:
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt = (
                "Đây là ảnh sticker. Dựa vào nét mặt/cử chỉ, hãy suy đoán cảm xúc chính "
                "(vui, buồn, sốc, giận, dỗi, chán, phấn khích, bình thản, yêu thương). Trả đúng một nhãn tiếng Việt."
            )
            model = genai.GenerativeModel(MODEL_VISION)
            resp = model.generate_content([prompt, image])
            return (resp.text or "").strip().lower()[:40]
        except Exception:
            pass
    return "khó đoán"

def agent_web_answer(text: str) -> str:
    return serper_search(text, 3) or ""

def agent_crypto_ta(text: str) -> str:
    return summarize_crypto_from_web(text) or "Mình chưa lấy được dữ liệu kỹ thuật đáng tin. Bạn nhắc mình thử lại nhé."

# ============== BRAND GUARD & SYSTEM NOTE ==============
def brand_guard(text: str) -> str:
    if not text:
        return text
    bad = [
        r"được\s+google\s+phát\s+triển", r"do\s+google\s+xây\s+dựng",
        r"sản\s+phẩm\s+của\s+google", r"của\s+openai", r"của\s+anthropic",
        r"của\s+deepmind", r"của\s+gemini", r"mình\s+thuộc\s+google",
        r"gemini\s+phát\s+triển",
    ]
    out = text
    for pat in bad:
        out = re.sub(pat, BRAND_DEVLINE, out, flags=re.IGNORECASE)
    out = re.sub(r"trí tuệ nhân tạo(.*)google", f"trí tuệ nhân tạo {BRAND_DEVLINE}", out, flags=re.IGNORECASE)
    return out

def constitution_excerpt() -> str:
    if RUNTIME_PROMPT.get("json"):
        j = RUNTIME_PROMPT["json"]
        values = j.get("core_values", {})
        workflow = j.get("workflow", {})
        qa = j.get("qa_validation", {}).get("scoring", {})
        parts = []
        if values:
            parts.append("Giá trị: " + ", ".join(values.keys()))
        if workflow:
            parts.append("Quy trình: " + workflow.get("flowchart", "input -> output"))
        if qa:
            parts.append("QA tổng: " + str(qa.get("max_total", 60)))
        return " / ".join(parts)
    if RUNTIME_PROMPT.get("raw_text"):
        return "Áp dụng Hiến pháp Prompt nội bộ ở chế độ rút gọn."
    return ""

def system_note(profile: Dict[str, Any], salute: Optional[str], avoid_greeting: bool) -> str:
    name = (profile or {}).get("display_name") or "bạn"
    who = salute or name
    base = (
        "Bạn là trợ lý hội thoại tiếng Việt của Locaith AI, nói tự nhiên như người thật, kín đáo và lịch sự. "
        f"Gọi đối phương là '{who}'. Không dùng ký tự lạ/định dạng đậm nghiêng. "
        f"Về nguồn gốc: luôn nêu rõ {BRAND_DEVLINE} Tuyệt đối không nói bạn thuộc hoặc do Google/OpenAI/Anthropic phát triển. "
        "Được phép lập 'kỹ thuật snapshot' cho crypto bằng cách tổng hợp chỉ báo/khối lượng/tin tức công khai; NHƯNG không đưa lời khuyên đầu tư. "
        "Chỉ gợi ý về Locaith khi người dùng hỏi hoặc có tín hiệu rõ ràng. "
        + constitution_excerpt()
    )
    if avoid_greeting:
        base += " Không mở đầu bằng câu chào; đi thẳng vào nội dung."
    return base

def agent_responder(profile: Dict[str, Any], salute: Optional[str], user_text: str,
                    ctx: str, web_ctx: str, vision_ctx: str, mood_ctx: str,
                    tech_ctx: str, weather_ctx: str,
                    mode: str, avoid_greeting: bool) -> str:
    style = ("Phong cách: gần gũi, gãy gọn; không liệt kê khô khan; kết thúc bằng một câu hỏi ngắn.")
    mode_hint = {
        "GENERAL": "Trò chuyện bình thường.",
        "EMPATHY": "Lắng nghe và đồng cảm trước, sau đó hỏi mở.",
        "SALES": "Khám phá nhu cầu, hỏi bối cảnh; không bán hàng khiên cưỡng.",
        "STICKER": "Phản hồi dựa trên cảm xúc ước lượng từ sticker.",
        "VISION": "Giải thích dựa trên thông tin từ ảnh.",
        "BRAND": f"Nếu hỏi về công ty/ai phát triển → giới thiệu ngắn: {BRAND_INTRO}",
        "CRYPTO_TA": "Tổng hợp chỉ báo kỹ thuật và tin liên quan (không lời khuyên).",
        "WEATHER": "Tường thuật thời tiết ngắn gọn, dễ hiểu.",
    }.get(mode, "Trò chuyện tự nhiên.")
    parts = []
    if ctx: parts.append(f"Ngữ cảnh gần đây:\n{ctx}")
    if weather_ctx: parts.append(f"Thời tiết:\n{weather_ctx}")
    if tech_ctx: parts.append(f"Crypto snapshot:\n{tech_ctx}")
    if web_ctx and not tech_ctx: parts.append(f"Thông tin từ internet:\n{web_ctx}")
    if vision_ctx: parts.append(f"Thông tin từ ảnh:\n{vision_ctx}")
    if mood_ctx: parts.append(f"Tâm trạng ước lượng từ sticker: {mood_ctx}")
    bundle = "\n\n".join(parts)

    content = f"{style}\nChế độ: {mode_hint}\n\n{bundle}\n\nTin nhắn của người dùng:\n{user_text}"
    model = genai.GenerativeModel(MODEL_RESPONDER)
    resp = model.generate_content(
        [{"role": "user", "parts": [system_note(profile, salute, avoid_greeting)]},
         {"role": "user", "parts": [content]}],
        generation_config={"temperature": 0.6}
    )
    try:
        out = resp.text.strip()
    except Exception:
        out = "Xin lỗi, mình đang hơi bận. Bạn nhắn lại giúp mình sau một lát nhé."
    return brand_guard(out)

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
    print("Locaith AI – Zalo webhook started v3.4.0")
    if PROMPT_CONFIG_PATH:
        print("Loaded runtime prompt from:", PROMPT_CONFIG_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.4.0", "ts": time.time()}

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

    if is_spamming(user_id):
        zalo_send_text(user_id, escalate_spam(user_id))
        return {"status": "spam"}

    s = ensure_session(user_id)
    if not s["profile"]:
        s["profile"] = zalo_get_profile(user_id)

    # follow
    if event_name == "follow":
        if not s["welcomed"]:
            msg = welcome_line(s["profile"])
            zalo_send_text(user_id, msg)
            s["welcomed"] = True
            push_history(user_id, "assistant", msg)
        return {"status": "ok"}

    if user_id in _ban_until and time.time() < _ban_until[user_id]:
        return {"status": "banned"}

    text = get_text(event)
    salute = parse_salute(text) or s.get("salute")
    if salute and s.get("salute") != salute:
        s["salute"] = salute

    img_bytes = get_image_or_sticker_bytes(event) if event_name in ["user_send_image", "user_send_sticker"] else None
    has_image = bool(img_bytes) and event_name == "user_send_image"

    # acks cho các event không có text
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
    tech_ctx = ""
    weather_ctx = ""

    if event_name == "user_send_sticker":
        mood_ctx = agent_sticker_mood(img_bytes)
        mode = "EMPATHY"

    if has_image:
        vision_ctx = agent_vision_summary(img_bytes)
        mode = "VISION"

    if plan.get("need_weather"):
        weather_ctx = get_weather_snapshot(text)

    if mode == "CRYPTO_TA":
        tech_ctx = agent_crypto_ta(text)
    elif plan.get("need_web") and text:
        web_ctx = agent_web_answer(text)

    # welcome 1 lần, cấm responder lặp chào
    avoid_greeting = False
    justWelcomed = False
    if not s["welcomed"]:
        msg = welcome_line(s["profile"])
        zalo_send_text(user_id, msg)
        push_history(user_id, "assistant", msg)
        s["welcomed"] = True
        avoid_greeting = True
        justWelcomed = True

    if text or vision_ctx or mood_ctx or weather_ctx or tech_ctx:
        push_history(user_id, "user", text or "[non-text]")

        final = agent_responder(
            s["profile"], s.get("salute"), text,
            recent_context(user_id, 8), web_ctx, vision_ctx, mood_ctx,
            tech_ctx, weather_ctx, mode, avoid_greeting
        )

        if justWelcomed and re.fullmatch(r"(chào|xin chào|alo|hi|hello)[!.\s]*", (text or "").lower()):
            final = "Hôm nay bạn muốn mình giúp điều gì?"

        final = brand_guard(final)
        zalo_send_text(user_id, final)
        push_history(user_id, "assistant", final)

    s["notes"].clear()
    s["last_seen"] = time.time()
    return {"status": "ok"}

# Optional KB ingest
@app.post("/kb/url")
def kb_url(user_id: str = Form(...), url: str = Form(...)):
    return {"ok": True, "note": "Placeholder ingest. Có thể mở rộng RAG sau."}
