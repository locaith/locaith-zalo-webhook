# main.py
# Locaith AI – Zalo OA Chatbot (v3.7.0 no-refresh)
# - Không dùng refresh token/cached token
# - Luôn gửi appsecret_proof khi gọi Zalo
# - Gửi token qua query string; dùng user_id_by_app; chia nhỏ tin nhắn dài
# - Multi-agent: weather (OpenWeather + Serper), web search (Serper), vision/OCR (Gemini), sticker mood
# - Chống lặp chào, chống spam, brand-guard
# - /health, /zalo/webhook (GET verify + POST events)

import os, re, io, json, time, hmac, hashlib, random
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# ===================== ENV & CONFIG =====================
load_dotenv()

ZALO_OA_TOKEN       = os.getenv("ZALO_OA_TOKEN", "").strip()
ZALO_APP_SECRET     = os.getenv("ZALO_APP_SECRET", "").strip()
ZALO_VERIFY_FILE    = os.getenv("ZALO_VERIFY_FILE", "").strip()

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
SERPER_API_KEY      = os.getenv("SERPER_API_KEY", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

PROMPT_CONFIG_PATH  = os.getenv("PROMPT_CONFIG_PATH", "").strip()

EMOJI_ENABLED       = os.getenv("EMOJI_ENABLED", "true").lower() == "true"
MAX_MSG_PER_30S     = int(os.getenv("MAX_MSG_PER_30S", "6"))
BAN_DURATION_SEC    = int(os.getenv("BAN_DURATION_SEC", str(24*3600)))
HISTORY_TURNS       = int(os.getenv("HISTORY_TURNS", "12"))
ZALO_CHUNK_LIMIT    = int(os.getenv("ZALO_CHUNK_LIMIT", "900"))   # an toàn < 1000 ký tự
ZALO_CHUNK_PAUSE    = float(os.getenv("ZALO_CHUNK_PAUSE", "0.25"))

# Brand guard
BRAND_NAME     = "Locaith AI"
BRAND_DEVLINE  = "được đội ngũ founder của Locaith phát triển."
BRAND_OFFERING = "các giải pháp Chatbot AI và Website (website hoàn chỉnh hoặc landing page)."
BRAND_INTRO    = f"{BRAND_NAME} là một startup Việt, {BRAND_DEVLINE} Chúng mình cung cấp {BRAND_OFFERING}"

# Gemini
if not GEMINI_API_KEY:
    raise RuntimeError("Thiếu GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_RESPONDER = "gemini-2.5-flash"
MODEL_VISION    = "gemini-2.5-pro"

# Thông báo nếu thiếu Zalo token/secret
if not ZALO_OA_TOKEN:
    print("WARNING: ZALO_OA_TOKEN trống — sẽ không gửi được tin nhắn.")
if not ZALO_APP_SECRET:
    print("WARNING: ZALO_APP_SECRET trống — không tính được appsecret_proof; Zalo có thể từ chối.")

# ===================== APP =====================
app = FastAPI(title="Locaith AI – Zalo OA", version="3.7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ===================== GLOBAL STATE =====================
_rate: Dict[str, List[float]] = {}
_warn: Dict[str, int] = {}
_ban_until: Dict[str, float] = {}
_processed: Dict[str, float] = {}
_session: Dict[str, Dict[str, Any]] = {}

def ensure_session(uid: str) -> Dict[str, Any]:
    return _session.setdefault(uid, {"welcomed": False, "profile": None, "salute": None, "history": [], "last_seen": time.time()})

def push_history(uid: str, role: str, text: str):
    s = ensure_session(uid)
    s["history"].append({"role": role, "text": text, "ts": time.time()})
    if len(s["history"]) > HISTORY_TURNS:
        s["history"] = s["history"][-HISTORY_TURNS:]

def recent_context(uid: str, k: int = 8) -> str:
    s = ensure_session(uid)
    return "\n".join(("USER: " if h["role"]=="user" else "ASSISTANT: ")+h["text"] for h in s["history"][-k:])

# ===================== UTILS =====================
def emoji(s: str) -> str: return s if EMOJI_ENABLED else ""

def is_spamming(uid: str) -> bool:
    now = time.time()
    if uid in _ban_until and now < _ban_until[uid]: return True
    bucket = _rate.setdefault(uid, [])
    bucket.append(now)
    _rate[uid] = [t for t in bucket if now - t <= 30]
    return len(_rate[uid]) > MAX_MSG_PER_30S

def escalate_spam(uid: str) -> str:
    c = _warn.get(uid, 0) + 1; _warn[uid] = c
    if c == 1: return "Tin nhắn hơi dày, mình xin phép giảm nhịp một chút nhé. Nếu lặp lại mình sẽ tạm khóa 24 giờ."
    _ban_until[uid] = time.time() + BAN_DURATION_SEC
    return "Bạn đã bị tạm khóa tương tác 24 giờ do gửi quá nhiều tin trong thời gian ngắn."

def extract_event_id(evt: dict) -> str:
    mid = (evt.get("message") or {}).get("msg_id")
    if mid: return f"msg_{mid}"
    ts = str(evt.get("timestamp","")); ev = evt.get("event_name","")
    txt = ((evt.get("message") or {}).get("text") or "")
    hx = hashlib.sha256((ts+ev+txt).encode("utf-8")).hexdigest()[:16]
    return f"{ts}_{ev}_{hx}"

def already_processed(eid: str) -> bool:
    if not eid: return False
    if eid in _processed: return True
    _processed[eid] = time.time()
    if len(_processed) > 800:
        for k,_ in sorted(_processed.items(), key=lambda x:x[1])[:200]: _processed.pop(k,None)
    return False

def strip_accents(s: str) -> str:
    import unicodedata as ud
    nfkd = ud.normalize("NFKD", s or "")
    return "".join([c for c in nfkd if not ud.combining(c)])

# ===================== PROMPT RUNTIME =====================
def load_runtime_prompt() -> Dict[str, Any]:
    cfg = {"raw_text": "", "json": None}
    if not PROMPT_CONFIG_PATH: return cfg
    try:
        data = open(PROMPT_CONFIG_PATH, "r", encoding="utf-8").read()
        try: cfg["json"] = json.loads(data)
        except Exception: cfg["raw_text"] = data
    except Exception as e:
        print("Cannot load prompt file:", e)
    return cfg
RUNTIME_PROMPT = load_runtime_prompt()

def constitution_excerpt() -> str:
    if RUNTIME_PROMPT.get("json"):
        j = RUNTIME_PROMPT["json"]
        values = j.get("core_values", {})
        workflow = j.get("workflow", {})
        qa = j.get("qa_validation", {}).get("scoring", {})
        parts = []
        if values: parts.append("Giá trị: " + ", ".join(values.keys()))
        if workflow: parts.append("Quy trình: " + workflow.get("flowchart","input -> output"))
        if qa: parts.append("QA tổng: " + str(qa.get("max_total",60)))
        return " / ".join(parts)
    if RUNTIME_PROMPT.get("raw_text"): return "Áp dụng Hiến pháp Prompt nội bộ ở chế độ rút gọn."
    return ""

# ===================== ZALO HELPERS =====================
def appsecret_proof(access_token: str, app_secret: str) -> str:
    return hmac.new(app_secret.encode("utf-8"), access_token.encode("utf-8"), hashlib.sha256).hexdigest()

def zalo_params() -> Dict[str, str]:
    p = {"access_token": ZALO_OA_TOKEN}
    if ZALO_APP_SECRET:
        p["appsecret_proof"] = appsecret_proof(ZALO_OA_TOKEN, ZALO_APP_SECRET)
    return p

def _smart_split(text: str, limit: int) -> List[str]:
    s = (text or "").strip()
    if len(s) <= limit: return [s]
    parts, buf = [], ""
    paras = re.split(r"\n{2,}", s)
    for p in paras:
        p = p.strip()
        if not p: continue
        if len(p) > limit:
            sents = re.split(r"(?<=[\.\!\?\…;:])\s+", p)
            for sent in sents:
                sent = sent.strip()
                if not sent: continue
                if len(sent) > limit:
                    for i in range(0, len(sent), limit):
                        chunk = sent[i:i+limit]
                        if len(buf) + len(chunk) + 1 > limit:
                            parts.append(buf.strip()); buf = ""
                        buf += ((" " if buf else "") + chunk)
                else:
                    if len(buf) + len(sent) + 1 > limit:
                        parts.append(buf.strip()); buf = ""
                    buf += ((" " if buf else "") + sent)
            if buf: parts.append(buf.strip()); buf = ""
        else:
            if len(buf) + len(p) + 2 > limit:
                parts.append(buf.strip()); buf = ""
            buf += (("\n\n" if buf else "") + p)
    if buf: parts.append(buf.strip())
    return parts

def _zalo_send_text_once(user_id: str, text: str) -> dict:
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    payload = {"recipient": {"user_id": user_id}, "message": {"text": text}}
    try:
        r = requests.post(url, params=zalo_params(), json=payload, timeout=15)
        print("ZALO_SEND_RESULT", r.status_code, r.text)
        return r.json() if r.text else {}
    except Exception as e:
        print("Send error:", e); return {}

def zalo_send_text(user_id: str, text: str) -> None:
    if not text: return
    chunks = _smart_split(text, ZALO_CHUNK_LIMIT)
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        prefix = f"({i}/{total}) " if total > 1 else ""
        _zalo_send_text_once(user_id, prefix + ch)
        if i < total: time.sleep(ZALO_CHUNK_PAUSE)

def zalo_get_profile(user_id: str) -> Dict[str, Any]:
    url = "https://openapi.zalo.me/v2.0/oa/getprofile"
    try:
        r = requests.post(url, params=zalo_params(), json={"user_id": user_id}, timeout=15)
        print("ZALO_GETPROFILE_RESULT", r.status_code, r.text)
        if r.status_code == 200:
            return r.json().get("data", {}) or {}
    except Exception as e:
        print("Get profile error:", e)
    return {}

# ===================== MEDIA HELPERS =====================
def get_text(evt: dict) -> str:
    return ((evt.get("message") or {}).get("text") or "").strip()

def get_image_or_sticker_bytes(evt: dict) -> Optional[bytes]:
    att = (evt.get("message") or {}).get("attachments") or []
    for a in att:
        if a.get("type") in ("image","sticker"):
            url = (a.get("payload") or {}).get("url") or (a.get("payload") or {}).get("href")
            if not url: continue
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and r.content: return r.content
            except Exception: pass
    return None

# ===================== NLP HELPERS =====================
def detect_day(text: str) -> str:
    t = (text or "").lower()
    return "tomorrow" if any(k in t for k in ["ngày mai", "mai", "tomorrow"]) else "today"

def detect_location_candidates(text: str) -> List[str]:
    t = text or ""
    cands = []
    for pat in [r"(?:ở|tại)\s+([^?.,!\n]+)", r"(thời tiết|nhiệt độ|mưa)\s+(?:ở|tại)?\s*([^?.,!\n]+)"]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m: cands.append(m.group(len(m.groups())))
    m = re.search(r"(?:ở|tại)\s+([^?.,!\n]+)$", t, flags=re.IGNORECASE)
    if m: cands.append(m.group(1))
    cands.append(t)
    uniq = []
    for x in cands:
        x = (x or "").strip()
        if x and x not in uniq: uniq.append(x)
    return uniq[:3]

# ===================== SERPER HELPERS =====================
def serper_search(q: str, n: int = 3) -> str:
    if not SERPER_API_KEY: return ""
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": q, "num": n, "gl": "vn", "hl": "vi"},
            timeout=12
        )
        if r.status_code != 200: return ""
        d = r.json()
        lines = []
        ab = d.get("answerBox") or {}
        if ab.get("answer"): lines.append(f"Trả lời nhanh: {ab['answer']}")
        kg = d.get("knowledgeGraph") or {}
        if kg.get("title") and kg.get("type"):
            lines.append(f"[KG] {kg.get('title')} ({kg.get('type')})")
        for it in (d.get("organic") or [])[:n]:
            t = it.get("title",""); s = it.get("snippet",""); u = it.get("link","")
            if t or s: lines.append(f"- {t}. {s} ({u})")
        return "\n".join(lines)
    except Exception as e:
        print("Serper error:", e); return ""

def serper_normalize_place(free_text: str) -> Optional[str]:
    if not SERPER_API_KEY: return None
    query = f"{free_text} city OR thành phố OR tỉnh OR quốc gia"
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5, "gl": "vn", "hl": "vi"},
            timeout=12
        )
        if r.status_code != 200: return None
        d = r.json()
        kg = d.get("knowledgeGraph") or {}
        title = (kg.get("title") or "").strip()
        ktype = (kg.get("type") or "").lower()
        if title and any(k in ktype for k in ["city","thành phố","province","country","đô thị","vietnam"]):
            return title
        for it in (d.get("organic") or []):
            t = (it.get("title") or "").strip()
            if not t: continue
            name = re.split(r"[-–|·]", t)[0].strip()
            if 2 <= len(name) <= 64 and not name.lower().startswith(("thời tiết","weather")):
                return name
    except Exception as e:
        print("Serper norm place error:", e)
    return None

# ===================== WEATHER =====================
def _owm_geocode(city: str, key: str):
    try:
        r = requests.get("https://api.openweathermap.org/geo/1.0/direct",
                         params={"q": city, "limit": 1, "appid": key}, timeout=10)
        if r.status_code != 200 or not r.json(): return None
        it = r.json()[0]
        return {"lat": it["lat"], "lon": it["lon"], "name": it.get("name", city), "country": it.get("country", "")}
    except Exception as e:
        print("OWM geocode error:", e); return None

def _weather_advice(desc: str, tmin: float, tavg: float, tmax: float, wind_max: float) -> str:
    d = (desc or "").lower()
    tips = []
    if any(k in d for k in ["mưa","dông","giông","mưa rào","rain"]): tips.append("mang áo mưa hoặc áo khoác chống nước")
    if tmax is not None and tmax >= 34: tips.append("uống đủ nước, hạn chế ra nắng gắt buổi trưa")
    if tmin is not None and tmin <= 20: tips.append("mặc ấm khi ra ngoài sáng/tối")
    if wind_max is not None and wind_max >= 10: tips.append("để ý gió mạnh nếu đi xe máy/biển")
    return ("Lời khuyên: " + "; ".join(tips)) if tips else ""

def _summarize_tomorrow(forecast_json: dict):
    city = forecast_json.get("city", {})
    tz = city.get("timezone", 0); name = city.get("name", "")
    lst = forecast_json.get("list", []) or []
    if not lst: return None
    now_utc = time.time()
    local_now = now_utc + tz
    local_tomorrow = time.gmtime(local_now + 86400)
    target_ymd = (local_tomorrow.tm_year, local_tomorrow.tm_mon, local_tomorrow.tm_mday)
    points = []
    for it in lst:
        tt = time.gmtime(it["dt"] + tz)
        if (tt.tm_year, tt.tm_mon, tt.tm_mday) == target_ymd:
            main = it.get("main", {}); w = (it.get("weather") or [{}])[0]
            wind = it.get("wind", {}); rain = (it.get("rain") or {}).get("3h", 0.0)
            points.append({"temp": main.get("temp"), "desc": w.get("description",""),
                           "wind": wind.get("speed", 0.0), "rain": rain})
    if not points: return None
    temps = [p["temp"] for p in points if p["temp"] is not None]
    tmin = min(temps) if temps else None
    tmax = max(temps) if temps else None
    tavg = sum(temps)/len(temps) if temps else None
    wind_max = max(p["wind"] for p in points)
    desc = max(set(p["desc"] for p in points), key=lambda d: sum(1 for p in points if p["desc"] == d))
    will_rain = any(p["rain"] and p["rain"] > 0 for p in points)
    advice = _weather_advice(desc, tmin, tavg, tmax, wind_max)
    rain_note = "Có khả năng mưa." if will_rain else "Khả năng mưa thấp."
    line = (f"Dự báo ngày mai ở {name}: {desc}. "
            f"Nhiệt độ trung bình ~{round(tavg)}°C (min {round(tmin)}°C, max {round(tmax)}°C), "
            f"gió tối đa {round(wind_max,1)} m/s. {rain_note}")
    if advice: line += f"\n{advice}"
    return line

def get_weather_snapshot(text: str) -> str:
    key = OPENWEATHER_API_KEY
    if not key: return "Chưa cấu hình OpenWeather API key."
    day = detect_day(text)
    candidates = detect_location_candidates(text)
    norm = None
    for cand in candidates:
        geo = _owm_geocode(cand, key)
        if geo: norm = geo; break
        hint = serper_normalize_place(cand)
        if hint:
            geo = _owm_geocode(hint, key)
            if geo: norm = geo; break
        m = strip_accents(cand)
        if m != cand:
            geo = _owm_geocode(m, key)
            if geo: norm = geo; break
    if not norm: return "Mình chưa xác định được địa danh bạn muốn tra. Bạn nhắn lại tên thành phố rõ hơn giúp mình nhé."
    try:
        if day == "today":
            r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                             params={"lat": norm["lat"], "lon": norm["lon"], "appid": key, "units": "metric", "lang": "vi"},
                             timeout=10)
            if r.status_code != 200: return f"Không lấy được thời tiết hiện tại cho {norm['name']}."
            d = r.json()
            desc = (d.get("weather") or [{}])[0].get("description","")
            temp = d.get("main",{}).get("temp","?"); feels= d.get("main",{}).get("feels_like","?")
            hum  = d.get("main",{}).get("humidity","?"); wind = d.get("wind",{}).get("speed","?")
            return f"Thời tiết hiện tại ở {norm['name']}: {desc}. {temp}°C (cảm giác {feels}°C), ẩm {hum}%, gió {wind} m/s."
        else:
            r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                             params={"lat": norm["lat"], "lon": norm["lon"], "appid": key, "units": "metric", "lang": "vi"},
                             timeout=10)
            if r.status_code != 200: return f"Không lấy được dự báo cho {norm['name']}."
            s = _summarize_tomorrow(r.json())
            return s or f"Mình chưa tổng hợp được dự báo cho {norm['name']}."
    except Exception as e:
        print("OpenWeather error:", e)
        return "Mình không lấy được thông tin thời tiết lúc này."

# ===================== WEB/CRYPTO HELPERS =====================
COIN_SYNONYM = {
    "bitcoin":"BTC","btc":"BTC","ethereum":"ETH","eth":"ETH","binance coin":"BNB","bnb":"BNB",
    "solana":"SOL","sol":"SOL","worldcoin":"WLD","wld":"WLD","ton":"TON","toncoin":"TON",
    "dogecoin":"DOGE","doge":"DOGE","cardano":"ADA","ada":"ADA"
}
def guess_symbol_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b[A-Z]{2,6}\b", text or "")
    if m: return m.group(0)
    low = (text or "").lower()
    for k,v in COIN_SYNONYM.items():
        if k in low: return v
    # fallback nhờ Serper (ticker/coin)
    try:
        q = f"{text} coin symbol OR mã token"
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": q, "num": 5, "gl": "vn", "hl": "vi"}, timeout=12
        )
        if r.status_code == 200:
            d = r.json()
            pat = re.compile(r"\b[A-Z]{2,6}\b")
            kg = d.get("knowledgeGraph") or {}
            for field in [kg.get("title",""), kg.get("description","")]:
                m = pat.search(field or "")
                if m: return m.group(0)
            for it in (d.get("organic") or []):
                for field in [it.get("title",""), it.get("snippet","")]:
                    m = pat.search(field or "")
                    if m: return m.group(0)
    except Exception as e:
        print("Serper guess symbol error:", e)
    return None

def serper_bundle(qs: List[str]) -> str:
    blocks = []
    for q in qs:
        res = serper_search(q, 3)
        if res: blocks.append(f"• {q}\n{res}")
    return "\n\n".join(blocks)

def crypto_snapshot(text: str) -> str:
    sym = guess_symbol_from_text(text)
    if not sym:
        return "Mình chưa nhận diện được mã coin/mã cổ phiếu. Bạn nhắc mình mã/tên cụ thể hơn nhé."
    bundle = serper_bundle([
        f"giá {sym} hôm nay",
        f"{sym} technical analysis RSI MACD today",
        f"{sym} khối lượng giao dịch hôm nay",
        f"{sym} tin tức hôm nay"
    ])
    if not bundle:
        return f"Chưa tổng hợp được dữ liệu đáng tin cho {sym}."
    return f"Snapshot kỹ thuật cho {sym} (tổng hợp công khai, không phải lời khuyên):\n\n{bundle}"

# ===================== PLANNER & RESPONDER =====================
def planner(text: str, has_image: bool, event_name: str) -> Dict[str, Any]:
    t = (text or "").lower()
    if has_image: return {"mode":"VISION","need_web":False,"need_empathy":False,"need_weather":False}
    if event_name == "user_send_sticker": return {"mode":"STICKER","need_web":False,"need_empathy":True,"need_weather":False}
    if any(k in t for k in ["thời tiết","nhiệt độ","mưa","weather","forecast"]):
        return {"mode":"WEATHER","need_web":False,"need_empathy":False,"need_weather":True}
    if any(k in t for k in ["crypto","coin","token","btc","eth","wld","sol","rsi","macd","kháng cự","hỗ trợ","mã cổ phiếu","ticker","chứng khoán"]):
        return {"mode":"CRYPTO_TA","need_web":True,"need_empathy":False,"need_weather":False}
    if any(k in t for k in ["locaith","chatbot","website","landing","giải pháp","triển khai","báo giá","ở đâu","công ty gì","ai phát triển"]):
        return {"mode":"BRAND","need_web":False,"need_empathy":False,"need_weather":False}
    need_web = any(k in t for k in ["giá","hôm nay","mới nhất","tin tức","kết quả"])
    empathy_kw = ["mệt","buồn","lo","chán","khó chịu","áp lực","con mình","gia đình","stress"]
    need_empathy = any(k in t for k in empathy_kw)
    return {"mode":"GENERAL","need_web":need_web,"need_empathy":need_empathy,"need_weather":False}

def agent_vision_summary(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "Ảnh không đọc được."
    prompt = ("Nếu ảnh là biểu đồ tài chính, tóm tắt ngắn (khung thời gian, xu hướng, hỗ trợ/kháng cự nổi bật) "
              "và nhắc đây không phải lời khuyên. Nếu ảnh thường, mô tả súc tích hoặc OCR chữ quan trọng.")
    model = genai.GenerativeModel(MODEL_VISION)
    resp = model.generate_content([prompt, image])
    try: return (resp.text or "").strip()
    except Exception: return "Không trích xuất được nội dung từ ảnh."

def agent_sticker_mood(image_bytes: Optional[bytes]) -> str:
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt = ("Đây là sticker. Dựa vào nét mặt/cử chỉ, đoán cảm xúc chính (vui, buồn, sốc, giận, dỗi, chán, phấn khích, bình thản, yêu thương). "
                      "Chỉ trả đúng một nhãn tiếng Việt.")
            model = genai.GenerativeModel(MODEL_VISION)
            resp = model.generate_content([prompt, image])
            return (resp.text or "").strip().lower()[:40]
        except Exception: pass
    return "khó đoán"

def brand_guard(text: str) -> str:
    if not text: return text
    bad = [r"được\s+google\s+phát\s+triển", r"do\s+google\s+xây\s+dựng", r"sản\s+phẩm\s+của\s+google",
           r"của\s+openai", r"của\s+anthropic", r"của\s+deepmind", r"của\s+gemini", r"mình\s+thuộc\s+google",
           r"gemini\s+phát\s+triển"]
    out = text
    for pat in bad: out = re.sub(pat, BRAND_DEVLINE, out, flags=re.IGNORECASE)
    out = re.sub(r"trí tuệ nhân tạo(.*)google", f"trí tuệ nhân tạo {BRAND_DEVLINE}", out, flags=re.IGNORECASE)
    return out

def system_note(profile: Dict[str, Any], salute: Optional[str], avoid_greeting: bool) -> str:
    name = (profile or {}).get("display_name") or "bạn"
    who = salute or name
    base = ("Bạn là trợ lý hội thoại tiếng Việt của Locaith AI, nói tự nhiên như người thật, kín đáo và lịch sự. "
            f"Gọi đối phương là '{who}'. Không dùng ký tự lạ/định dạng đậm nghiêng. "
            f"Về nguồn gốc: luôn nêu rõ {BRAND_DEVLINE} Tuyệt đối không nói bạn thuộc hoặc do Google/OpenAI/Anthropic phát triển. "
            "Được phép lập 'kỹ thuật snapshot' cho crypto bằng cách tổng hợp chỉ báo/khối lượng/tin tức công khai; KHÔNG đưa lời khuyên đầu tư. "
            "Chỉ gợi ý về Locaith khi người dùng hỏi hoặc có tín hiệu rõ ràng. " + constitution_excerpt())
    if avoid_greeting: base += " Không mở đầu bằng câu chào; đi thẳng vào nội dung."
    return base

def agent_responder(profile: Dict[str, Any], salute: Optional[str], user_text: str,
                    ctx: str, web_ctx: str, vision_ctx: str, mood_ctx: str,
                    tech_ctx: str, weather_ctx: str,
                    mode: str, avoid_greeting: bool) -> str:
    style = "Phong cách: gần gũi, gãy gọn; không liệt kê khô khan; kết thúc bằng một câu hỏi ngắn."
    mode_hint = {
        "GENERAL":"Trò chuyện bình thường.",
        "BRAND":f"Nếu hỏi về công ty/ai phát triển -> giới thiệu ngắn: {BRAND_INTRO}",
        "VISION":"Giải thích dựa trên thông tin từ ảnh.",
        "STICKER":"Phản hồi theo cảm xúc ước lượng từ sticker.",
        "CRYPTO_TA":"Tổng hợp chỉ báo/khối lượng/tin liên quan (không lời khuyên).",
        "WEATHER":"Tường thuật thời tiết ngắn gọn."
    }.get(mode, "Trò chuyện tự nhiên.")
    parts = []
    if ctx: parts.append(f"Ngữ cảnh gần đây:\n{ctx}")
    if weather_ctx: parts.append(f"Thời tiết:\n{weather_ctx}")
    if tech_ctx: parts.append(f"Crypto/Stock:\n{tech_ctx}")
    if web_ctx and not tech_ctx: parts.append(f"Thông tin web:\n{web_ctx}")
    if vision_ctx: parts.append(f"Từ ảnh:\n{vision_ctx}")
    if mood_ctx: parts.append(f"Tâm trạng sticker: {mood_ctx}")
    bundle = "\n\n".join(parts)
    content = f"{style}\nChế độ: {mode_hint}\n\n{bundle}\n\nTin nhắn của người dùng:\n{user_text}"
    model = genai.GenerativeModel(MODEL_RESPONDER)
    resp = model.generate_content(
        [{"role":"user","parts":[system_note(profile, salute, avoid_greeting)]},
         {"role":"user","parts":[content]}],
        generation_config={"temperature":0.6}
    )
    try: out = (resp.text or "").strip()
    except Exception: out = "Xin lỗi, mình đang hơi bận. Bạn nhắn lại giúp mình sau một lát nhé."
    return brand_guard(out)

# ===================== ROUTES =====================
@app.on_event("startup")
async def on_start():
    print("Locaith AI – Zalo webhook started v3.7.0")
    if PROMPT_CONFIG_PATH: print("Loaded runtime prompt from:", PROMPT_CONFIG_PATH)

@app.get("/health")
def health(): return {"status":"ok","version":"3.7.0","ts":time.time()}

@app.get("/zalo/webhook")
def webhook_verify(challenge: str = ""):
    return {"challenge": challenge} if challenge else {"error":"missing challenge"}

def extract_user_id_for_send(event: dict) -> Optional[str]:
    return event.get("user_id_by_app") or (event.get("sender") or {}).get("id")

@app.post("/zalo/webhook")
async def webhook(req: Request):
    event = await req.json()
    print("EVENT:", json.dumps(event, ensure_ascii=False))
    event_id = extract_event_id(event)
    if already_processed(event_id): return {"status":"duplicate_ignored"}

    event_name = event.get("event_name","")
    user_id = extract_user_id_for_send(event)
    if not user_id: return {"status":"no_user"}

    if is_spamming(user_id):
        zalo_send_text(user_id, escalate_spam(user_id)); return {"status":"spam"}

    s = ensure_session(user_id)
    if not s["profile"]:
        s["profile"] = zalo_get_profile(user_id)

    # follow -> chào 1 lần
    if event_name == "follow":
        if not s["welcomed"]:
            name = (s["profile"] or {}).get("display_name") or "bạn"
            msg = f"Chào {name}. Rất vui được trò chuyện cùng bạn." + (" 🙂" if EMOJI_ENABLED else "")
            zalo_send_text(user_id, msg)
            push_history(user_id, "assistant", msg)
            s["welcomed"] = True
        return {"status":"ok"}

    if user_id in _ban_until and time.time() < _ban_until[user_id]:
        return {"status":"banned"}

    text = get_text(event)
    salute = re.search(r"\b(anh|chị|em)\s+[A-Za-zÀ-ỹĐđ][\wÀ-ỹĐđ\s]*", text or "", re.IGNORECASE)
    salute = salute.group(0).strip() if salute else s.get("salute")
    if salute and s.get("salute") != salute: s["salute"] = salute

    has_img = False
    img_bytes = None
    if event_name in ["user_send_image","user_send_sticker"]:
        img_bytes = get_image_or_sticker_bytes(event)
        has_img = bool(img_bytes) and event_name == "user_send_image"

    if event_name in ["user_send_gif","user_send_audio","user_send_video","user_send_file","user_send_location"] and not text:
        short = {"user_send_gif":"Mình đã nhận ảnh động.",
                 "user_send_audio":"Mình đã nhận voice.",
                 "user_send_video":"Mình đã nhận video.",
                 "user_send_file":"Mình đã nhận file.",
                 "user_send_location":"Mình đã nhận vị trí."}[event_name]
        zalo_send_text(user_id, short); push_history(user_id, "assistant", short); return {"status":"ok"}

    # Không gửi thêm lời chào riêng ở lần đầu có text để tránh lặp
    if not s["welcomed"] and (text or img_bytes):
        s["welcomed"] = True
        avoid_greeting = True
    else:
        avoid_greeting = False

    # Plan
    plan = planner(text, has_img, event_name); mode = plan["mode"]
    web_ctx = ""; vision_ctx = ""; mood_ctx = ""; tech_ctx = ""; weather_ctx = ""

    if event_name == "user_send_sticker":
        mood_ctx = agent_sticker_mood(img_bytes); mode = "STICKER"
    if has_img:
        vision_ctx = agent_vision_summary(img_bytes); mode = "VISION"
    if plan.get("need_weather"):
        weather_ctx = get_weather_snapshot(text)
    if mode == "CRYPTO_TA":
        tech_ctx = crypto_snapshot(text)
    elif plan.get("need_web") and text:
        web_ctx = serper_search(text, 3)

    if text or vision_ctx or mood_ctx or weather_ctx or tech_ctx:
        push_history(user_id, "user", text or "[non-text]")
        final = agent_responder(
            s["profile"], s.get("salute"), text,
            recent_context(user_id, 8), web_ctx, vision_ctx, mood_ctx,
            tech_ctx, weather_ctx, mode, avoid_greeting
        )
        # Nếu người dùng chỉ gõ "chào/hi/hello" ở tin đầu tiên -> trả lời hỏi han ngắn gọn
        if avoid_greeting and re.fullmatch(r"(chào|xin chào|alo|hi|hello)[!.\s]*", (text or "").lower()):
            final = "Mình có thể giúp gì cho bạn hôm nay?"
        final = brand_guard(final)
        zalo_send_text(user_id, final)
        push_history(user_id, "assistant", final)

    s["last_seen"] = time.time()
    return {"status":"ok"}

# Optional: lộ trình ingest URL (placeholder)
@app.post("/kb/url")
def kb_url(user_id: str = Form(...), url: str = Form(...)):
    return {"ok": True, "note": "Placeholder ingest. Có thể mở rộng RAG sau."}
