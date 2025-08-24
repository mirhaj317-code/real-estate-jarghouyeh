   import streamlit as st
import sqlite3
import hashlib
import time
import pandas as pd
import folium
import math
import requests
from streamlit_folium import st_folium
from datetime import datetime
from typing import Optional, Dict, Any, List

# =========================
# ------- THEME -----------
# =========================
def custom_style():
    st.markdown("""
    <style>
      :root{
        --prim:#8B3A3A; /* قهوه‌ای ایرانی */
        --prim-dark:#6f2e2e;
        --gold:#C5A572;
        --cream:#FBF5E6;
        --ink:#2e2e2e;
      }
      html, body, [class*="css"]  {
        font-family: Vazirmatn, Tahoma, sans-serif;
        background: var(--cream);
        color: var(--ink);
      }
      .stApp { background: linear-gradient(180deg, #fffaf1, #f9f1df 180px, #fffaf1); }
      .stButton>button{
        background: var(--prim);
        color: #fff;
        border: 0;
        border-radius: 14px;
        padding: 10px 18px;
        font-weight: 700;
        box-shadow: 0 6px 20px rgba(139,58,58,.25);
      }
      .stButton>button:hover{ background: var(--prim-dark); }
      .stTextInput>div>input, .stNumberInput>div>div>input, textarea, select{
        border:2px solid var(--gold)!important; border-radius:12px!important;
        background:#fffaf6!important;
      }
      .pill{
        display:inline-block;background:#fff;border:1px solid var(--gold);
        padding:6px 10px;border-radius:999px;margin:2px 4px;font-size:12px
      }
      .card{
        background:#fff; border:1px solid #eadfc7; border-radius:18px; padding:16px; margin:10px 0;
        box-shadow: 0 6px 26px rgba(197,165,114,.12);
      }
      header[data-testid="stHeader"] { background: transparent; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# ------- DB --------------
# =========================
DB_NAME = "real_estate.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def migrate_db():
    conn = get_conn()
    c = conn.cursor()

    # users
    c.execute("""
      CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password_hash TEXT,
        role TEXT DEFAULT 'public',
        phone TEXT
      );
    """)
    # properties
    c.execute("""
      CREATE TABLE IF NOT EXISTS properties(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        price INTEGER,
        area INTEGER,
        city TEXT,
        property_type TEXT,
        latitude REAL,
        longitude REAL,
        address TEXT,
        owner_email TEXT,
        description TEXT,
        rooms INTEGER,
        building_age INTEGER,
        facilities TEXT,
        video_url TEXT,
        status TEXT DEFAULT 'draft'
      );
    """)
    # images
    c.execute("""
      CREATE TABLE IF NOT EXISTS images(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        image BLOB,
        FOREIGN KEY(property_id) REFERENCES properties(id)
      );
    """)
    # comments/ratings
    c.execute("""
      CREATE TABLE IF NOT EXISTS comments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        user_email TEXT,
        comment TEXT,
        rating INTEGER,
        created_at TEXT,
        FOREIGN KEY(property_id) REFERENCES properties(id)
      );
    """)
    # favorites
    c.execute("""
      CREATE TABLE IF NOT EXISTS favorites(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        user_email TEXT,
        created_at TEXT
      );
    """)
    # messages (chat)
    c.execute("""
      CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_id INTEGER,
        sender_email TEXT,
        receiver_email TEXT,
        body TEXT,
        created_at TEXT
      );
    """)
    # payments
    c.execute("""
      CREATE TABLE IF NOT EXISTS payments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_temp_json TEXT,  -- پیش‌نویس اطلاعات ملک
        user_email TEXT,
        amount INTEGER,
        authority TEXT,
        ref_id TEXT,
        status TEXT,  -- initiated | paid | failed
        created_at TEXT
      );
    """)
    # مهاجرت‌های ایمن (اضافه کردن ستون‌های جدید اگر قبلاً نبودند)
    safe_alters = [
        ("properties", "address", "TEXT"),
        ("properties", "video_url", "TEXT"),
        ("properties", "status", "TEXT"),
        ("users", "phone", "TEXT"),
    ]
    for table, col, coltype in safe_alters:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

# =========================
# ------- AUTH ------------
# =========================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, password, role="public", phone=None) -> bool:
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("INSERT INTO users(name,email,password_hash,role,phone) VALUES(?,?,?,?,?)",
                  (name, email, hash_password(password), role, phone))
        conn.commit(); conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email, password) -> Optional[Dict[str, Any]]:
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT name, role, phone FROM users WHERE email=?", (email,))
    row = c.fetchone(); conn.close()
    if not row: return None
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE email=?", (email,))
    ph = c.fetchone()[0]; conn.close()
    if ph == hash_password(password):
        return {"email": email, "name": row[0], "role": row[1], "phone": row[2]}
    return None

def reset_password(email, new_password) -> bool:
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE users SET password_hash=? WHERE email=?",
              (hash_password(new_password), email))
    conn.commit(); ok = c.rowcount>0; conn.close()
    return ok

# =========================
# ------- UTIL ------------
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    dLat=math.radians(lat2-lat1)
    dLon=math.radians(lon2-lon1)
    a=math.sin(dLat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def badge(text): 
    st.markdown(f"<span class='pill'>{text}</span>", unsafe_allow_html=True)

# =========================
# ------- PAYMENT (Zarinpal) ----
# =========================
def zp_config():
    # secrets.toml:
    # [zarinpal]
    # merchant_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    # sandbox = true/false
    # [app]
    # base_url = "http://localhost:8501"
    cfg = {
        "merchant_id": st.secrets["zarinpal"]["merchant_id"],
        "sandbox": bool(st.secrets["zarinpal"].get("sandbox", True)),
        "base_url": st.secrets["app"]["base_url"],
    }
    return cfg

def zp_endpoints(sandbox: bool):
    base = "https://sandbox.zarinpal.com/pg/v4/payment" if sandbox else "https://api.zarinpal.com/pg/v4/payment"
    return {
        "request": f"{base}/request.json",
        "verify": f"{base}/verify.json",
        "startpay": "https://sandbox.zarinpal.com/pg/StartPay" if sandbox else "https://www.zarinpal.com/pg/StartPay"
    }

def create_payment_request(amount:int, description:str, email:str, mobile:str, callback_url:str) -> Optional[str]:
    cfg = zp_config(); ep = zp_endpoints(cfg["sandbox"])
    payload = {
        "merchant_id": cfg["merchant_id"],
        "amount": amount,
        "description": description,
        "callback_url": callback_url,
        "metadata": {"email": email or "", "mobile": mobile or ""}
    }
    try:
        r = requests.post(ep["request"], json=payload, timeout=15)
        data = r.json()
        if data.get("data") and data["data"].get("authority"):
            return data["data"]["authority"]
        else:
            st.error(f"خطای ایجاد تراکنش: {data.get('errors') or data}")
    except Exception as e:
        st.error(f"خطای اتصال به درگاه: {e}")
    return None

def verify_payment(amount:int, authority:str) -> Dict[str,Any]:
    cfg = zp_config(); ep = zp_endpoints(cfg["sandbox"])
    payload = {
        "merchant_id": cfg["merchant_id"],
        "amount": amount,
        "authority": authority
    }
    try:
        r = requests.post(ep["verify"], json=payload, timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# ------- PROPERTIES ------
# =========================
def add_property_row(data: Dict[str,Any], images: List[bytes], publish:bool=False) -> int:
    conn=get_conn(); c=conn.cursor()
    c.execute("""INSERT INTO properties
    (title,price,area,city,property_type,latitude,longitude,address,owner_email,description,rooms,building_age,facilities,video_url,status)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        data['title'], data['price'], data['area'], data['city'], data['property_type'],
        data['latitude'], data['longitude'], data.get('address'),
        data['owner_email'], data.get('description'), data.get('rooms'),
        data.get('building_age'), data.get('facilities'), data.get('video_url'),
        'published' if publish else 'draft'
    ))
    pid = c.lastrowid
    for img in images:
        c.execute("INSERT INTO images(property_id,image) VALUES(?,?)", (pid, img))
    conn.commit(); conn.close()
    return pid

def list_properties_df(filters: Dict[str,Any]) -> pd.DataFrame:
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT * FROM properties WHERE status='published'")
    rows = c.fetchall()
    cols=[d[0] for d in c.description]
    conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df

    # فیلترها
    if filters.get("city"):
        df = df[df["city"].isin(filters["city"])]
    if filters.get("property_type"):
        df = df[df["property_type"].isin(filters["property_type"])]

    def between(series, lo, hi):
        if lo is not None: series = series[series >= lo]
        if hi is not None: series = series[series <= hi]
        return series.index

    idx = df.index
    idx = idx.intersection(between(df["price"], filters.get("min_price"), filters.get("max_price")))
    idx = idx.intersection(between(df["area"], filters.get("min_area"), filters.get("max_area")))
    idx = idx.intersection(between(df["rooms"], filters.get("min_rooms"), filters.get("max_rooms")))
    idx = idx.intersection(between(df["building_age"], filters.get("min_age"), filters.get("max_age")))
    df = df.loc[idx]

    # امکانات (contains all)
    if filters.get("facilities"):
        for f in filters["facilities"]:
            df = df[df["facilities"].fillna("").str.contains(f, case=False, na=False)]

    # شعاع جغرافیایی
    if filters.get("center_lat") is not None and filters.get("center_lon") is not None and filters.get("radius_km"):
        center_lat = filters["center_lat"]; center_lon = filters["center_lon"]; R = filters["radius_km"]
        def in_radius(row):
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]): return False
            return haversine_km(center_lat, center_lon, row["latitude"], row["longitude"]) <= R
        df = df[df.apply(in_radius, axis=1)]

    return df

def property_images(prop_id:int) -> List[bytes]:
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT image FROM images WHERE property_id=?", (prop_id,))
    rows=c.fetchall(); conn.close()
    return [r[0] for r in rows]

# =========================
# ------- MAP -------------
# =========================
def show_map(df: pd.DataFrame):
    if df.empty:
        st.info("هیچ ملکی مطابق فیلترها پیدا نشد.")
        return
    m = folium.Map(
        location=[df['latitude'].dropna().mean(), df['longitude'].dropna().mean()],
        zoom_start=12, tiles="CartoDB positron"
    )
    try:
        from folium.plugins import MarkerCluster
        cluster = MarkerCluster().add_to(m)
    except Exception:
        cluster = m

    for _, row in df.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]): 
            continue
        html = f"""
        <div style='font-family:Tahoma;'>
          <b>{row["title"]}</b><br>
          نوع: {row["property_type"]} | شهر: {row["city"]}<br>
          قیمت: {row["price"]:,} تومان | متراژ: {row["area"]} متر<br>
          <small>{(row.get("address") or "")[:80]}</small>
        </div>
        """
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=folium.Popup(html, max_width=300),
            tooltip=row["title"],
            icon=folium.Icon(color="red", icon="home")
        ).add_to(cluster)
    st_folium(m, width=900, height=560)

# =========================
# ------- COMMENTS --------
# =========================
def add_comment(pid:int, user_email:str, comment:str, rating:int):
    conn=get_conn(); c=conn.cursor()
    c.execute("INSERT INTO comments(property_id,user_email,comment,rating,created_at) VALUES(?,?,?,?,?)",
              (pid, user_email, comment, rating, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def load_comments(pid:int) -> pd.DataFrame:
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT user_email, comment, rating, created_at FROM comments WHERE property_id=? ORDER BY id DESC", (pid,))
    rows=c.fetchall(); conn.close()
    return pd.DataFrame(rows, columns=["user_email","comment","rating","created_at"])

# =========================
# ------- FAVORITES -------
# =========================
def toggle_fav(pid:int, user_email:str):
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT id FROM favorites WHERE property_id=? AND user_email=?", (pid,user_email))
    r=c.fetchone()
    if r:
        c.execute("DELETE FROM favorites WHERE id=?", (r[0],))
        conn.commit(); conn.close(); return False
    else:
        c.execute("INSERT INTO favorites(property_id,user_email,created_at) VALUES(?,?,?)",
                  (pid,user_email, datetime.utcnow().isoformat()))
        conn.commit(); conn.close(); return True

def list_favorites(user_email:str) -> pd.DataFrame:
    conn=get_conn(); c=conn.cursor()
    c.execute("""SELECT p.* FROM favorites f
                 JOIN properties p ON p.id=f.property_id
                 WHERE f.user_email=? AND p.status='published'""", (user_email,))
    rows=c.fetchall(); cols=[d[0] for d in c.description]; conn.close()
    return pd.DataFrame(rows, columns=cols)

# =========================
# ------- CHAT ------------
# =========================
def send_message(pid:int, sender:str, receiver:str, body:str):
    conn=get_conn(); c=conn.cursor()
    c.execute("INSERT INTO messages(property_id,sender_email,receiver_email,body,created_at) VALUES(?,?,?,?,?)",
              (pid, sender, receiver, body, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def load_chat(pid:int, a:str, b:str) -> List[Dict[str,Any]]:
    conn=get_conn(); c=conn.cursor()
    c.execute("""SELECT sender_email, body, created_at FROM messages
                 WHERE property_id=? AND (sender_email IN (?,?) AND receiver_email IN (?,?))
                 ORDER BY id ASC""", (pid, a,b,a,b))
    rows=c.fetchall(); conn.close()
    return [{"sender":r[0], "body":r[1], "at":r[2]} for r in rows]

# =========================
# ------- UI PAGES --------
# =========================
def signup_page():
    st.subheader("ثبت‌نام")
    name = st.text_input("نام کامل")
    email = st.text_input("ایمیل")
    phone = st.text_input("شماره تماس (اختیاری)")
    password = st.text_input("رمز عبور", type="password")
    if st.button("ایجاد حساب"):
        if name and email and password:
            ok = register_user(name, email, password, role="public", phone=phone or None)
            if ok: st.success("ثبت‌نام موفق. حالا وارد شوید.")
            else:  st.error("این ایمیل قبلاً ثبت شده.")
        else:
            st.warning("همه فیلدهای ضروری را پر کنید.")

def login_page():
    st.subheader("ورود")
    email = st.text_input("ایمیل")
    password = st.text_input("رمز عبور", type="password")
    colA, colB = st.columns(2)
    if colA.button("ورود"):
        u = login_user(email, password)
        if u:
            st.session_state["user"] = u
            st.success(f"خوش آمدی {u['name']} 🌹")
            st.experimental_rerun()
        else:
            st.error("ایمیل یا رمز عبور اشتباه است.")

    if colB.button("فراموشی رمز عبور"):
        st.session_state["show_reset"] = True

    if st.session_state.get("show_reset"):
        st.info("رمز جدید را تنظیم کنید (فعلاً بدون ایمیل تایید).")
        e = st.text_input("ایمیل ثبت‌شده", key="rp_e")
        npw = st.text_input("رمز جدید", type="password", key="rp_p")
        if st.button("تغییر رمز"):
            if reset_password(e, npw):
                st.success("رمز تغییر کرد. وارد شوید.")
                st.session_state["show_reset"] = False
            else:
                st.error("ایمیل یافت نشد.")

def property_filters():
    st.markdown("### فیلتر دقیق جستجو")
    cities = st.multiselect("شهر", options=sorted(list_all("city")))
    types  = st.multiselect("نوع ملک", options=sorted(list_all("property_type")))
    c1, c2 = st.columns(2)
    min_price = c1.number_input("حداقل قیمت (تومان)", min_value=0, step=1_000_000, value=0)
    max_price = c2.number_input("حداکثر قیمت (تومان)", min_value=0, step=1_000_000, value=0)
    a1, a2 = st.columns(2)
    min_area = a1.number_input("حداقل متراژ", min_value=0, step=1, value=0)
    max_area = a2.number_input("حداکثر متراژ", min_value=0, step=1, value=0)
    r1, r2 = st.columns(2)
    min_rooms = r1.number_input("حداقل اتاق", min_value=0, step=1, value=0)
    max_rooms = r2.number_input("حداکثر اتاق", min_value=0, step=1, value=0)
    g1, g2 = st.columns(2)
    min_age = g1.number_input("حداقل سن بنا", min_value=0, step=1, value=0)
    max_age = g2.number_input("حداکثر سن بنا", min_value=0, step=1, value=0)
    facilities = st.multiselect("امکانات (شامل شود)", ["آسانسور","پارکینگ","انباری","بالکن","استخر","سونا","روف‌گاردن","کمد دیواری"])

    st.markdown("#### جستجو بر اساس شعاع مکانی (اختیاری)")
    d1, d2, d3 = st.columns(3)
    center_lat = d1.number_input("عرض جغرافیایی مرکز", format="%.6f")
    center_lon = d2.number_input("طول جغرافیایی مرکز", format="%.6f")
    radius_km  = d3.number_input("شعاع (کیلومتر)", min_value=0, step=1)

    return {
        "city": cities or None,
        "property_type": types or None,
        "min_price": min_price or None,
        "max_price": max_price or None if max_price>0 else None,
        "min_area": min_area or None,
        "max_area": max_area or None if max_area>0 else None,
        "min_rooms": min_rooms or None,
        "max_rooms": max_rooms or None if max_rooms>0 else None,
        "min_age": min_age or None,
        "max_age": max_age or None if max_age>0 else None,
        "facilities": facilities or None,
        "center_lat": center_lat if center_lat else None,
        "center_lon": center_lon if center_lon else None,
        "radius_km": radius_km if radius_km else None
    }

def list_all(colname:str) -> set:
    conn=get_conn(); c=conn.cursor()
    try:
        c.execute(f"SELECT DISTINCT {colname} FROM properties WHERE status='published' AND {colname} IS NOT NULL")
        vals = {r[0] for r in c.fetchall() if r[0]}
    except Exception:
        vals=set()
    conn.close(); return vals

def property_card(row: pd.Series, user: Optional[Dict[str,Any]]):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"#### {row['title']}")
        badge(row["property_type"]); badge(row["city"])
        st.write(f"**قیمت:** {int(row['price']):,} تومان | **متراژ:** {int(row['area'])} متر | **اتاق:** {int(row['rooms'] or 0)}")
        if row.get("address"):
            st.caption(f"📍 {row['address']}")
        if row.get("description"):
            st.write((row['description'] or "")[:280])

        imgs = property_images(int(row["id"]))
        if imgs:
            st.image(imgs[0], use_column_width=True)

        cols = st.columns(5)
        if user:
            if cols[0].button("علاقه‌مندی ❤️", key=f"fav_{row['id']}"):
                ok = toggle_fav(int(row["id"]), user["email"])
                st.success("به علاقه‌مندی‌ها اضافه/حذف شد.")
        if row.get("video_url"):
            cols[1].link_button("🎥 تور ویدئویی", row["video_url"])
        cols[2].button("نمایش روی نقشه 🗺️", key=f"map_{row['id']}")
        if user and user["email"] != row["owner_email"]:
            if cols[3].button("گفتگو 💬", key=f"chat_{row['id']}"):
                st.session_state["chat_pid"] = int(row["id"])
                st.experimental_rerun()
        if cols[4].button("نظرات ⭐", key=f"rev_{row['id']}"):
            st.session_state["review_pid"] = int(row["id"])
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def public_panel(user: Dict[str,Any]):
    st.subheader(f"خوش آمدی {user['name']} 🌿")
    st.markdown("### ثبت ملک (برای عموم) – هزینه: **۲۰٬۰۰۰ تومان** از طریق زرین‌پال")

    with st.expander("فرم ثبت ملک", expanded=False):
        title = st.text_input("عنوان")
        price = st.number_input("قیمت (تومان)", min_value=0, step=1_000_000)
        area  = st.number_input("متراژ (متر)", min_value=0, step=1)
        city  = st.text_input("شهر")
        ptype = st.selectbox("نوع ملک", ["آپارتمان","ویلایی","مغازه","زمین","دفتر"])
        c1,c2 = st.columns(2)
        lat = c1.number_input("عرض جغرافیایی", format="%.6f")
        lon = c2.number_input("طول جغرافیایی", format="%.6f")
        address = st.text_input("آدرس")
        rooms = st.number_input("تعداد اتاق", min_value=0, step=1)
        age   = st.number_input("سن بنا", min_value=0, step=1)
        facilities = st.text_area("امکانات (با کاما جدا کن)")
        desc  = st.text_area("توضیحات")
        video = st.text_input("لینک تور ویدئویی/۳۶۰ (اختیاری)")
        uploaded = st.file_uploader("تصاویر (حداکثر ۵ عدد)", type=["png","jpg","jpeg"], accept_multiple_files=True)

        PAY_AMOUNT = 20000  # تومان
        if st.button("پرداخت و ثبت نهایی"):
            if not title or price<=0 or not city or not uploaded:
                st.error("موارد ضروری: عنوان، قیمت، شهر، حداقل یک تصویر.")
            else:
                # ایجاد درخواست پرداخت
                try:
                    cfg = zp_config()
                    callback = f"{cfg['base_url']}/?pg=callback"
                    authority = create_payment_request(
                        amount=PAY_AMOUNT,
                        description=f"ثبت آگهی ملک: {title}",
                        email=user["email"],
                        mobile=user.get("phone") or "",
                        callback_url=callback
                    )
                    if authority:
                        # ذخیره پیش‌نویس در payments
                        import json, base64
                        images_b = [f.read() for f in uploaded][:5]
                        draft = {
                            "title": title, "price": int(price), "area": int(area), "city": city, "property_type": ptype,
                            "latitude": float(lat or 0), "longitude": float(lon or 0), "address": address,
                            "owner_email": user["email"], "description": desc, "rooms": int(rooms or 0),
                            "building_age": int(age or 0), "facilities": facilities, "video_url": video
                        }
                        conn=get_conn(); c=conn.cursor()
                        c.execute("INSERT INTO payments(property_temp_json,user_email,amount,authority,ref_id,status,created_at) VALUES(?,?,?,?,?,?,?)",
                                  (json.dumps(draft), user["email"], PAY_AMOUNT, authority, None, "initiated", datetime.utcnow().isoformat()))
                        conn.commit(); conn.close()

                        eps = zp_endpoints(cfg["sandbox"])
                        st.success("در حال انتقال به درگاه…")
                        st.link_button("رفتن به درگاه زرین‌پال", f"{eps['startpay']}/{authority}", type="primary")
                        st.info("پس از پرداخت، به برنامه برگردانده می‌شوید و آگهی منتشر می‌گردد.")
                    else:
                        st.error("عدم موفقیت در ساخت تراکنش.")
                except Exception as e:
                    st.error(f"پیکربندی درگاه ناقص است: {e}")

    st.markdown("---")
    st.markdown("### جستجو و نقشه")
    filt = property_filters()
    df = list_properties_df(filt)
    show_map(df)
    st.markdown("### نتایج")
    for _, row in df.iterrows():
        property_card(row, st.session_state.get("user"))

    # بخش‌های پویا: نظرات / چت
    if st.session_state.get("review_pid"):
        pid = st.session_state["review_pid"]
        st.markdown("---"); st.subheader("نظرات و امتیازها")
        rating = st.slider("امتیاز", 1, 5, 5)
        cm = st.text_area("نظر شما")
        if st.button("ثبت نظر"):
            add_comment(pid, user["email"], cm, rating)
            st.success("ثبت شد.")
            st.session_state["review_pid"] = None
            st.experimental_rerun()
        dfc = load_comments(pid)
        if not dfc.empty:
            st.dataframe(dfc)

    if st.session_state.get("chat_pid"):
        pid = st.session_state["chat_pid"]
        st.markdown("---"); st.subheader("گفتگو")
        # گیرنده را مالک ملک قرار می‌دهیم
        conn=get_conn(); c=conn.cursor()
        c.execute("SELECT owner_email FROM properties WHERE id=?", (pid,))
        owner = (c.fetchone() or [""])[0]; conn.close()
        if owner and owner != user["email"]:
            msgs = load_chat(pid, user["email"], owner)
            for m in msgs:
                st.chat_message("user" if m["sender"]==user["email"] else "assistant").write(m["body"])
            txt = st.chat_input("پیامت را بنویس…")
            if txt:
                send_message(pid, user["email"], owner, txt)
                st.experimental_rerun()
        else:
            st.info("مالک پیدا نشد.")

def agent_panel(user: Dict[str,Any]):
    st.subheader("پنل مشاور")
    st.info("مدیریت آگهی‌های خودت")
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT * FROM properties WHERE owner_email=?", (user["email"],))
    rows=c.fetchall(); cols=[d[0] for d in c.description]; conn.close()
    df=pd.DataFrame(rows, columns=cols)
    if df.empty:
        st.info("هنوز ملکی اضافه نکرده‌ای (کاربران عمومی پس از پرداخت منتشر می‌شوند).")
    else:
        st.dataframe(df[["id","title","price","city","property_type","status"]])
    st.markdown("---")
    st.subheader("علاقه‌مندی‌های کاربران برای املاک تو")
    # (می‌توان توسعه داد: گزارش‌ها، فروش، لیدها …)

def admin_panel(user: Dict[str,Any]):
    st.subheader("پنل مدیر")
    st.markdown("### مدیریت کاربران")
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT name,email,role,phone FROM users ORDER BY id DESC")
    users=c.fetchall()
    for name,email,role,phone in users:
        col1,col2,col3,col4,col5 = st.columns([3,3,2,3,2])
        col1.write(name); col2.write(email); col3.write(role); col4.write(phone or "—")
        if col5.button(f"ارتقا به مشاور", key=f"mk_{email}"):
            c2=get_conn(); cc=c2.cursor()
            cc.execute("UPDATE users SET role='agent' WHERE email=?", (email,))
            c2.commit(); c2.close(); st.success("انجام شد"); st.experimental_rerun()

    st.markdown("---")
    st.markdown("### مدیریت املاک (انتشار/حذف)")
    c = get_conn().cursor()
    c.execute("SELECT id,title,owner_email,status FROM properties ORDER BY id DESC")
    props=c.fetchall()
    for pid,title,owner,status in props:
        col1,col2,col3,col4 = st.columns([3,3,2,3])
        col1.write(f"{pid} | {title}"); col2.write(owner); col3.write(status)
        if status!="published":
            if col4.button("انتشار", key=f"pub_{pid}"):
                cx=get_conn(); cc=cx.cursor()
                cc.execute("UPDATE properties SET status='published' WHERE id=?", (pid,))
                cx.commit(); cx.close(); st.experimental_rerun()
        else:
            if col4.button("حذف", key=f"del_{pid}"):
                cx=get_conn(); cc=cx.cursor()
                cc.execute("DELETE FROM properties WHERE id=?", (pid,))
                cc.execute("DELETE FROM images WHERE property_id=?", (pid,))
                cx.commit(); cx.close(); st.experimental_rerun()

# =========================
# ------- CALLBACK (Payment) ----
# =========================
def handle_payment_callback():
    # زرین‌پال با پارامترهای ?Authority=...&Status=OK برمی‌گردد
    q = st.query_params
    authority = q.get("Authority")
    status = q.get("Status")
    pg = q.get("pg")
    if pg != "callback": 
        return
    st.markdown("## نتیجه پرداخت")

    if not authority:
        st.error("Authority یافت نشد.")
        return

    # پیدا کردن پرداخت آغاز شده
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT id, property_temp_json, user_email, amount FROM payments WHERE authority=? AND status='initiated'", (authority,))
    row=c.fetchone()
    if not row:
        st.error("تراکنش متناظر پیدا نشد یا قبلاً پردازش شده.")
        return
    pay_id, draft_json, user_email, amount = row

    if status != "OK":
        c.execute("UPDATE payments SET status='failed' WHERE id=?", (pay_id,))
        conn.commit(); conn.close()
        st.error("پرداخت لغو/ناموفق شد.")
        return

    # verify
    res = verify_payment(amount=int(amount), authority=authority)
    data = res.get("data") or {}
    code = data.get("code")
    ref_id = data.get("ref_id")
    if code==100 or code==101:
        # موفق: درج ملک و انتشار
        import json
        draft = json.loads(draft_json)
        # درج و انتشار
        # (تصاویر داخل draft نبودند – پرداخت از صفحه‌ی فرم می‌آید؛
        #  برای سادگی، این نسخه تصاویر را در مرحله‌ی پرداخت نگه نمی‌دارد؛ می‌توان مرحله‌ی آپلود بعد از پرداخت گذاشت)
        pid = add_property_row(draft, images=[], publish=True)

        c.execute("UPDATE payments SET status='paid', ref_id=? WHERE id=?", (str(ref_id), pay_id))
        conn.commit(); conn.close()
        st.success(f"پرداخت موفق ✅ کد پیگیری: {ref_id}")
        st.info(f"آگهی شما منتشر شد. شناسه ملک: {pid}")
    else:
        c.execute("UPDATE payments SET status='failed' WHERE id=?", (pay_id,))
        conn.commit(); conn.close()
        st.error(f"عدم تأیید پرداخت: {res}")

# =========================
# ------- MAIN ------------
# =========================
def main():
    custom_style()
    migrate_db()

    st.title("سامانه پیشرفته مدیریت املاک 🏛️✨")

    # هندل کال‌بک پرداخت اگر برگشت از درگاه باشد
    handle_payment_callback()

    user = st.session_state.get("user")
    with st.sidebar:
        st.header("منو")
        if not user:
            page = st.radio("صفحه", ["ورود", "ثبت‌نام"], index=0, horizontal=True)
            if page=="ثبت‌نام":
                signup_page()
            else:
                login_page()
            st.caption("💡 ابتدا وارد شوید تا امکانات کامل را ببینید.")
        else:
            st.write(f"👤 {user['name']} | نقش: {user['role']}")
            page = st.radio("بخش‌ها", ["عمومی", "مشاور", "مدیر", "علاقه‌مندی‌ها"], index=0)
            if st.button("خروج"):
                del st.session_state["user"]
                st.experimental_rerun()

    if user:
        if page=="عمومی":
            public_panel(user)
        elif page=="مشاور":
            if user["role"] in ("agent","admin"):
                agent_panel(user)
            else:
                st.warning("برای دسترسی به پنل مشاور، از ادمین ارتقا بگیرید.")
        elif page=="مدیر":
            if user["role"]=="admin":
                admin_panel(user)
            else:
                st.warning("دسترسی لازم ندارید.")
        elif page=="علاقه‌مندی‌ها":
            st.subheader("لیست علاقه‌مندی‌ها")
            favdf = list_favorites(user["email"])
            if favdf.empty: st.info("هنوز چیزی اضافه نکرده‌ای.")
            else:
                show_map(favdf)
                for _, row in favdf.iterrows():
                    property_card(row, user)

if __name__ == "__main__":
    main()
