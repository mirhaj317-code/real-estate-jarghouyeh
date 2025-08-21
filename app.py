import streamlit as st
import sqlite3
import hashlib
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import os
import base64

# -----------------------------
# تنظیمات عمومی و استایل سنتی
# -----------------------------
st.set_page_config(page_title="سامانه املاک پیشرفته", page_icon="🏛️", layout="wide")

def custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', Tahoma, sans-serif !important;
        direction: rtl;
    }
    .block-container { padding-top: 1rem; }
    .stButton>button {
        background: linear-gradient(135deg, #9c2f2f, #c1542e);
        color: #fff; border: none; border-radius: 14px; padding: 10px 22px;
        font-weight: 700; box-shadow: 0 6px 18px rgba(156,47,47,.25);
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        border: 2px solid #c1542e !important; border-radius: 12px !important;
    }
    .card {
        background: #fff6ef; border: 1px solid #f3d2b9; border-radius: 16px; padding: 16px; margin-bottom: 14px;
        box-shadow: 0 4px 18px rgba(193,84,46,.1);
    }
    .pill { display:inline-block; background:#ffe6d7; color:#9c2f2f; padding:4px 10px; border-radius:999px; margin-left:8px; }
    .title {
        background: linear-gradient(135deg,#9c2f2f,#c1542e); -webkit-background-clip:text; color: transparent;
        font-weight: 800; font-size: 1.8rem; margin-bottom: .8rem;
    }
    </style>
    """, unsafe_allow_html=True)

custom_style()

# -----------------------------
# خواندن تنظیمات
# -----------------------------
MERCHANT_ID = None
APP_BASE_URL = "http://localhost:8501"

try:
    MERCHANT_ID = st.secrets["zarinpal"]["merchant_id"]
except Exception:
    MERCHANT_ID = os.getenv("ZARINPAL_MERCHANT_ID", None)

try:
    APP_BASE_URL = st.secrets["app"]["base_url"]
except Exception:
    APP_BASE_URL = os.getenv("APP_BASE_URL", APP_BASE_URL)

# -----------------------------
# اتصال دیتابیس و ساخت جداول + مهاجرت نرم
# -----------------------------
DB_NAME = "real_estate.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def setup_db():
    conn = get_conn()
    c = conn.cursor()

    # کاربران
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            role TEXT DEFAULT 'user',
            phone TEXT
        )
    """)

    # املاک
    c.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            price INTEGER,
            area INTEGER,
            city TEXT,
            property_type TEXT,
            latitude REAL,
            longitude REAL,
            owner_id INTEGER,
            description TEXT,
            rooms INTEGER,
            building_age INTEGER,
            facilities TEXT,
            is_public INTEGER DEFAULT 1,
            created_at TEXT,
            FOREIGN KEY (owner_id) REFERENCES users(id)
        )
    """)

    # تصاویر
    c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER,
            image BLOB,
            FOREIGN KEY (property_id) REFERENCES properties(id)
        )
    """)

    # پرداخت‌ها
    c.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount INTEGER,
            authority TEXT,
            ref_id TEXT,
            status TEXT,
            created_at TEXT,
            property_temp_json TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # مهاجرت نرم: افزودن ستون‌ها در صورت نبودن
    def add_column(table, col, coltype):
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass

    add_column("users", "phone", "TEXT")
    add_column("properties", "is_public", "INTEGER DEFAULT 1")
    add_column("properties", "created_at", "TEXT")

    conn.commit()
    conn.close()

setup_db()

# -----------------------------
# امنیت پسورد
# -----------------------------
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# -----------------------------
# کاربران: ثبت/ورود/بازنشانی
# -----------------------------
def add_user(name, email, password, role="user", phone=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO users (name, email, password_hash, role, phone) VALUES (?,?,?,?,?)",
              (name, email, hash_password(password), role, phone))
    conn.commit()
    conn.close()

def get_user_by_email(email):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id,name,email,password_hash,role,phone FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row

def login_user(email, password):
    row = get_user_by_email(email)
    if row and row[3] == hash_password(password):
        return {"id": row[0], "name": row[1], "email": row[2], "role": row[4], "phone": row[5]}
    return None

def reset_password(email, new_password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE users SET password_hash=? WHERE email=?", (hash_password(new_password), email))
    conn.commit()
    conn.close()

# -----------------------------
# زرین‌پال: درخواست و تأیید پرداخت
# -----------------------------
ZP_REQ = "https://api.zarinpal.com/pg/v4/payment/request.json"
ZP_VERIFY = "https://api.zarinpal.com/pg/v4/payment/verify.json"
ZP_START = "https://www.zarinpal.com/pg/StartPay/"

def zarinpal_request(amount: int, description: str, email: str = "", phone: str = "", property_temp_json: str = ""):
    if not MERCHANT_ID:
        st.error("مرچنت زرین‌پال تنظیم نشده است. لطفاً در secrets.toml مقدار merchant_id را قرار دهید.")
        return None

    payload = {
        "merchant_id": MERCHANT_ID,
        "amount": amount,
        "callback_url": APP_BASE_URL,  # به همین صفحه برمی‌گردد
        "description": description,
        "metadata": {"email": email or "", "mobile": phone or ""}
    }
    try:
        r = requests.post(ZP_REQ, json=payload, timeout=15)
        data = r.json()
        if "data" in data and data["data"].get("authority"):
            authority = data["data"]["authority"]
            # ذخیره رکورد پرداخت در وضعیت "INIT"
            conn = get_conn()
            c = conn.cursor()
            user_id = st.session_state["user"]["id"]
            c.execute("""INSERT INTO payments (user_id, amount, authority, ref_id, status, created_at, property_temp_json)
                         VALUES (?,?,?,?,?,?,?)""",
                      (user_id, amount, authority, None, "INIT", datetime.utcnow().isoformat(), property_temp_json))
            conn.commit()
            conn.close()
            return f"{ZP_START}{authority}"
        else:
            st.error("خطا در دریافت لینک پرداخت از زرین‌پال.")
            st.code(data)
            return None
    except Exception as e:
        st.error(f"اشکال در اتصال به زرین‌پال: {e}")
        return None

def zarinpal_verify(authority: str, amount: int):
    if not MERCHANT_ID:
        st.error("مرچنت زرین‌پال تنظیم نشده است.")
        return None

    payload = {"merchant_id": MERCHANT_ID, "amount": amount, "authority": authority}
    try:
        r = requests.post(ZP_VERIFY, json=payload, timeout=15)
        data = r.json()
        return data
    except Exception as e:
        st.error(f"اشکال در تأیید پرداخت: {e}")
        return None

# -----------------------------
# مدیریت املاک
# -----------------------------
def insert_property(prop):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO properties
        (title, price, area, city, property_type, latitude, longitude, owner_id,
         description, rooms, building_age, facilities, is_public, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        prop["title"], prop["price"], prop["area"], prop["city"], prop["property_type"],
        prop["latitude"], prop["longitude"], prop["owner_id"], prop["description"],
        prop["rooms"], prop["building_age"], prop["facilities"], 1, datetime.utcnow().isoformat()
    ))
    prop_id = c.lastrowid
    # تصاویر
    for img_bytes in prop.get("images", []):
        c.execute("INSERT INTO images (property_id, image) VALUES (?, ?)", (prop_id, img_bytes))
    conn.commit()
    conn.close()
    return prop_id

def query_properties(filters):
    conn = get_conn()
    c = conn.cursor()
    clauses = []
    params = []
    if filters.get("city"):
        clauses.append("city IN (%s)" % ",".join(["?"]*len(filters["city"])))
        params.extend(filters["city"])
    if filters.get("property_type"):
        clauses.append("property_type IN (%s)" % ",".join(["?"]*len(filters["property_type"])))
        params.extend(filters["property_type"])
    if filters.get("min_price") is not None:
        clauses.append("price >= ?"); params.append(filters["min_price"])
    if filters.get("max_price") is not None and filters["max_price"] > 0:
        clauses.append("price <= ?"); params.append(filters["max_price"])
    if filters.get("min_area") is not None:
        clauses.append("area >= ?"); params.append(filters["min_area"])
    if filters.get("max_area") is not None and filters["max_area"] > 0:
        clauses.append("area <= ?"); params.append(filters["max_area"])
    if filters.get("rooms") is not None and filters["rooms"] > -1:
        clauses.append("rooms >= ?"); params.append(filters["rooms"])

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    q = f"""SELECT id, title, price, area, city, property_type, latitude, longitude,
                   owner_id, description, rooms, building_age, facilities, created_at
            FROM properties {where} ORDER BY id DESC"""
    c.execute(q, tuple(params))
    rows = c.fetchall()
    conn.close()
    cols = ["id","title","price","area","city","property_type","latitude","longitude",
            "owner_id","description","rooms","building_age","facilities","created_at"]
    return pd.DataFrame(rows, columns=cols)

def get_property_images(pid: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT image FROM images WHERE property_id=?", (pid,))
    imgs = [r[0] for r in c.fetchall()]
    conn.close()
    return imgs

# -----------------------------
# صفحات احراز هویت
# -----------------------------
def signup_page():
    with st.container():
        st.markdown('<div class="title">ثبت‌نام</div>', unsafe_allow_html=True)
        name = st.text_input("نام کامل")
        email = st.text_input("ایمیل")
        phone = st.text_input("شماره تماس")
        password = st.text_input("رمز عبور", type="password")
        role = st.selectbox("نقش", ["user", "agent", "admin"])
        if st.button("ثبت‌نام"):
            if name and email and password:
                try:
                    add_user(name, email, password, role=role, phone=phone)
                    st.success("ثبت‌نام شد. حالا وارد شوید ✅")
                except Exception as e:
                    st.error(f"این ایمیل قبلاً ثبت شده یا خطا: {e}")
            else:
                st.warning("همه فیلدهای ضروری را پر کنید.")

def login_page():
    with st.container():
        st.markdown('<div class="title">ورود</div>', unsafe_allow_html=True)
        email = st.text_input("ایمیل")
        password = st.text_input("رمز عبور", type="password")
        c1, c2 = st.columns(2)
        if c1.button("ورود"):
            user = login_user(email, password)
            if user:
                st.session_state["user"] = user
                st.success(f"خوش آمدی {user['name']} 🌹")
                st.rerun()
            else:
                st.error("ایمیل یا رمز عبور اشتباه است ❌")
        if c2.button("فراموشی رمز عبور"):
            reset_password_page()

def reset_password_page():
    with st.container():
        st.markdown('<div class="title">بازیابی رمز عبور</div>', unsafe_allow_html=True)
        email = st.text_input("ایمیل ثبت‌شده")
        new_pass = st.text_input("رمز جدید", type="password")
        if st.button("تغییر رمز"):
            if email and new_pass:
                reset_password(email, new_pass)
                st.success("رمز عبور تغییر کرد ✅")
            else:
                st.warning("ایمیل و رمز جدید را وارد کنید.")

# -----------------------------
# ثبت ملک عمومی (نیازمند پرداخت واقعی)
# -----------------------------
def public_property_form():
    st.markdown('<div class="title">ثبت ملک (عمومی)</div>', unsafe_allow_html=True)
    with st.form("public_prop_form", clear_on_submit=False):
        title = st.text_input("عنوان ملک")
        price = st.number_input("قیمت (تومان)", min_value=0, step=100000)
        area = st.number_input("متراژ (مترمربع)", min_value=0, step=1)
        city = st.text_input("شهر")
        property_type = st.selectbox("نوع ملک", ["آپارتمان","ویلایی","مغازه","زمین","اداری","سایر"])
        latitude = st.number_input("عرض جغرافیایی (lat)", format="%.6f", value=35.6892)
        longitude = st.number_input("طول جغرافیایی (lon)", format="%.6f", value=51.3890)
        description = st.text_area("توضیحات")
        rooms = st.number_input("تعداد اتاق", min_value=0, step=1)
        building_age = st.number_input("سن بنا (سال)", min_value=0, step=1)
        facilities = st.text_area("امکانات (با , جدا کنید)")
        images = st.file_uploader("تصاویر (تا ۵ فایل)", accept_multiple_files=True, type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("پرداخت ۲۰,۰۰۰ تومان و ثبت")
    if submitted:
        # آماده‌سازی داده‌های موقت ملک
        imgs_bytes = []
        if images:
            if len(images) > 5:
                st.error("حداکثر ۵ تصویر مجاز است.")
                return
            for f in images:
                imgs_bytes.append(f.read())

        prop_temp = {
            "title": title, "price": int(price), "area": int(area), "city": city,
            "property_type": property_type, "latitude": float(latitude), "longitude": float(longitude),
            "owner_id": st.session_state["user"]["id"], "description": description, "rooms": int(rooms),
            "building_age": int(building_age), "facilities": facilities, "images": imgs_bytes
        }

        # اعتبارسنجی اولیه
        if not title or price <= 0 or not city:
            st.error("عنوان، قیمت و شهر الزامی هستند.")
            return

        # درخواست پرداخت واقعی
        pay_url = zarinpal_request(
            amount=20000,
            description=f"ثبت ملک عمومی توسط {st.session_state['user']['email']}",
            email=st.session_state["user"]["email"],
            phone=st.session_state["user"]["phone"] or "",
            property_temp_json=base64.b64encode(str(prop_temp).encode()).decode()
        )
        if pay_url:
            st.success("لینک پرداخت آماده شد. برای تکمیل روی لینک زیر بزنید:")
            st.markdown(f"[پرداخت امن از طریق زرین‌پال]({pay_url})")
            st.info("پس از پرداخت، به همین صفحه بازگردانده می‌شوید و ملک ثبت خواهد شد.")

# -----------------------------
# رسیدگی به بازگشت از زرین‌پال (Callback)
# -----------------------------
def handle_payment_callback():
    # زرین‌پال معمولاً ?Authority=XXXX&Status=OK برمی‌گرداند
    qs = st.query_params
    if not qs:
        return
    authority = qs.get("Authority")
    status = qs.get("Status")
    if not authority:
        return
    st.info("در حال بررسی وضعیت پرداخت…")

    # پرداخت را پیدا کنیم
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, user_id, amount, authority, status, property_temp_json FROM payments WHERE authority=?", (authority,))
    row = c.fetchone()

    if not row:
        conn.close()
        st.error("تراکنش پیدا نشد.")
        return

    pay_id, user_id, amount, authority_db, status_db, prop_temp_b64 = row

    if status != "OK":
        c.execute("UPDATE payments SET status=? WHERE id=?", ("CANCELLED", pay_id))
        conn.commit()
        conn.close()
        st.error("پرداخت توسط کاربر لغو شد.")
        # پاک کردن پارامترها از URL
        st.query_params.clear()
        return

    # اگر قبلاً موفق شده بود
    if status_db == "SUCCESS":
        conn.close()
        st.success("این پرداخت قبلاً تایید و ثبت شده است.")
        st.query_params.clear()
        return

    # Verify
    verify_data = zarinpal_verify(authority_db, amount)
    if not verify_data:
        conn.close()
        st.error("عدم امکان تأیید پرداخت.")
        st.query_params.clear()
        return

    # کد موفقیت در زرین‌پال v4 معمولاً 100 یا 101
    code = verify_data.get("data", {}).get("code")
    ref_id = verify_data.get("data", {}).get("ref_id")

    if code in (100, 101):
        # ذخیره ref_id و SUCCESS
        c.execute("UPDATE payments SET status=?, ref_id=? WHERE id=?", ("SUCCESS", str(ref_id), pay_id))
        conn.commit()

        # ملک را از property_temp_json بسازیم
        try:
            # decode prop temp
            prop_str = base64.b64decode(prop_temp_b64.encode()).decode()
            # امن‌تر: eval نکنیم، چون str(dict) ذخیره کردیم؛ ساده‌ترین روش پارس،
            # به خاطر محدودیت، اینجا یک پارس ساده انجام می‌دهیم:
            # بهتر: از json.dumps / json.loads استفاده شود. (در این نسخه، استرینگ ساده‌ست)
            # برای اطمینان، می‌تونیم از literal_eval استفاده کنیم:
            import ast
            prop_temp = ast.literal_eval(prop_str)

            pid = insert_property(prop_temp)
            st.success(f"پرداخت تایید شد ✅ کد رهگیری: {ref_id}\n\nملک با شناسه #{pid} ثبت شد.")
        except Exception as e:
            st.error(f"پرداخت موفق بود ولی ثبت ملک دچار خطا شد: {e}")
        finally:
            conn.close()
            st.query_params.clear()
    else:
        c.execute("UPDATE payments SET status=? WHERE id=?", ("FAILED", pay_id))
        conn.commit()
        conn.close()
        st.error("پرداخت ناموفق بود یا تایید نشد.")
        st.query_params.clear()

# -----------------------------
# نمایش و فیلتر املاک + نقشه
# -----------------------------
def properties_explorer():
    st.markdown('<div class="title">جستجوی پیشرفته املاک</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        min_price = col1.number_input("حداقل قیمت", min_value=0, step=100000, value=0)
        max_price = col2.number_input("حداکثر قیمت", min_value=0, step=100000, value=0)
        min_area = col3.number_input("حداقل متراژ", min_value=0, step=1, value=0)
        max_area = col4.number_input("حداکثر متراژ", min_value=0, step=1, value=0)
        rooms = col5.number_input("حداقل اتاق", min_value=0, step=1, value=0)
        type_sel = col6.multiselect("نوع ملک", ["آپارتمان","ویلایی","مغازه","زمین","اداری","سایر"])
        city_sel = st.multiselect("شهر", options=get_all_cities())

    filters = {
        "min_price": min_price if min_price>0 else None,
        "max_price": max_price if max_price>0 else None,
        "min_area": min_area if min_area>0 else None,
        "max_area": max_area if max_area>0 else None,
        "rooms": rooms if rooms>0 else None,
        "property_type": type_sel if type_sel else None,
        "city": city_sel if city_sel else None
    }
    df = query_properties(filters)
    st.caption(f"تعداد نتایج: {len(df)}")
    if len(df):
        # لیست کارت‌ها
        for _, r in df.iterrows():
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**{r['title']}**  <span class='pill'>{r['city']}</span>  <span class='pill'>{r['property_type']}</span>", unsafe_allow_html=True)
                st.write(f"قیمت: {r['price']:,} تومان | متراژ: {r['area']} متر | اتاق: {r['rooms']} | سن بنا: {r['building_age'] or 0} سال")
                if r['facilities']:
                    st.write(f"امکانات: {r['facilities']}")
                st.write(r['description'] or "")
                # تصاویر
                imgs = get_property_images(int(r['id']))
                if imgs:
                    thumbs = []
                    for b in imgs[:3]:
                        b64 = base64.b64encode(b).decode()
                        thumbs.append(f"<img src='data:image/jpeg;base64,{b64}' style='width:120px;height:80px;object-fit:cover;border-radius:10px;margin:4px;border:1px solid #f3d2b9;'/>")
                    st.markdown("".join(thumbs), unsafe_allow_html=True)

                # نقشه کوچک
                try:
                    m = folium.Map(location=[r['latitude'], r['longitude']], zoom_start=14)
                    folium.Marker([r['latitude'], r['longitude']], popup=r['title']).add_to(m)
                    st_folium(m, width=700, height=300)
                except:
                    pass
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("نتیجه‌ای مطابق فیلترها یافت نشد.")

def get_all_cities():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT DISTINCT city FROM properties WHERE city IS NOT NULL AND city<>'' ORDER BY city")
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return rows

# -----------------------------
# داشبوردها
# -----------------------------
def dashboard_user():
    st.markdown('<div class="title">داشبورد کاربر</div>', unsafe_allow_html=True)
    tabs = st.tabs(["افزودن ملک عمومی (پرداخت)", "جستجو و نقشه"])
    with tabs[0]:
        public_property_form()
    with tabs[1]:
        properties_explorer()

def dashboard_agent():
    st.markdown('<div class="title">پنل مشاور</div>', unsafe_allow_html=True)
    st.info("مشاهده و مدیریت املاک متعلق به شما:")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id,title,city,price FROM properties WHERE owner_id=? ORDER BY id DESC",
              (st.session_state["user"]["id"],))
    rows = c.fetchall()
    conn.close()
    if rows:
        df = pd.DataFrame(rows, columns=["id","عنوان","شهر","قیمت"])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("هنوز ملکی ثبت نکرده‌اید.")
    st.markdown("---")
    properties_explorer()

def dashboard_admin():
    st.markdown('<div class="title">پنل مدیریت</div>', unsafe_allow_html=True)
    st.info("لیست کاربران")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id,name,email,role,phone FROM users ORDER BY id DESC")
    users = c.fetchall()
    dfu = pd.DataFrame(users, columns=["id","نام","ایمیل","نقش","تلفن"])
    st.dataframe(dfu, use_container_width=True)
    st.markdown("---")
    st.info("لیست املاک")
    c.execute("SELECT id,title,city,price,owner_id FROM properties ORDER BY id DESC")
    props = c.fetchall()
    dfp = pd.DataFrame(props, columns=["id","عنوان","شهر","قیمت","شناسه مالک"])
    st.dataframe(dfp, use_container_width=True)
    conn.close()
    st.markdown("---")
    properties_explorer()

# -----------------------------
# ناوبری و ورود/خروج
# -----------------------------
def sidebar_menu():
    st.sidebar.markdown("### سامانه املاک 🏛️")
    if MERCHANT_ID:
        st.sidebar.success("زرین‌پال: تنظیم شده ✅")
    else:
        st.sidebar.error("مرچنت زرین‌پال تنظیم نشده ❌")
    st.sidebar.caption("برای تست لوکال، base_url را در secrets تنظیم کن.")

    if "user" not in st.session_state:
        page = st.sidebar.radio("صفحه", ["ورود", "ثبت‌نام", "بازیابی رمز"])
        if page == "ثبت‌نام":
            signup_page()
        elif page == "بازیابی رمز":
            reset_password_page()
        else:
            login_page()
    else:
        u = st.session_state["user"]
        st.sidebar.markdown(f"**{u['name']}** ({u['role']})")
        if st.sidebar.button("خروج"):
            del st.session_state["user"]
            st.rerun()

# -----------------------------
# اجرای برنامه
# -----------------------------
def main():
    # اگر برگشت از درگاه باشد، قبل از بقیه صفحات بررسی شود
    handle_payment_callback()

    if "user" not in st.session_state:
        sidebar_menu()
        st.markdown('<div class="title">به سامانه املاک پیشرفته خوش آمدید</div>', unsafe_allow_html=True)
        st.info("برای استفاده، ابتدا وارد شوید یا ثبت‌نام کنید.")
        properties_explorer()
    else:
        sidebar_menu()
        role = st.session_state["user"]["role"]
        if role == "admin":
            dashboard_admin()
        elif role == "agent":
            dashboard_agent()
        else:
            dashboard_user()

if __name__ == "__main__":
    main()
    
   

   
       
  


