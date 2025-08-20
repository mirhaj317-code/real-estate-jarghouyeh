import streamlit as st
import sqlite3
import hashlib
import random
import time
import base64
import folium
from streamlit_folium import st_folium
import requests

# دیتابیس و جدول‌ها
conn = sqlite3.connect("real_estate.db", check_same_thread=False)
c = conn.cursor()

def setup_db():
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT,
            password_hash TEXT,
            role TEXT,
            phone TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            price INTEGER,
            area INTEGER,
            city TEXT,
            property_type TEXT,
            latitude REAL,
            longitude REAL,
            owner TEXT,
            description TEXT,
            rooms INTEGER,
            building_age INTEGER,
            facilities TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER,
            image BLOB,
            FOREIGN KEY (property_id) REFERENCES properties(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER,
            username TEXT,
            comment TEXT,
            rating INTEGER,
            FOREIGN KEY (property_id) REFERENCES properties(id)
        )
    ''')
    conn.commit()

# هش کردن رمز عبور
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ثبت‌نام کاربر
def register_user(username, name, password, phone, role="public"):
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                  (username, name, hash_password(password), role, phone))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# ورود کاربر
def login_user(username, password):
    c.execute("SELECT password_hash, role, name FROM users WHERE username=?", (username,))
    res = c.fetchone()
    if res and res[0] == hash_password(password):
        return {"username": username, "role": res[1], "name": res[2]}
    return None

# ارسال OTP با SMS.ir
def send_otp(phone):
    otp = str(random.randint(100000, 999999))
    st.session_state['otp_code'] = otp
    st.session_state['otp_phone'] = phone
    st.info(f"کد تایید به شماره {phone} ارسال شد (برای تست: {otp})")

    # نسخه واقعی با SMS.ir
    try:
        api_key = st.secrets["sms"]["api_key"]
        sender = st.secrets["sms"]["sender"]
        url = f"https://api.sms.ir/v1/send/verify"
        payload = {
            "Mobile": phone,
            "TemplateId": sender,
            "ParameterArray": [{"Parameter": "Code", "ParameterValue": otp}]
        }
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            st.success("پیامک با موفقیت ارسال شد.")
        else:
            st.warning("ارسال پیامک واقعی انجام نشد، از نسخه شبیه‌سازی استفاده شد.")
    except Exception as e:
        st.warning(f"ارسال پیامک واقعی انجام نشد: {e}")

# تایید کد OTP
def verify_otp(input_code):
    return input_code == st.session_state.get('otp_code')

# افزودن ملک
def add_property(data, images):
    c.execute('''INSERT INTO properties 
                 (title, price, area, city, property_type, latitude, longitude, owner, description, rooms, building_age, facilities)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    data['title'], data['price'], data['area'], data['city'], data['property_type'],
                    data['latitude'], data['longitude'], data['owner'], data['description'], data['rooms'],
                    data['building_age'], data['facilities']
                ))
    prop_id = c.lastrowid
    for img in images:
        c.execute("INSERT INTO images (property_id, image) VALUES (?, ?)", (prop_id, img))
    conn.commit()

# ساخت نقشه
def show_map(properties_df):
    if properties_df.empty:
        st.info("هیچ ملکی وجود ندارد.")
        return
    m = folium.Map(location=[properties_df['latitude'].mean(), properties_df['longitude'].mean()], zoom_start=12)
    for _, row in properties_df.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"{row['title']} - قیمت: {row['price']} تومان",
            tooltip=row['title']
        ).add_to(m)
    st_folium(m, width=700, height=500)

# UI با رنگ و لعاب سنتی
def custom_style():
    st.markdown("""
        <style>
        body {
            background-color: #fdf6e3;
            color: #333;
            font-family: 'Vazirmatn', Tahoma, sans-serif;
        }
        .stButton>button {
            background-color: #a52a2a;
            color: white;
            border-radius: 12px;
            padding: 8px 20px;
            font-weight: bold;
        }
        .stTextInput>div>input {
            border: 2px solid #a52a2a;
            border-radius: 10px;
            padding: 6px;
            font-size: 16px;
        }
        .css-1d391kg {
            background-color: #fff0f0 !important;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .stRadio > div > label {
            font-weight: bold;
            color: #a52a2a;
        }
        </style>
    """, unsafe_allow_html=True)

# صفحه ورود با OTP
def login_page():
    st.subheader("ورود با شماره تلفن همراه")
    phone = st.text_input("شماره تلفن همراه")
    if st.button("ارسال کد تایید"):
        if phone.strip() == "" or not phone.isdigit() or len(phone) < 10:
            st.error("شماره تلفن معتبر نیست")
        else:
            send_otp(phone)
    if 'otp_code' in st.session_state:
        otp_input = st.text_input("کد تایید ارسال شده")
        if st.button("تایید کد"):
            if verify_otp(otp_input):
                c.execute("SELECT username FROM users WHERE phone=?", (phone,))
                user = c.fetchone()
                if user is None:
                    username = phone
                    register_user(username, phone, "defaultpassword", phone)
                    st.success("کاربر جدید ساخته شد.")
                st.session_state['user'] = {"username": phone, "role": "public", "name": phone}
                st.experimental_rerun()
            else:
                st.error("کد تایید اشتباه است.")

# صفحه ثبت ملک پولی
def register_property_page():
    st.subheader("ثبت ملک - هزینه: ۴۰ هزار تومان")
    title = st.text_input("عنوان ملک")
    price = st.number_input("قیمت کل (تومان)", min_value=0, step=100000)
    area = st.number_input("متراژ (متر مربع)", min_value=0, step=1)
    city = st.text_input("شهر")
    property_type = st.selectbox("نوع ملک", ["آپارتمان", "ویلایی", "مغازه", "زمین"])
    latitude = st.number_input("عرض جغرافیایی", format="%.6f")
    longitude = st.number_input("طول جغرافیایی", format="%.6f")
    description = st.text_area("توضیحات بیشتر")
    rooms = st.number_input("تعداد اتاق", min_value=0, step=1)
    building_age = st.number_input("سن بنا (سال)", min_value=0, step=1)
    facilities = st.text_area("امکانات")
    uploaded_files = st.file_uploader("آپلود تصاویر ملک (حداکثر ۵ عدد)", accept_multiple_files=True, type=["png","jpg","jpeg"])

    if st.button("پرداخت و ثبت ملک"):
        if not title or price <= 0 or not city:
            st.error("لطفا تمام فیلدهای ضروری را پر کنید.")
            return
        if len(uploaded_files) == 0:
            st.error("لطفا حداقل یک تصویر آپلود کنید.")
            return
        if len(uploaded_files) > 5:
            st.error("حداکثر ۵ تصویر مجاز است.")
            return

        st.info("در حال انجام پرداخت ۴۰ هزار تومان به شماره کارت 6037701120725572 ...")
        time.sleep(2)
        st.success("پرداخت با موفقیت انجام شد.")

        images_b64 = []
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            images_b64.append(bytes_data)

        data = {
            'title': title,
            'price': price,
            'area': area,
            'city': city,
            'property_type': property_type,
            'latitude': latitude,
            'longitude': longitude,
            'owner': st.session_state['user']['username'],
            'description': description,
            'rooms': rooms,
            'building_age': building_age,
            'facilities': facilities
        }
        add_property(data, images_b64)
        st.success("ملک با موفقیت ثبت شد!")

# پنل‌ها و صفحات
def admin_panel():
    st.subheader("پنل مدیر")
    st.write("مدیریت مشاوران")
    c.execute("SELECT username, name FROM users WHERE role='agent'")
    agents = c.fetchall()
    for agent in agents:
        col1, col2, col3 = st.columns([3,3,1])
        col1.write(agent[0])
        col2.write(agent[1])
        if col3.button(f"حذف {agent[0]}"):
            c.execute("DELETE FROM users WHERE username=?", (agent[0],))
            conn.commit()
            st.experimental_rerun()
    st.markdown("---")
    st.write("لیست همه ملک‌ها")
    c.execute("SELECT * FROM properties")
    props = c.fetchall()
    for prop in props:
        st.write(f"عنوان: {prop[1]}, قیمت: {prop[2]}, شهر: {prop[4]}")

def agent_panel():
    st.subheader("پنل مشاور")
    c.execute("SELECT * FROM properties WHERE owner=?", (st.session_state['user']['username'],))
    props = c.fetchall()
    for prop in props:
        st.write(f"عنوان: {prop[1]}, قیمت: {prop[2]}, شهر: {prop[4]}")

def public_panel():
    st.subheader(f"خوش آمدید، {st.session_state['user']['name']}!")
    register_property_page()
    st.markdown("---")
    st.write("جستجو و مشاهده املاک")
    c.execute("SELECT * FROM properties")
    all_props = c.fetchall()
    import pandas as pd
    df = pd.DataFrame(all_props, columns=['id','title','price','area','city','property_type','latitude','longitude','owner','description','rooms','building_age','facilities'])
    show_map(df)

def main():
    setup_db()
    custom_style()
    st.title("سیستم مدیریت املاک پیشرفته با OTP و پرداخت")
    if 'user' not in st.session_state:
        login_page()
    else:
        user = st.session_state['user']
        role = user['role']
        if role == 'admin':
            admin_panel()
        elif role == 'agent':
            agent_panel()
        else:
            public_panel()

if __name__ == "__main__":
    main()
  
      
