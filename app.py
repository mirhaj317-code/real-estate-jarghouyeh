import streamlit as st
import sqlite3
import hashlib
import random
import string
import folium
from streamlit_folium import st_folium

# -----------------------------
# دیتابیس
# -----------------------------
def get_connection():
    return sqlite3.connect("real_estate.db")

def create_tables():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            role TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            description TEXT,
            price REAL,
            latitude REAL,
            longitude REAL,
            is_public INTEGER,
            paid INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

# -----------------------------
# رمزنگاری پسورد
# -----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# مدیریت کاربر
# -----------------------------
def add_user(name, email, password, role="user"):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
              (name, email, hash_password(password), role))
    conn.commit()
    conn.close()

def login_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, password_hash, role, name FROM users WHERE email=?", (email,))
    data = c.fetchone()
    conn.close()
    if data and data[1] == hash_password(password):
        return {"id": data[0], "role": data[2], "name": data[3]}
    return None

def reset_password(email, new_password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET password_hash=? WHERE email=?",
              (hash_password(new_password), email))
    conn.commit()
    conn.close()

# -----------------------------
# مدیریت ملک
# -----------------------------
def add_property(user_id, title, desc, price, lat, lon, is_public, paid):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO properties (user_id, title, description, price, latitude, longitude, is_public, paid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, title, desc, price, lat, lon, is_public, paid))
    conn.commit()
    conn.close()

def get_properties(filter_public=True):
    conn = get_connection()
    c = conn.cursor()
    if filter_public:
        c.execute("SELECT * FROM properties WHERE is_public=1 AND paid=1")
    else:
        c.execute("SELECT * FROM properties")
    data = c.fetchall()
    conn.close()
    return data

# -----------------------------
# پرداخت آنلاین شبیه‌سازی‌شده
# -----------------------------
def process_payment(amount):
    # اینجا میتونی با درگاه واقعی جایگزین کنی
    st.info(f"پرداخت {amount} تومان در حال انجام...")
    time.sleep(2)
    st.success("پرداخت موفق ✅")
    return True

# -----------------------------
# صفحات
# -----------------------------
def signup_page():
    st.subheader("ثبت‌نام")
    name = st.text_input("نام کامل")
    email = st.text_input("ایمیل")
    password = st.text_input("رمز عبور", type="password")
    if st.button("ثبت‌نام"):
        if name and email and password:
            try:
                add_user(name, email, password)
                st.success("ثبت‌نام با موفقیت انجام شد ✅ حالا وارد شوید.")
            except:
                st.error("این ایمیل قبلا ثبت شده است ❌")
        else:
            st.warning("لطفاً همه‌ی فیلدها را پر کنید.")

def login_page():
    st.subheader("ورود")
    email = st.text_input("ایمیل")
    password = st.text_input("رمز عبور", type="password")
    if st.button("ورود"):
        user = login_user(email, password)
        if user:
            st.session_state["user"] = user
            st.success(f"خوش آمدی {user['name']} 🌹")
        else:
            st.error("ایمیل یا رمز عبور اشتباه است ❌")
    if st.button("فراموشی رمز عبور"):
        reset_password_page()

def reset_password_page():
    st.subheader("بازیابی رمز عبور")
    email = st.text_input("ایمیل ثبت‌شده")
    new_pass = st.text_input("رمز جدید", type="password")
    if st.button("تغییر رمز"):
        reset_password(email, new_pass)
        st.success("رمز عبور با موفقیت تغییر کرد ✅")

def add_property_page():
    st.subheader("ثبت ملک جدید")
    title = st.text_input("عنوان ملک")
    desc = st.text_area("توضیحات")
    price = st.number_input("قیمت (تومان)", min_value=0)
    lat = st.number_input("عرض جغرافیایی", format="%.6f")
    lon = st.number_input("طول جغرافیایی", format="%.6f")
    is_public = st.checkbox("نمایش عمومی (نیاز به پرداخت)")
    paid = 0
    if is_public and st.button("پرداخت و ثبت ملک"):
        if process_payment(price):
            paid = 1
            add_property(st.session_state["user"]["id"], title, desc, price, lat, lon, 1, paid)
    elif st.button("ثبت ملک"):
        add_property(st.session_state["user"]["id"], title, desc, price, lat, lon, 0, paid)
        st.success("ملک ثبت شد ✅")

def show_properties_page():
    st.subheader("املاک ثبت‌شده")
    data = get_properties()
    for p in data:
        st.write(f"🏠 {p[2]} | {p[4]} تومان")
        st.write(f"{p[3]}")
        if p[5] and p[6]:
            m = folium.Map(location=[p[5], p[6]], zoom_start=15)
            st_folium(m, width=700, height=400)

def user_dashboard():
    st.title("داشبورد کاربر")
    st.write("اینجا کارهای مخصوص کاربران نمایش داده می‌شود...")
    add_property_page()
    show_properties_page()

def admin_dashboard():
    st.title("داشبورد ادمین")
    st.write("اینجا مدیریت کامل املاک و کاربران است...")
    add_property_page()
    show_properties_page()

# -----------------------------
# اصلی
# -----------------------------
def main():
    st.sidebar.title("منوی اصلی")
    menu = st.sidebar.selectbox("برو به صفحه", ["ورود", "ثبت‌نام"])
    create_tables()

    if "user" not in st.session_state:
        if menu == "ثبت‌نام":
            signup_page()
        else:
            login_page()
    else:
        role = st.session_state["user"]["role"]
        if role == "admin":
            admin_dashboard()
        else:
            user_dashboard()
        if st.sidebar.button("خروج"):
            del st.session_state["user"]
            st.rerun()

if __name__ == '__main__':
    main()
