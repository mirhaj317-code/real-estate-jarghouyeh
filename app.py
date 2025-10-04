# app.py — اپلیکیشن املاک و مستغلات شهرستان جرقویه (نسخه کامل + هوش مصنوعی)
import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import folium
import math
import json
import re
import time
import os
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import base64
import io
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import jdatetime
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
from faker import Faker
from streamlit_echarts import st_echarts
import pytz

# =========================
# CONFIG / SETTINGS پیشرفته
# =========================
DB_NAME = "real_estate_jargouyeh.db"
ADMIN_EMAIL = "mirhaj57@gmail.com"  # تنها مدیر سیستم
DEFAULT_LISTING_FEE = 20000  # تومان
MAX_UPLOAD_IMAGES = 8
MAX_IMAGE_SIZE_MB = 8
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
COMMENT_COOLDOWN_SEC = 15
CHAT_COOLDOWN_SEC = 8
BACKUP_DIR = "backups"
os.makedirs(BACKUP_DIR, exist_ok=True)
CACHE_TTL = 300

# =========================
# AI MARKET ANALYTICS - تحلیل بازار هوشمند
# =========================
class RealEstateAnalytics:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        
    def prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """آماده‌سازی داده‌های بازار برای تحلیل"""
        features = []
        
        for _, row in df.iterrows():
            feature = {
                'area': row['area'],
                'rooms': row.get('rooms', 0),
                'building_age': row.get('building_age', 0),
                'latitude': row.get('latitude', 0),
                'longitude': row.get('longitude', 0),
                'city_encoded': self._encode_city(row['city']),
                'type_encoded': self._encode_property_type(row['property_type'])
            }
            features.append(feature)
            
        return pd.DataFrame(features)
    
    def _encode_city(self, city: str) -> int:
        cities = {"جرقویه": 1, "اصفهان": 2, "شهرضا": 3, "نجف آباد": 4}
        return cities.get(city, 0)
    
    def _encode_property_type(self, prop_type: str) -> int:
        types = {"آپارتمان": 1, "ویلایی": 2, "مغازه": 3, "زمین": 4, "دفتر": 5}
        return types.get(prop_type, 0)
    
    def train_price_model(self, df: pd.DataFrame):
        """آموزش مدل پیش‌بینی قیمت"""
        if df.empty:
            return
            
        X = self.prepare_market_data(df)
        y = df['price'].values
        
        # حذف مقادیر نامعتبر
        valid_indices = ~(X.isna().any(axis=1) | pd.isna(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) > 5:  # حداقل داده برای آموزش
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
    
    def predict_price(self, property_data: Dict[str, Any]) -> float:
        """پیش‌بینی قیمت برای یک ملک جدید"""
        if not hasattr(self, 'is_trained'):
            return property_data.get('price', 0)
            
        X = self.prepare_market_data(pd.DataFrame([property_data]))
        X_scaled = self.scaler.transform(X)
        predicted_price = self.model.predict(X_scaled)[0]
        
        return max(0, predicted_price)
    
    def get_market_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل روندهای بازار"""
        if df.empty:
            return {}
            
        trends = {
            'total_properties': len(df),
            'avg_price': df['price'].mean(),
            'avg_price_per_meter': (df['price'] / df['area']).mean(),
            'popular_cities': df['city'].value_counts().head(3).to_dict(),
            'popular_types': df['property_type'].value_counts().head(3).to_dict(),
            'price_by_city': df.groupby('city')['price'].mean().to_dict(),
            'price_by_type': df.groupby('property_type')['price'].mean().to_dict()
        }
        
        # محاسبه تغییرات قیمت در طول زمان
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            monthly_trend = df.groupby(df['created_at'].dt.to_period('M'))['price'].mean()
            trends['monthly_trend'] = monthly_trend.to_dict()
        
        return trends

# =========================
# AI PROPERTY RECOMMENDER - سیستم پیشنهاد هوشمند
# =========================
class PropertyRecommender:
    """کلاس PropertyRecommender که در کد فراخوانی شده بود"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.similarity_threshold = 0.3
        
    def update_recommendations_for_user(self, user_email: str):
        """به‌روزرسانی پیشنهادات برای کاربر - تابعی که در کد فراخوانی شده بود"""
        try:
            conn = get_conn()
            
            # دریافت علاقه‌مندی‌های کاربر
            c = conn.cursor()
            c.execute("""
                SELECT p.* FROM favorites f
                JOIN properties p ON f.property_id = p.id
                WHERE f.user_email = ? AND p.status = 'published'
            """, (user_email,))
            user_favorites = c.fetchall()
            
            if not user_favorites:
                conn.close()
                return
                
            # دریافت تمام املاک فعال
            properties_df = pd.read_sql("""
                SELECT * FROM properties WHERE status='published'
            """, conn)
            
            if properties_df.empty:
                conn.close()
                return
                
            # آموزش مدل
            features = self._extract_features(properties_df)
            feature_matrix = self.vectorizer.fit_transform(features)
            
            # محاسبه شباهت
            fav_indices = []
            for fav in user_favorites:
                for i, prop_id in enumerate(properties_df['id']):
                    if prop_id == fav[0]:
                        fav_indices.append(i)
                        break
            
            if not fav_indices:
                conn.close()
                return
                
            fav_vectors = feature_matrix[fav_indices]
            mean_fav_vector = fav_vectors.mean(axis=0)
            similarities = cosine_similarity(mean_fav_vector, feature_matrix).flatten()
            
            # ذخیره پیشنهادات در دیتابیس
            c.execute("DELETE FROM ai_recommendations WHERE user_email=?", (user_email,))
            
            top_indices = similarities.argsort()[-10:][::-1]
            for i in top_indices:
                if similarities[i] > self.similarity_threshold and properties_df.iloc[i]['owner_email'] != user_email:
                    reason = self._generate_reason(properties_df.iloc[i], user_favorites)
                    c.execute("""
                        INSERT INTO ai_recommendations (user_email, property_id, score, reason, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_email, properties_df.iloc[i]['id'], float(similarities[i]), reason, now_iso()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating recommendations: {e}")

    def _extract_features(self, df: pd.DataFrame) -> List[str]:
        """استخراج ویژگی‌ها از داده‌های ملک"""
        features = []
        for _, row in df.iterrows():
            feature_text = (
                f"{row['property_type']} {row['city']} "
                f"{row.get('facilities', '')} {row.get('description', '')} "
                f"{'آسانسور' if 'آسانسور' in str(row.get('facilities', '')) else ''} "
                f"{'پارکینگ' if 'پارکینگ' in str(row.get('facilities', '')) else ''} "
                f"{'استخر' if 'استخر' in str(row.get('facilities', '')) else ''}"
            )
            features.append(feature_text)
        return features

    def _generate_reason(self, property_data, user_favorites) -> str:
        """تولید دلیل پیشنهاد"""
        reasons = []
        
        # بررسی تطابق نوع ملک
        fav_types = [fav[4] for fav in user_favorites]  # property_type
        if property_data['property_type'] in fav_types:
            reasons.append(f"نوع {property_data['property_type']} مشابه علاقه‌مندی‌های شما")
        
        # بررسی تطابق شهر
        fav_cities = [fav[3] for fav in user_favorites]  # city
        if property_data['city'] in fav_cities:
            reasons.append(f"موقعیت در {property_data['city']} منطبق بر ترجیحات شما")
        
        return " - ".join(reasons) if reasons else "پیشنهاد مبتنی بر تحلیل علاقه‌مندی‌های شما"

class AdvancedPropertyRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.similarity_threshold = 0.3
        
    def enhanced_property_features(self, df: pd.DataFrame) -> List[str]:
        """استخراج ویژگی‌های پیشرفته از ملک‌ها"""
        features = []
        for _, row in df.iterrows():
            feature_text = (
                f"{row['property_type']} {row['city']} "
                f"{row.get('facilities', '')} {row.get('description', '')} "
                f"{'آسانسور' if 'آسانسور' in str(row.get('facilities', '')) else ''} "
                f"{'پارکینグ' if 'پارکینگ' in str(row.get('facilities', '')) else ''} "
                f"{'استخر' if 'استخر' in str(row.get('facilities', '')) else ''}"
            )
            features.append(feature_text)
        return features
    
    def train_advanced_model(self, properties_df: pd.DataFrame, user_behavior: pd.DataFrame = None):
        """آموزش مدل پیشرفته با درنظرگیری رفتار کاربران"""
        features = self.enhanced_property_features(properties_df)
        self.feature_matrix = self.vectorizer.fit_transform(features)
        self.property_ids = properties_df['id'].tolist()
        self.properties_data = properties_df
        
        # اضافه کردن تحلیل رفتار کاربران
        if user_behavior is not None and not user_behavior.empty:
            self.user_preferences = self._analyze_user_behavior(user_behavior)
        else:
            self.user_preferences = {}
    
    def _analyze_user_behavior(self, user_behavior: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل رفتار کاربران برای شخصی‌سازی بهتر"""
        preferences = {}
        
        for user_email, group in user_behavior.groupby('user_email'):
            user_prefs = {
                'preferred_types': group['property_type'].value_counts().head(3).index.tolist(),
                'preferred_cities': group['city'].value_counts().head(3).index.tolist(),
                'avg_price_range': (group['price'].min(), group['price'].max()),
                'preferred_amenities': self._extract_common_amenities(group)
            }
            preferences[user_email] = user_prefs
            
        return preferences
    
    def _extract_common_amenities(self, user_properties: pd.DataFrame) -> List[str]:
        """استخراج امکانات مورد علاقه کاربر"""
        all_amenities = []
        for facilities in user_properties['facilities'].dropna():
            if isinstance(facilities, str):
                all_amenities.extend([amenity.strip() for amenity in facilities.split(',')])
        
        from collections import Counter
        common_amenities = [amenity for amenity, count in Counter(all_amenities).most_common(5) if count > 1]
        return common_amenities
    
    def get_personalized_recommendations(self, user_email: str, top_n: int = 5) -> List[Tuple[int, float, str]]:
        """دریافت پیشنهادات شخصی‌سازی شده"""
        if not hasattr(self, 'feature_matrix'):
            return []
            
        user_favorites = self._get_user_favorites(user_email)
        
        if not user_favorites:
            return self._get_popular_recommendations(top_n)
        
        # محاسبه شباهت بر اساس علاقه‌مندی‌های کاربر
        fav_indices = [i for i, pid in enumerate(self.property_ids) if pid in user_favorites]
        
        if not fav_indices:
            return self._get_popular_recommendations(top_n)
            
        fav_vectors = self.feature_matrix[fav_indices]
        mean_fav_vector = fav_vectors.mean(axis=0)
        
        similarities = cosine_similarity(mean_fav_vector, self.feature_matrix).flatten()
        
        # اعمال فیلترهای شخصی‌سازی شده
        if user_email in self.user_preferences:
            similarities = self._apply_user_preferences(user_email, similarities)
        
        # حذف ملک‌های مورد علاقه از نتایج
        for i in fav_indices:
            similarities[i] = -1
            
        # انتخاب بهترین پیشنهادات
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = []
        
        for i in top_indices:
            if similarities[i] > self.similarity_threshold:
                reason = self._generate_recommendation_reason(user_email, i, similarities[i])
                recommendations.append((self.property_ids[i], similarities[i], reason))
        
        return recommendations
    
    def _get_user_favorites(self, user_email: str) -> List[int]:
        """دریافت علاقه‌مندی‌های کاربر از دیتابیس"""
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT property_id FROM favorites WHERE user_email=?", (user_email,))
        favorites = [row[0] for row in c.fetchall()]
        conn.close()
        return favorites
    
    def _get_popular_recommendations(self, top_n: int) -> List[Tuple[int, float, str]]:
        """پیشنهاد املاک محبوب در صورت عدم وجود علاقه‌مندی"""
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            SELECT p.id, COUNT(f.id) as fav_count 
            FROM properties p 
            LEFT JOIN favorites f ON p.id = f.property_id 
            WHERE p.status='published' 
            GROUP BY p.id 
            ORDER BY fav_count DESC, p.views DESC 
            LIMIT ?
        """, (top_n,))
        popular = [(row[0], 0.5, "ملک محبوب در سیستم") for row in c.fetchall()]
        conn.close()
        return popular
    
    def _apply_user_preferences(self, user_email: str, similarities: np.ndarray) -> np.ndarray:
        """اعمال ترجیحات کاربر بر روی امتیازهای شباهت"""
        prefs = self.user_preferences[user_email]
        
        for i, prop_id in enumerate(self.property_ids):
            prop_data = self.properties_data[self.properties_data['id'] == prop_id].iloc[0]
            
            # تطابق نوع ملک
            if prop_data['property_type'] in prefs['preferred_types']:
                similarities[i] *= 1.2
                
            # تطابق شهر
            if prop_data['city'] in prefs['preferred_cities']:
                similarities[i] *= 1.15
                
            # تطابق محدوده قیمت
            min_price, max_price = prefs['avg_price_range']
            if min_price <= prop_data['price'] <= max_price:
                similarities[i] *= 1.1
        
        return similarities
    
    def _generate_recommendation_reason(self, user_email: str, prop_index: int, similarity: float) -> str:
        """تولید دلیل پیشنهاد برای کاربر"""
        prop_data = self.properties_data.iloc[prop_index]
        reasons = []
        
        if user_email in self.user_preferences:
            prefs = self.user_preferences[user_email]
            
            if prop_data['property_type'] in prefs['preferred_types']:
                reasons.append(f"نوع {prop_data['property_type']} مشابه انتخاب‌های قبلی شما")
                
            if prop_data['city'] in prefs['preferred_cities']:
                reasons.append(f"موقعیت در {prop_data['city']} منطبق بر ترجیحات شما")
        
        if similarity > 0.7:
            reasons.append("شباهت بسیار بالا با علاقه‌مندی‌های شما")
        elif similarity > 0.5:
            reasons.append("شباهت بالا با سلیقه شما")
        else:
            reasons.append("پیشنهاد مبتنی بر تحلیل بازار")
        
        return " - ".join(reasons) if reasons else "پیشنهاد هوشمند سیستم"

# =========================
# SMART PRICE ADVISOR - مشاور قیمت هوشمند
# =========================
class SmartPriceAdvisor:
    def __init__(self):
        self.analytics = RealEstateAnalytics()
        
    def analyze_property_value(self, property_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """آنالیز ارزش ملک و ارائه توصیه قیمت"""
        self.analytics.train_price_model(market_data)
        
        predicted_price = self.analytics.predict_price(property_data)
        actual_price = property_data.get('price', 0)
        
        analysis = {
            'predicted_price': int(predicted_price),
            'actual_price': actual_price,
            'price_difference': int(predicted_price - actual_price),
            'price_ratio': predicted_price / actual_price if actual_price > 0 else 1,
            'market_analysis': self.analytics.get_market_trends(market_data)
        }
        
        # تولید توصیه
        analysis['recommendation'] = self._generate_price_recommendation(analysis)
        analysis['confidence_score'] = self._calculate_confidence_score(market_data)
        
        return analysis
    
    def _generate_price_recommendation(self, analysis: Dict[str, Any]) -> str:
        """تولید توصیه قیمت بر اساس تحلیل"""
        ratio = analysis['price_ratio']
        diff = analysis['price_difference']
        
        if ratio > 1.2:
            return f"✅ قیمت بسیار مناسب - {diff:,} تومان کمتر از ارزش بازار"
        elif ratio > 1.1:
            return f"💰 قیمت مناسب - {diff:,} تومان کمتر از ارزش بازار"
        elif ratio > 0.9:
            return "⚖️ قیمت منطقی - نزدیک به ارزش بازار"
        elif ratio > 0.8:
            return f"📈 قیمت کمی بالا - {abs(diff):,} تومان بیشتر از ارزش بازار"
        else:
            return f"⚠️ قیمت بالا - {abs(diff):,} تومان بیشتر از ارزش بازار"
    
    def _calculate_confidence_score(self, market_data: pd.DataFrame) -> float:
        """محاسبه امتیاز اطمینان از تحلیل"""
        if len(market_data) < 10:
            return 0.3
        elif len(market_data) < 50:
            return 0.6
        else:
            return 0.9

# =========================
# INITIALIZE AI SYSTEMS - راه‌اندازی سیستم‌های هوشمند
# =========================
market_analytics = RealEstateAnalytics()
property_recommender = AdvancedPropertyRecommender()
price_advisor = SmartPriceAdvisor()
simple_property_recommender = PropertyRecommender()  # اضافه کردن PropertyRecommender

def initialize_ai_systems():
    """راه‌اندازی اولیه سیستم‌های هوشمند"""
    try:
        conn = get_conn()
        
        # بارگذاری داده‌های بازار
        properties_df = pd.read_sql("""
            SELECT * FROM properties WHERE status='published'
        """, conn)
        
        # بارگذاری رفتار کاربران
        user_behavior_df = pd.read_sql("""
            SELECT f.user_email, p.property_type, p.city, p.price, p.facilities
            FROM favorites f
            JOIN properties p ON f.property_id = p.id
        """, conn)
        
        conn.close()
        
        if not properties_df.empty:
            # آموزش سیستم‌های هوشمند
            market_analytics.train_price_model(properties_df)
            property_recommender.train_advanced_model(properties_df, user_behavior_df)
            
        return True
    except Exception as e:
        st.error(f"خطا در راه‌اندازی سیستم هوشمند: {e}")
        return False

# =========================
# UTIL — DB CONNECTION / MIGRATIONS پیشرفته
# =========================
def get_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def migrate_db():
    conn = get_conn(); c = conn.cursor()

    # ایجاد جداول اصلی با بهبودهای عملکرد
    tables = [
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'public' CHECK(role IN ('public', 'agent', 'admin')),
            phone TEXT,
            rating REAL DEFAULT 5.0 CHECK(rating >= 0 AND rating <= 5),
            created_at TEXT,
            last_login TEXT,
            profile_image BLOB,
            bio TEXT,
            verified INTEGER DEFAULT 0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS properties(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price INTEGER NOT NULL CHECK(price >= 0),
            area INTEGER NOT NULL CHECK(area > 0),
            city TEXT NOT NULL,
            property_type TEXT NOT NULL,
            latitude REAL CHECK(latitude >= -90 AND latitude <= 90),
            longitude REAL CHECK(longitude >= -180 AND longitude <= 180),
            address TEXT,
            owner_email TEXT NOT NULL,
            description TEXT,
            rooms INTEGER DEFAULT 0 CHECK(rooms >= 0),
            building_age INTEGER DEFAULT 0 CHECK(building_age >= 0),
            facilities TEXT,
            video_url TEXT,
            status TEXT DEFAULT 'draft' CHECK(status IN ('draft', 'published', 'sold', 'archived')),
            views INTEGER DEFAULT 0 CHECK(views >= 0),
            featured INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(owner_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS images(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            image BLOB NOT NULL,
            is_primary INTEGER DEFAULT 0,
            uploaded_at TEXT,
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS comments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            user_email TEXT NOT NULL,
            comment TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
            created_at TEXT,
            helpful INTEGER DEFAULT 0,
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS favorites(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            user_email TEXT NOT NULL,
            created_at TEXT,
            notes TEXT,
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE,
            UNIQUE(property_id, user_email)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            sender_email TEXT NOT NULL,
            receiver_email TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT,
            is_read INTEGER DEFAULT 0 CHECK(is_read IN (0, 1)),
            message_type TEXT DEFAULT 'text' CHECK(message_type IN ('text', 'image', 'document')),
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY(sender_email) REFERENCES users(email) ON DELETE CASCADE,
            FOREIGN KEY(receiver_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS payments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_temp_json TEXT NOT NULL,
            user_email TEXT NOT NULL,
            amount INTEGER NOT NULL CHECK(amount >= 0),
            authority TEXT,
            ref_id TEXT,
            status TEXT CHECK(status IN ('initiated', 'paid', 'failed', 'refunded')),
            created_at TEXT,
            updated_at TEXT,
            payment_gateway TEXT DEFAULT 'zarinpal' CHECK(payment_gateway IN ('zarinpal', 'idpay', 'nextpay')),
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT CHECK(type IN ('info', 'success', 'warning', 'error', 'message', 'comment', 'favorite')),
            is_read INTEGER DEFAULT 0 CHECK(is_read IN (0, 1)),
            created_at TEXT,
            related_id INTEGER,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS user_activity(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TEXT,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS property_views(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            user_agent TEXT,
            ip_address TEXT,
            created_at TEXT,
            user_email TEXT,
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS subscriptions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            search_filters TEXT NOT NULL,
            is_active INTEGER DEFAULT 1 CHECK(is_active IN (0, 1)),
            created_at TEXT,
            updated_at TEXT,
            frequency TEXT DEFAULT 'instant' CHECK(frequency IN ('instant', 'daily', 'weekly')),
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reporter_email TEXT NOT NULL,
            reported_item_type TEXT CHECK(reported_item_type IN ('property', 'comment', 'user')),
            reported_item_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'reviewed', 'resolved')),
            created_at TEXT,
            resolved_at TEXT,
            resolved_by TEXT,
            FOREIGN KEY(reporter_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            buyer_email TEXT NOT NULL,
            seller_email TEXT NOT NULL,
            price INTEGER NOT NULL,
            transaction_date TEXT,
            status TEXT CHECK(status IN ('pending', 'completed', 'cancelled')),
            contract_url TEXT,
            created_at TEXT,
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE,
            FOREIGN KEY(buyer_email) REFERENCES users(email) ON DELETE CASCADE,
            FOREIGN KEY(seller_email) REFERENCES users(email) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ai_recommendations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            property_id INTEGER NOT NULL,
            score REAL NOT NULL,
            reason TEXT,
            created_at TEXT,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE,
            FOREIGN KEY(property_id) REFERENCES properties(id) ON DELETE CASCADE
        );
        """
    ]

    for table in tables:
        c.execute(table)

    # ایجاد ایندکس‌ها برای بهبود عملکرد
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_properties_city ON properties(city)",
        "CREATE INDEX IF NOT EXISTS idx_properties_type ON properties(property_type)",
        "CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price)",
        "CREATE INDEX IF NOT EXISTS idx_properties_status ON properties(status)",
        "CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_email)",
        "CREATE INDEX IF NOT EXISTS idx_comments_property ON comments(property_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_email)",
        "CREATE INDEX IF NOT EXISTS idx_messages_receiver ON messages(receiver_email)",
        "CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_email)",
        "CREATE INDEX IF NOT EXISTS idx_activity_user ON user_activity(user_email)",
        "CREATE INDEX IF NOT EXISTS idx_views_property ON property_views(property_id)",
        "CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_email)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_buyer ON transactions(buyer_email)",
        "CREATE INDEX IF NOT EXISTS idx_ai_recommendations_user ON ai_recommendations(user_email)"
    ]

    for index in indexes:
        try:
            c.execute(index)
        except sqlite3.Error as e:
            st.error(f"خطا در ایجاد ایندکس: {e}")

    # اضافه کردن ستون‌های جدید در صورت عدم وجود
    safe_alters = [
        ("users", "last_login", "TEXT"),
        ("users", "profile_image", "BLOB"),
        ("users", "bio", "TEXT"),
        ("users", "verified", "INTEGER DEFAULT 0"),
        ("properties", "featured", "INTEGER DEFAULT 0"),
        ("properties", "updated_at", "TEXT"),
        ("images", "is_primary", "INTEGER DEFAULT 0"),
        ("images", "uploaded_at", "TEXT"),
        ("comments", "helpful", "INTEGER DEFAULT 0"),
        ("favorites", "notes", "TEXT"),
        ("messages", "message_type", "TEXT DEFAULT 'text'"),
        ("payments", "updated_at", "TEXT"),
        ("payments", "payment_gateway", "TEXT DEFAULT 'zarinpal'"),
        ("notifications", "related_id", "INTEGER"),
        ("user_activity", "ip_address", "TEXT"),
        ("user_activity", "user_agent", "TEXT"),
        ("property_views", "user_email", "TEXT"),
        ("subscriptions", "updated_at", "TEXT"),
        ("subscriptions", "frequency", "TEXT DEFAULT 'instant'"),
        ("reports", "resolved_at", "TEXT"),
        ("reports", "resolved_by", "TEXT")
    ]
    
    for table, col, coltype in safe_alters:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass

    # ایجاد کاربر مدیر در صورت عدم وجود
    c.execute("SELECT * FROM users WHERE email=?", (ADMIN_EMAIL,))
    if not c.fetchone():
        admin_password_hash = hash_password("admin123!")  # رمز عبور پیش‌فرض
        c.execute("""
            INSERT INTO users (name, email, password_hash, role, verified, created_at) 
            VALUES (?, ?, ?, 'admin', 1, ?)
        """, ("مدیر سیستم", ADMIN_EMAIL, admin_password_hash, now_iso()))

    conn.commit(); conn.close()

# =========================
# AUTH پیشرفته
# =========================
def hash_password(password: str) -> str:
    salt = os.urandom(32)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex()

def valid_email(email:str)->bool:
    return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email or ""))

def valid_phone(phone:str)->bool:
    if not phone: return True
    return bool(re.match(r"^(\+98|0)?9\d{9}$", phone.replace(" ", "")))

def strong_password(pw:str)->bool:
    if not pw or len(pw) < 6: return False  # از ۸ به ۶ کاهش دادم
    return True

def simple_password(pw:str)->bool:
    return len(pw) >= 4  # پسورد ۴ کاراکتری هم قبول کن

def register_user(name: str, email: str, password: str, role="public", phone=None, bio=None) -> bool:
    if not (name and valid_email(email) and (strong_password(password) or simple_password(password)) and valid_phone(phone or "")):
        return False
    try:
        conn = get_conn(); c = conn.cursor()
        c.execute("INSERT INTO users(name,email,password_hash,role,phone,bio,created_at) VALUES(?,?,?,?,?,?,?)",
                  (name.strip(), email.strip().lower(), hash_password(password), role, (phone or "").strip() or None, bio, now_iso()))
        conn.commit(); conn.close()
        
        track_user_activity(email, "register", f"User {name} registered successfully")
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email: str, password: str) -> Optional[Dict[str,Any]]:
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT name, role, phone, password_hash, email, bio, verified FROM users WHERE email=?", (email.strip().lower(),))
    row = c.fetchone(); conn.close()
    
    if not row: return None
    name, role, phone, ph, em, bio, verified = row
    
    # بررسی دسترسی مدیر
    if em == ADMIN_EMAIL and role != 'admin':
        # بروزرسانی نقش به مدیر
        conn = get_conn(); c = conn.cursor()
        c.execute("UPDATE users SET role='admin' WHERE email=?", (ADMIN_EMAIL,))
        conn.commit(); conn.close()
        role = 'admin'
    
    if verify_password(password, ph):
        conn = get_conn(); c = conn.cursor()
        c.execute("UPDATE users SET last_login=? WHERE email=?", (now_iso(), em))
        conn.commit(); conn.close()
        
        track_user_activity(em, "login", f"User {name} logged in successfully")
        return {"email": em, "name": name, "role": role, "phone": phone, "bio": bio, "verified": verified}
    return None

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def reset_password(email: str, new_password: str) -> bool:
    if not (valid_email(email) and (strong_password(new_password) or simple_password(new_password))): return False
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE users SET password_hash=? WHERE email=?", (hash_password(new_password), email.strip().lower()))
    conn.commit(); ok = c.rowcount>0; conn.close()
    
    if ok:
        track_user_activity(email, "password_reset", "Password reset successfully")
    return ok

# =========================
# UTIL HELPERS پیشرفته
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    dLat=math.radians(lat2-lat1)
    dLon=math.radians(lon2-lon1)
    a=math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def badge(text, color="#C5A572", bg_color="#fff"):
    st.markdown(f"<span class='pill' style='display:inline-block;background:{bg_color};border:1px solid {color};padding:6px 10px;border-radius:999px;margin:2px 4px;font-size:12px'>{text}</span>", unsafe_allow_html=True)

def now_iso():
    return datetime.utcnow().isoformat()

def cooldown_ok(key:str, seconds:int)->bool:
    last = st.session_state.get(key)
    if not last:
        st.session_state[key] = time.time()
        return True
    if time.time() - last >= seconds:
        st.session_state[key] = time.time()
        return True
    return False

def get_client_ip():
    try:
        return "127.0.0.1"
    except:
        return "unknown"

def format_price(price):
    if price >= 1000000000:
        return f"{price/1000000000:.1f} میلیارد"
    elif price >= 1000000:
        return f"{price/1000000:.1f} میلیون"
    else:
        return f"{price:,}"

def get_persian_date():
    now = jdatetime.datetime.now()
    return now.strftime("%Y/%m/%d")

def reshape_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# =========================
# SEO HELPERS پیشرفته
# =========================
def seo_meta(base_url:str, title:str, description:str, path:str="", image_url:str="", keywords:str=""):
    url = base_url.rstrip("/") + (path if path.startswith("/") else f"/{path}")
    tags = f"""
    <meta name="description" content="{description}"/>
    <meta name="keywords" content="{keywords or 'املاک, جرقویه, خرید ملک, فروش ملک, اجاره, آپارتمان, ویلا, مغازه'}"/>
    <link rel="canonical" href="{url}"/>
    <meta property="og:title" content="{title}"/>
    <meta property="og:description" content="{description}"/>
    <meta property="og:type" content="website"/>
    <meta property="og:url" content="{url}"/>
    {f'<meta property="og:image" content="{image_url}"/>' if image_url else ''}
    <meta name="twitter:card" content="summary_large_image"/>
    <meta name="twitter:title" content="{title}"/>
    <meta name="twitter:description" content="{description}"/>
    <meta name="robots" content="index, follow"/>
    """
    st.markdown(tags, unsafe_allow_html=True)

def jsonld_property(prop: dict, base_url:str) -> str:
    url = f"{base_url.rstrip('/')}/?pg=view&pid={prop['id']}"
    data = {
      "@context": "https://schema.org",
      "@type": "RealEstateListing",
      "url": url,
      "name": prop.get("title") or "",
      "description": (prop.get("description") or "")[:300],
      "address": {
        "@type": "PostalAddress",
        "streetAddress": prop.get("address") or "",
        "addressLocality": prop.get("city") or "",
        "addressCountry": "IR"
      },
      "offers": {
        "@type": "Offer",
        "price": int(prop.get("price") or 0),
        "priceCurrency": "IRR",
        "availability": "https://schema.org/InStock"
      },
      "numberOfRooms": int(prop.get("rooms") or 0),
      "floorSize": {
        "@type": "QuantitativeValue",
        "value": int(prop.get("area") or 0),
        "unitCode": "MTK"
      },
      "geo": {
        "@type": "GeoCoordinates",
        "latitude": float(prop.get("latitude") or 0),
        "longitude": float(prop.get("longitude") or 0)
      }
    }
    return json.dumps(data, ensure_ascii=False)

def generate_sitemap(base_url: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT id FROM properties WHERE status='published'")
    property_ids = [row[0] for row in c.fetchall()]
    conn.close()
    
    sitemap = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    
    sitemap.append(f'<url><loc>{base_url}</loc><changefreq>daily</changefreq><priority>1.0</priority></url>')
    
    for pid in property_ids:
        sitemap.append(f'<url><loc>{base_url}/?pg=view&pid={pid}</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>')
    
    sitemap.append('</urlset>')
    return '\n'.join(sitemap)

# =========================
# SIMPLE PAYMENT SYSTEM (بدون درگاه خارجی)
# =========================
def create_simple_payment(amount: int, description: str, user_email: str) -> str:
    """سیستم پرداخت ساده برای توسعه آینده"""
    payment_id = f"pay_{int(time.time())}_{user_email}"
    
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO payments (property_temp_json, user_email, amount, authority, status, created_at)
        VALUES (?, ?, ?, ?, 'initiated', ?)
    """, (description, user_email, amount, payment_id, now_iso()))
    conn.commit()
    conn.close()
    
    return payment_id

def complete_simple_payment(payment_id: str) -> bool:
    """تکمیل پرداخت ساده"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE payments SET status='paid', updated_at=? WHERE authority=?", 
              (now_iso(), payment_id))
    success = c.rowcount > 0
    conn.commit()
    conn.close()
    return success

# =========================
# DATA ACCESS پیشرفته
# =========================
def add_property_row(data: Dict[str,Any], images: List[bytes], publish:bool=False) -> int:
    conn=get_conn(); c=conn.cursor()
    c.execute("""INSERT INTO properties
    (title,price,area,city,property_type,latitude,longitude,address,owner_email,description,rooms,building_age,facilities,video_url,status,created_at,updated_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        data['title'].strip(), int(data['price']), int(data['area']), data['city'].strip(), data['property_type'],
        float(data['latitude']), float(data['longitude']), (data.get('address') or "").strip(),
        data['owner_email'], (data.get('description') or "").strip(), int(data.get('rooms') or 0),
        int(data.get('building_age') or 0), (data.get('facilities') or "").strip(), (data.get('video_url') or "").strip(),
        'published' if publish else 'draft', now_iso(), now_iso()
    ))
    pid = c.lastrowid
    
    for i, img in enumerate(images[:MAX_UPLOAD_IMAGES]):
        is_primary = 1 if i == 0 else 0
        c.execute("INSERT INTO images(property_id,image,is_primary,uploaded_at) VALUES(?,?,?,?)", 
                 (pid, img, is_primary, now_iso()))
    
    conn.commit(); conn.close()
    
    add_notification(data['owner_email'], f"ملک '{data['title']}' با موفقیت ثبت شد!", "success", related_id=pid)
    track_user_activity(data['owner_email'], "add_property", f"Added property {pid}: {data['title']}")
    
    check_subscriptions(data)
    update_recommendations_for_new_property(pid)
    
    return pid

@st.cache_data(ttl=CACHE_TTL)
def list_properties_df_cached(filters_json: str) -> pd.DataFrame:
    filters = json.loads(filters_json)
    return list_properties_df(filters)

def list_properties_df(filters: Dict[str,Any]) -> pd.DataFrame:
    conn=get_conn(); c=conn.cursor()
    query = "SELECT * FROM properties WHERE status='published'"
    params = []
    
    if filters.get("city"):
        placeholders = ','.join('?' * len(filters["city"]))
        query += f" AND city IN ({placeholders})"
        params.extend(filters["city"])
    
    if filters.get("property_type"):
        placeholders = ','.join('?' * len(filters["property_type"]))
        query += f" AND property_type IN ({placeholders})"
        params.extend(filters["property_type"])
    
    c.execute(query, params)
    rows = c.fetchall(); cols=[d[0] for d in c.description]; conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df

    def between(series, lo, hi):
        if lo is not None: series = series[series >= lo]
        if hi is not None: series = series[series <= hi]
        return series.index

    idx = df.index
    idx = idx.intersection(between(df["price"], filters.get("min_price"), filters.get("max_price")))
    idx = idx.intersection(between(df["area"], filters.get("min_area"), filters.get("max_area")))
    idx = idx.intersection(between(df["rooms"], filters.get("min_rooms"), filters.get("max_rooms")))
    idx = idx.intersection(between(df["building_age"], filters.get("min_age"), filters.get("max_age")))
    if len(idx)==0: return df.iloc[0:0]
    df = df.loc[idx]

    if filters.get("facilities"):
        for f in filters["facilities"]:
            df = df[df["facilities"].fillna("").str.contains(f, case=False, na=False)]

    if filters.get("center_lat") is not None and filters.get("center_lon") is not None and filters.get("radius_km"):
        center_lat = filters["center_lat"]; center_lon = filters["center_lon"]; R = filters["radius_km"]
        def in_radius(row):
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]): return False
            return haversine_km(center_lat, center_lon, row["latitude"], row["longitude"]) <= R
        df = df[df.apply(in_radius, axis=1)]

    return df.reset_index(drop=True)

def property_images(prop_id:int) -> List[bytes]:
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT image FROM images WHERE property_id=? ORDER BY is_primary DESC", (prop_id,))
    rows=c.fetchall(); conn.close()
    return [r[0] for r in rows]

def get_property_details(prop_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT * FROM properties WHERE id=?", (prop_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
        
    cols = [d[0] for d in c.description]
    prop = dict(zip(cols, row))
    
    c.execute("SELECT name, phone, rating, verified FROM users WHERE email=?", (prop["owner_email"],))
    owner = c.fetchone()
    if owner:
        prop["owner_name"] = owner[0]
        prop["owner_phone"] = owner[1]
        prop["owner_rating"] = owner[2]
        prop["owner_verified"] = owner[3]
    
    conn.close()
    return prop

# =========================
# MAP RENDER پیشرفته
# =========================
def show_map(df: pd.DataFrame, cluster=True):
    if df.empty:
        st.info("هیچ ملکی مطابق فیلترها پیدا نشد.")
        return
        
    center_lat = df['latitude'].dropna().mean()
    center_lon = df['longitude'].dropna().mean()
    
    if pd.isna(center_lat) or pd.isna(center_lon):
        center_lat, center_lon = 32.4279, 53.6880
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
    
    try:
        if cluster and len(df) > 10:
            from folium.plugins import MarkerCluster
            cluster = MarkerCluster().add_to(m)
        else:
            cluster = m
    except Exception:
        cluster = m
        
    for _, row in df.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]): continue
        
        html = f"""
        <div style='font-family:Tahoma; width: 250px;'>
            <h4 style='margin:0; color: #8B3A3A;'>{row["title"]}</h4>
            <p style='margin:5px 0;'>
                <b>نوع:</b> {row["property_type"]}<br>
                <b>شهر:</b> {row["city"]}<br>
                <b>قیمت:</b> {int(row["price"]):,} تومان<br>
                <b>متراژ:</b> {row["area"]} متر<br>
                <b>اتاق:</b> {row.get("rooms", 0)}
            </p>
            <p style='margin:0; font-size: 12px; color: #666;'>
                {(str(row.get("address") or ""))[:100]}
            </p>
            <a href='/?pg=view&pid={row["id"]}' 
               target='_blank' 
               style='display:block; text-align:center; background:#8B3A3A; color:white; padding:5px; border-radius:5px; margin-top:10px; text-decoration:none;'>
                مشاهده جزئیات
            </a>
        </div>
        """
        
        icon_color = "red"
        icon_type = "home"
        
        if row["property_type"] == "ویلایی":
            icon_color = "green"
            icon_type = "tree"
        elif row["property_type"] == "مغازه":
            icon_color = "blue"
            icon_type = "shopping-cart"
        elif row["property_type"] == "زمین":
            icon_color = "orange"
            icon_type = "map"
        elif row["property_type"] == "دفتر":
            icon_color = "purple"
            icon_type = "briefcase"
            
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=folium.Popup(html, max_width=300),
            tooltip=row["title"],
            icon=folium.Icon(color=icon_color, icon=icon_type, prefix="fa")
        ).add_to(cluster)
    
    from folium.plugins import MeasureControl
    m.add_child(MeasureControl())
    
    st_folium(m, width=900, height=560)

# =========================
# COMMENTS / FAV / CHAT پیشرفته
# =========================
def add_comment(pid:int, user_email:str, comment:str, rating:int):
    conn=get_conn(); c=conn.cursor()
    c.execute("INSERT INTO comments(property_id,user_email,comment,rating,created_at) VALUES(?,?,?,?,?)",
              (pid, user_email, comment[:1000], max(1,min(5,int(rating))), now_iso()))
    conn.commit()
    
    c.execute("SELECT owner_email FROM properties WHERE id=?", (pid,))
    owner_email = c.fetchone()[0]
    add_notification(owner_email, f"نظر جدید برای ملک شما ثبت شد! امتیاز: {rating}/5", "comment", related_id=pid)
    
    update_user_rating(owner_email)
    conn.close()

def load_comments(pid:int) -> pd.DataFrame:
    conn=get_conn(); c=conn.cursor()
    c.execute("""
        SELECT c.user_email, c.comment, c.rating, c.created_at, c.helpful, u.name 
        FROM comments c 
        JOIN users u ON c.user_email = u.email 
        WHERE c.property_id=? 
        ORDER BY c.id DESC
    """, (pid,))
    rows=c.fetchall(); conn.close()
    return pd.DataFrame(rows, columns=["user_email","comment","rating","created_at","helpful","user_name"])

def mark_comment_helpful(comment_id: int):
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE comments SET helpful = helpful + 1 WHERE id=?", (comment_id,))
    conn.commit(); conn.close()

def toggle_fav(pid:int, user_email:str, notes:str = None):
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT id, notes FROM favorites WHERE property_id=? AND user_email=?", (pid,user_email))
    r=c.fetchone()
    
    if r:
        if notes is not None:
            c.execute("UPDATE favorites SET notes=? WHERE id=?", (notes, r[0]))
        else:
            c.execute("DELETE FROM favorites WHERE id=?", (r[0],))
        conn.commit(); conn.close()
        track_user_activity(user_email, "remove_favorite", f"Removed property {pid} from favorites")
        return False
    else:
        c.execute("INSERT INTO favorites(property_id,user_email,created_at,notes) VALUES(?,?,?,?)",
                  (pid,user_email, now_iso(), notes))
        conn.commit(); conn.close()
        
        c = get_conn().cursor()
        c.execute("SELECT owner_email FROM properties WHERE id=?", (pid,))
        owner_email = c.fetchone()[0]
        add_notification(owner_email, "کاربر جدید ملک شما را به علاقه‌مندی‌ها اضافه کرد!", "favorite", related_id=pid)
        
        track_user_activity(user_email, "add_favorite", f"Added property {pid} to favorites")
        update_recommendations_for_user(user_email)
        
        return True

def list_favorites(user_email:str) -> pd.DataFrame:
    conn=get_conn(); c=conn.cursor()
    c.execute("""
        SELECT p.*, f.notes, f.created_at as fav_date 
        FROM favorites f
        JOIN properties p ON p.id=f.property_id
        WHERE f.user_email=? AND p.status='published'
        ORDER BY f.created_at DESC
    """, (user_email,))
    rows=c.fetchall()
    if not rows:
        conn.close(); return pd.DataFrame()
    cols=[d[0] for d in c.description]; conn.close()
    return pd.DataFrame(rows, columns=cols)

def send_message(pid:int, sender:str, receiver:str, body:str, message_type:str = "text"):
    conn=get_conn(); c=conn.cursor()
    c.execute("INSERT INTO messages(property_id,sender_email,receiver_email,body,created_at,message_type) VALUES(?,?,?,?,?,?)",
              (pid, sender, receiver, body[:1000], now_iso(), message_type))
    conn.commit(); conn.close()
    
    add_notification(receiver, "پیام جدید دریافت کردید!", "message", related_id=pid)
    track_user_activity(sender, "send_message", f"Sent message to {receiver} about property {pid}")

def load_chat(pid:int, a:str, b:str) -> List[Dict[str,Any]]:
    conn=get_conn(); c=conn.cursor()
    c.execute("""
        SELECT sender_email, body, created_at, message_type 
        FROM messages
        WHERE property_id=? AND (sender_email IN (?,?) AND receiver_email IN (?,?))
        ORDER BY id ASC
    """, (pid, a, b, a, b))
    rows=c.fetchall(); conn.close()
    return [{"sender":r[0], "body":r[1], "at":r[2], "type":r[3]} for r in rows]

def get_unread_message_count(user_email: str) -> int:
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE receiver_email=? AND is_read=0", (user_email,))
    count = c.fetchone()[0]; conn.close()
    return count

def mark_messages_as_read(property_id: int, user_email: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("""
        UPDATE messages SET is_read=1 
        WHERE property_id=? AND receiver_email=? AND is_read=0
    """, (property_id, user_email))
    conn.commit(); conn.close()

# =========================
# NEW FEATURES - سیستم نوتیفیکیشن پیشرفته
# =========================
def add_notification(user_email: str, message: str, notification_type: str = "info", related_id: int = None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO notifications (user_email, message, type, related_id, created_at) VALUES (?, ?, ?, ?, ?)",
              (user_email, message, notification_type, related_id, now_iso()))
    conn.commit()
    conn.close()

def get_unread_notifications_count(user_email: str) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM notifications WHERE user_email=? AND is_read=0", (user_email,))
    count = c.fetchone()[0]
    conn.close()
    return count

def get_notifications(user_email: str, limit: int = 20, unread_only: bool = False) -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    
    query = """
        SELECT id, message, type, created_at, is_read, related_id 
        FROM notifications 
        WHERE user_email=?
    """
    
    if unread_only:
        query += " AND is_read=0"
        
    query += " ORDER BY id DESC LIMIT ?"
    
    c.execute(query, (user_email, limit))
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=["id", "message", "type", "created_at", "is_read", "related_id"])

def mark_notification_as_read(notification_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read=1 WHERE id=?", (notification_id,))
    conn.commit()
    conn.close()

def mark_all_notifications_as_read(user_email: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read=1 WHERE user_email=?", (user_email,))
    conn.commit()
    conn.close()

# =========================
# NEW FEATURES - ردیابی فعالیت پیشرفته
# =========================
def track_user_activity(user_email: str, action: str, details: str = "", ip_address: str = None, user_agent: str = None):
    conn = get_conn()
    c = conn.cursor()
    
    if ip_address is None:
        ip_address = get_client_ip()
        
    if user_agent is None:
        user_agent = "Unknown"
    
    c.execute("""
        INSERT INTO user_activity (user_email, action, details, ip_address, user_agent, created_at) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_email, action, details, ip_address, user_agent, now_iso()))
    
    conn.commit()
    conn.close()

def get_user_activity(user_email: str, limit: int = 50) -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT action, details, created_at, ip_address 
        FROM user_activity 
        WHERE user_email=? 
        ORDER BY id DESC 
        LIMIT ?
    """, (user_email, limit))
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=["action", "details", "created_at", "ip_address"])

def get_system_activity(limit: int = 100) -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT u.name, ua.action, ua.details, ua.created_at, ua.ip_address
        FROM user_activity ua
        JOIN users u ON ua.user_email = u.email
        ORDER BY ua.id DESC 
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=["user", "action", "details", "created_at", "ip_address"])

# =========================
# NEW FEATURES - امتیازدهی کاربران پیشرفته
# =========================
def update_user_rating(user_email: str):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        SELECT AVG(rating) 
        FROM comments 
        WHERE property_id IN (SELECT id FROM properties WHERE owner_email = ?)
    """, (user_email,))
    
    new_rating = c.fetchone()[0] or 5.0
    
    c.execute("""
        SELECT COUNT(*) 
        FROM transactions 
        WHERE seller_email = ? AND status = 'completed'
    """, (user_email,))
    
    successful_transactions = c.fetchone()[0] or 0
    transaction_bonus = min(successful_transactions * 0.1, 0.5)
    
    final_rating = min(5.0, new_rating + transaction_bonus)
    
    c.execute("UPDATE users SET rating=? WHERE email=?", (final_rating, user_email))
    conn.commit()
    conn.close()
    return final_rating

def calculate_user_rating(user_email: str) -> float:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT rating FROM users WHERE email=?", (user_email,))
    rating = c.fetchone()[0] or 5.0
    conn.close()
    return round(rating, 1)

def verify_user(user_email: str, verified: bool = True):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE users SET verified=? WHERE email=?", (1 if verified else 0, user_email))
    conn.commit()
    conn.close()
    
    if verified:
        add_notification(user_email, "حساب کاربری شما تأیید شد!", "success")
    else:
        add_notification(user_email, "تأیید حساب کاربری شما لغو شد.", "warning")

# =========================
# NEW FEATURES - سیستم گزارش‌گیری پیشرفته
# =========================
def generate_property_report(property_id: int) -> Dict[str, Any]:
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM property_views WHERE property_id = ?", (property_id,))
    views = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM favorites WHERE property_id = ?", (property_id,))
    favorites = c.fetchone()[0]
    
    c.execute("SELECT AVG(rating) FROM comments WHERE property_id = ?", (property_id,))
    avg_rating = round(c.fetchone()[0] or 0, 1)
    
    c.execute("SELECT COUNT(*) FROM messages WHERE property_id = ?", (property_id,))
    messages = c.fetchone()[0]
    
    conn.close()
    
    return {
        "total_views": views,
        "total_favorites": favorites,
        "average_rating": avg_rating,
        "total_messages": messages,
        "performance_score": calculate_performance_score(views, favorites, avg_rating, messages)
    }

def calculate_performance_score(views: int, favorites: int, rating: float, messages: int) -> float:
    view_score = min(views / 10, 40)
    favorite_score = min(favorites * 8, 30)
    rating_score = rating * 4
    message_score = min(messages * 2, 10)
    
    return round(view_score + favorite_score + rating_score + message_score, 1)

def generate_user_report(user_email: str) -> Dict[str, Any]:
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM properties WHERE owner_email = ? AND status = 'published'", (user_email,))
    published_properties = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM transactions WHERE seller_email = ? AND status = 'completed'", (user_email,))
    sold_properties = c.fetchone()[0]
    
    c.execute("SELECT rating FROM users WHERE email = ?", (user_email,))
    rating = c.fetchone()[0] or 5.0
    
    c.execute("SELECT SUM(views) FROM properties WHERE owner_email = ?", (user_email,))
    total_views = c.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "published_properties": published_properties,
        "sold_properties": sold_properties,
        "rating": rating,
        "total_views": total_views,
        "success_rate": (sold_properties / published_properties * 100) if published_properties > 0 else 0
    }

# =========================
# NEW FEATURES - سیستم پیشنهادات هوشمند پیشرفته
# =========================
def get_smart_recommendations(user_email: str, limit: int = 5) -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        SELECT p.*, ar.score, ar.reason 
        FROM ai_recommendations ar
        JOIN properties p ON ar.property_id = p.id
        WHERE ar.user_email = ? AND p.status = 'published'
        ORDER BY ar.score DESC
        LIMIT ?
    """, (user_email, limit))
    
    rows = c.fetchall()
    
    if rows:
        cols = [d[0] for d in c.description]
        conn.close()
        return pd.DataFrame(rows, columns=cols)
    
    c.execute("""
        SELECT p.* FROM properties p
        WHERE p.property_type IN (
            SELECT DISTINCT property_type FROM favorites f
            JOIN properties p ON f.property_id = p.id
            WHERE f.user_email = ?
        ) AND p.status = 'published' AND p.owner_email != ?
        ORDER BY p.views DESC, p.created_at DESC
        LIMIT ?
    """, (user_email, user_email, limit))
    
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()

def update_recommendations_for_user(user_email: str):
    """تابع به‌روزرسانی پیشنهادات برای کاربر - از PropertyRecommender استفاده می‌کند"""
    simple_property_recommender.update_recommendations_for_user(user_email)

def update_recommendations_for_new_property(property_id: int):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT property_type, city, price, area FROM properties WHERE id=?", (property_id,))
    prop_info = c.fetchone()
    
    if not prop_info:
        conn.close()
        return
        
    prop_type, city, price, area = prop_info
    
    c.execute("""
        SELECT DISTINCT user_email 
        FROM favorites f
        JOIN properties p ON f.property_id = p.id
        WHERE p.property_type = ? OR p.city = ?
        OR (p.price BETWEEN ? * 0.7 AND ? * 1.3)
        OR (p.area BETWEEN ? * 0.7 AND ? * 1.3)
    """, (prop_type, city, price, price, area, area))
    
    users = [row[0] for row in c.fetchall()]
    conn.close()
    
    for user_email in users:
        update_recommendations_for_user(user_email)

# =========================
# NEW FEATURES - سیستم اشتراک‌گذاری پیشرفته
# =========================
def generate_share_links(property_id: int, base_url: str) -> Dict[str, str]:
    url = f"{base_url.rstrip('/')}/?pg=view&pid={property_id}"
    title = "ملک پیشنهادی در سامانه املاک جرقویه"
    
    return {
        "whatsapp": f"https://wa.me/?text={title}: {url}",
        "telegram": f"https://t.me/share/url?url={url}&text={title}",
        "email": f"mailto:?subject={title}&body=این ملک را بررسی کنید: {url}",
        "twitter": f"https://twitter.com/intent/tweet?text={title}&url={url}",
        "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={url}",
        "direct": url,
        "qr_code": f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={url}"
    }

def generate_embed_code(property_id: int, base_url: str, width: int = 300, height: int = 250) -> str:
    url = f"{base_url.rstrip('/')}/?pg=view&pid={property_id}"
    embed_url = f"{base_url.rstrip('/')}/embed/property/{property_id}"
    
    return f"""
    <iframe src="{embed_url}" width="{width}" height="{height}" frameborder="0" style="border:1px solid #ddd;">
        <a href="{url}">مشاهده ملک در سامانه املاک جرقویه</a>
    </iframe>
    """

# =========================
# NEW FEATURES - سیستم بک‌آپ پیشرفته
# =========================
def create_backup() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{BACKUP_DIR}/backup_{timestamp}.db"
    
    with open(DB_NAME, 'rb') as original:
        with open(backup_file, 'wb') as backup:
            backup.write(original.read())
    
    try:
        import gzip
        with open(backup_file, 'rb') as f_in:
            with gzip.open(f"{backup_file}.gz", 'wb') as f_out:
                f_out.writelines(f_in)
        os.remove(backup_file)
        backup_file = f"{backup_file}.gz"
    except:
        pass
    
    track_user_activity("system", "backup_created", f"Backup created: {backup_file}")
    
    return backup_file

def restore_backup(backup_file: str) -> bool:
    try:
        if backup_file.endswith('.gz'):
            import gzip
            with gzip.open(backup_file, 'rb') as f_in:
                with open(DB_NAME, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            with open(backup_file, 'rb') as backup:
                with open(DB_NAME, 'wb') as original:
                    original.write(backup.read())
        
        track_user_activity("system", "backup_restored", f"Backup restored: {backup_file}")
        return True
    except Exception as e:
        track_user_activity("system", "backup_restore_failed", f"Failed to restore backup: {str(e)}")
        return False

def auto_backup():
    try:
        if datetime.now().hour == 2:
            create_backup()
    except:
        pass

# =========================
# NEW FEATURES - سیستم عضویت و اطلاع‌رسانی پیشرفته
# =========================
def create_search_subscription(user_email: str, filters: Dict[str, Any], frequency: str = "instant"):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT id FROM subscriptions WHERE user_email=? AND search_filters=?", 
              (user_email, json.dumps(filters)))
    
    if c.fetchone():
        c.execute("UPDATE subscriptions SET is_active=1, frequency=?, updated_at=? WHERE user_email=? AND search_filters=?",
                  (frequency, now_iso(), user_email, json.dumps(filters)))
    else:
        c.execute("INSERT INTO subscriptions (user_email, search_filters, frequency, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                  (user_email, json.dumps(filters), frequency, now_iso(), now_iso()))
    
    conn.commit()
    conn.close()
    
    add_notification(user_email, "اشتراک جستجوی جدید شما ایجاد شد! هنگام پیدا شدن ملک جدید به شما اطلاع می‌دهیم.", "subscription")

def check_subscriptions(new_property: Dict[str, Any]):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT user_email, search_filters FROM subscriptions WHERE is_active=1")
    subscriptions = c.fetchall()
    conn.close()
    
    for user_email, filters_json in subscriptions:
        filters = json.loads(filters_json)
        matches = True
        
        if filters.get("city") and new_property["city"] not in filters["city"]:
            matches = False
        
        if filters.get("property_type") and new_property["property_type"] not in filters["property_type"]:
            matches = False
        
        if matches and filters.get("min_price") and new_property["price"] < filters["min_price"]:
            matches = False
            
        if matches and filters.get("max_price") and new_property["price"] > filters["max_price"]:
            matches = False
            
        if matches and filters.get("min_area") and new_property["area"] < filters["min_area"]:
            matches = False
            
        if matches and filters.get("max_area") and new_property["area"] > filters["max_area"]:
            matches = False
        
        if matches:
            add_notification(user_email, f"ملک جدیدی مطابق با جستجوی شما پیدا شد: {new_property['title']}", "match", related_id=new_property["id"])

def send_subscription_digests():
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        SELECT DISTINCT user_email, frequency 
        FROM subscriptions 
        WHERE is_active=1 AND frequency != 'instant'
    """)
    
    users = c.fetchall()
    
    for user_email, frequency in users:
        c.execute("SELECT MAX(updated_at) FROM subscriptions WHERE user_email=?", (user_email,))
        last_update = c.fetchone()[0]
        
        if not last_update:
            continue
            
        last_update_date = datetime.fromisoformat(last_update)
        now = datetime.now()
        
        send_digest = False
        if frequency == "daily" and (now - last_update_date).days >= 1:
            send_digest = True
        elif frequency == "weekly" and (now - last_update_date).days >= 7:
            send_digest = True
            
        if send_digest:
            create_and_send_digest(user_email, frequency)
            c.execute("UPDATE subscriptions SET updated_at=? WHERE user_email=?", (now_iso(), user_email))
    
    conn.commit()
    conn.close()

def create_and_send_digest(user_email: str, frequency: str):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT MAX(updated_at) FROM subscriptions WHERE user_email=?", (user_email,))
    last_update = c.fetchone()[0]
    
    if not last_update:
        conn.close()
        return
        
    c.execute("""
        SELECT p.* FROM properties p
        WHERE p.status='published' AND p.created_at > ?
        ORDER BY p.created_at DESC
        LIMIT 10
    """, (last_update,))
    
    new_properties = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    
    if not new_properties:
        return

# =========================
# NEW FEATURES - افزایش بازدید ملک پیشرفته
# =========================
def increment_property_views(property_id: int, user_email: str = None):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("UPDATE properties SET views = views + 1 WHERE id = ?", (property_id,))
    
    ip_address = get_client_ip()
    user_agent = "Unknown"
    
    c.execute("INSERT INTO property_views (property_id, user_agent, ip_address, user_email, created_at) VALUES (?, ?, ?, ?, ?)",
              (property_id, user_agent, ip_address, user_email, now_iso()))
    
    conn.commit()
    conn.close()

def get_property_analytics(property_id: int, period: str = "7d") -> Dict[str, Any]:
    conn = get_conn()
    c = conn.cursor()
    
    now = datetime.now()
    if period == "1d":
        start_date = now - timedelta(days=1)
    elif period == "7d":
        start_date = now - timedelta(days=7)
    elif period == "30d":
        start_date = now - timedelta(days=30)
    else:
        start_date = now - timedelta(days=7)
    
    c.execute("SELECT COUNT(*) FROM property_views WHERE property_id=? AND created_at >= ?", 
              (property_id, start_date.isoformat()))
    views_in_period = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM property_views WHERE property_id=?", (property_id,))
    total_views = c.fetchone()[0]
    
    c.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as count 
        FROM property_views 
        WHERE property_id=? AND created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY day
    """, (property_id, start_date.isoformat()))
    
    daily_views = c.fetchall()
    
    conn.close()
    
    return {
        "views_in_period": views_in_period,
        "total_views": total_views,
        "daily_views": daily_views
    }

# =========================
# NEW FEATURES - سیستم گزارش تخلف
# =========================
def report_item(reporter_email: str, item_type: str, item_id: int, reason: str):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO reports (reporter_email, reported_item_type, reported_item_id, reason, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (reporter_email, item_type, item_id, reason, now_iso()))
    
    conn.commit()
    conn.close()
    
    notify_admins(f"گزارش تخلف جدید برای {item_type} با شناسه {item_id}", "warning")

def get_pending_reports() -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        SELECT r.id, r.reporter_email, r.reported_item_type, r.reported_item_id, r.reason, r.created_at, u.name as reporter_name
        FROM reports r
        JOIN users u ON r.reporter_email = u.email
        WHERE r.status = 'pending'
        ORDER BY r.created_at DESC
    """)
    
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=["id", "reporter_email", "reported_item_type", "reported_item_id", "reason", "created_at", "reporter_name"])

def resolve_report(report_id: int, resolved_by: str, resolution: str):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        UPDATE reports 
        SET status='resolved', resolved_at=?, resolved_by=?
        WHERE id=?
    """, (now_iso(), resolved_by, report_id))
    
    conn.commit()
    conn.close()

# =========================
# NEW FEATURES - سیستم معاملات
# =========================
def create_transaction(property_id: int, buyer_email: str, seller_email: str, price: int):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO transactions (property_id, buyer_email, seller_email, price, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
    """, (property_id, buyer_email, seller_email, price, now_iso()))
    
    transaction_id = c.lastrowid
    
    conn.commit()
    conn.close()
    
    add_notification(seller_email, f"درخواست خرید جدید برای ملک شما دریافت شد!", "success", related_id=property_id)
    
    return transaction_id

def update_transaction_status(transaction_id: int, status: str):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT property_id, buyer_email, seller_email FROM transactions WHERE id=?", (transaction_id,))
    transaction = c.fetchone()
    
    if not transaction:
        conn.close()
        return False
        
    property_id, buyer_email, seller_email = transaction
    
    c.execute("UPDATE transactions SET status=?, transaction_date=? WHERE id=?", 
              (status, now_iso() if status == 'completed' else None, transaction_id))
    
    if status == 'completed':
        c.execute("UPDATE properties SET status='sold' WHERE id=?", (property_id,))
        
        add_notification(buyer_email, "خرید شما با موفقیت تکمیل شد!", "success", related_id=property_id)
        add_notification(seller_email, "فروش ملک شما با موفقیت تکمیل شد!", "success", related_id=property_id)
        
        update_user_rating(seller_email)
    
    conn.commit()
    conn.close()
    return True

def get_user_transactions(user_email: str) -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
        SELECT t.*, p.title as property_title 
        FROM transactions t
        JOIN properties p ON t.property_id = p.id
        WHERE t.buyer_email = ? OR t.seller_email = ?
        ORDER BY t.created_at DESC
    """, (user_email, user_email))
    
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=[d[0] for d in c.description])

# =========================
# NEW FEATURES - سیستم اطلاع‌رسانی به ادمین‌ها
# =========================
def notify_admins(message: str, notification_type: str = "info"):
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("SELECT email FROM users WHERE role='admin'")
    admins = [row[0] for row in c.fetchall()]
    
    for admin_email in admins:
        add_notification(admin_email, message, notification_type)
    
    conn.close()

# =========================
# AI-ENHANCED UI COMPONENTS - کامپوننت‌های هوشمند UI
# =========================
def show_ai_market_analysis():
    """نمایش تحلیل هوشمند بازار"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 📊 تحلیل هوشمند بازار املاک")
    
    try:
        conn = get_conn()
        properties_df = pd.read_sql("SELECT * FROM properties WHERE status='published'", conn)
        conn.close()
        
        if properties_df.empty:
            st.info("📈 داده کافی برای تحلیل بازار موجود نیست")
            return
        
        # تحلیل بازار
        trends = market_analytics.get_market_trends(properties_df)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🏘️ تعداد املاک", trends['total_properties'])
        col2.metric("💰 میانگین قیمت", f"{trends['avg_price']:,.0f} تومان")
        col3.metric("📐 قیمت هر متر", f"{trends['avg_price_per_meter']:,.0f} تومان")
        
        # نمودار توزیع قیمت
        fig_price = px.histogram(properties_df, x='price', 
                               title='توزیع قیمت املاک',
                               labels={'price': 'قیمت (تومان)'})
        st.plotly_chart(fig_price, use_container_width=True)
        
        # نمودار محبوبیت شهرها
        city_counts = properties_df['city'].value_counts()
        fig_cities = px.pie(values=city_counts.values, names=city_counts.index,
                          title='توزیع املاک بر اساس شهر')
        st.plotly_chart(fig_cities, use_container_width=True)
        
        # پیش‌بینی روند بازار
        st.markdown("#### 📈 پیش‌بینی روند بازار")
        if len(properties_df) > 10:
            st.success("✅ بازار در حال رشد - فرصت‌های سرمایه‌گذاری مناسب")
        else:
            st.info("ℹ️ داده‌های بیشتری برای پیش‌بینی دقیق‌تر نیاز است")
            
    except Exception as e:
        st.error(f"❌ خطا در تحلیل بازار: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_smart_price_analysis(property_data: Dict[str, Any]):
    """نمایش تحلیل قیمت هوشمند برای یک ملک"""
    st.markdown("<div class='traditional-tab'>", unsafe_allow_html=True)
    st.markdown("#### 💡 تحلیل قیمت هوشمند")
    
    try:
        conn = get_conn()
        market_df = pd.read_sql("SELECT * FROM properties WHERE status='published'", conn)
        conn.close()
        
        if market_df.empty:
            st.info("📊 تحلیل قیمت در دسترس نیست")
            return
        
        analysis = price_advisor.analyze_property_value(property_data, market_df)
        
        col1, col2 = st.columns(2)
        col1.metric("💰 قیمت پیشنهادی", f"{analysis['predicted_price']:,} تومان")
        col2.metric("🎯 اختلاف قیمت", f"{analysis['price_difference']:,} تومان", 
                   delta=f"{analysis['price_difference']:,}")
        
        st.info(f"**توصیه:** {analysis['recommendation']}")
        st.progress(min(int(analysis['confidence_score'] * 100), 100))
        st.caption(f"امتیاز اطمینان تحلیل: {analysis['confidence_score']:.0%}")
        
    except Exception as e:
        st.error(f"خطا در تحلیل قیمت: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_ai_recommendations(user_email: str):
    """نمایش پیشنهادات هوشمند"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 🧠 پیشنهادات هوشمند برای شما")
    
    try:
        recommendations = property_recommender.get_personalized_recommendations(user_email, 4)
        
        if not recommendations:
            st.info("🤔 برای دریافت پیشنهادات شخصی‌سازی شده، چند ملک را به علاقه‌مندی‌ها اضافه کنید")
            return
        
        for prop_id, score, reason in recommendations:
            conn = get_conn()
            c = conn.cursor()
            c.execute("SELECT * FROM properties WHERE id=?", (prop_id,))
            row = c.fetchone()
            conn.close()
            
            if row:
                cols = [d[0] for d in c.description]
                prop_data = dict(zip(cols, row))
                
                st.markdown(f"**🎯 دلیل پیشنهاد:** {reason}")
                st.markdown(f"**⭐ امتیاز تطابق:** {score:.0%}")
                property_card(pd.Series(prop_data), st.session_state.get("user"))
                st.markdown("---")
                
    except Exception as e:
        st.error(f"❌ خطا در دریافت پیشنهادات: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ENHANCED ADMIN PANEL - پنل مدیر بهبود یافته
# =========================
def admin_panel(user: Dict[str, Any]):
    """پنل مدیر با قابلیت‌های پیشرفته"""
    if user["email"] != ADMIN_EMAIL:
        st.error("⛔ دسترسی غیرمجاز - فقط مدیر اصلی سیستم مجاز است")
        return
    
    st.markdown("<div class='traditional-header'>", unsafe_allow_html=True)
    st.subheader("👑 پنل مدیر سیستم - نسخه پیشرفته")
    st.markdown("</div>", unsafe_allow_html=True)

    # تب‌های پیشرفته مدیریت
    tab1, tab2, tab3, tab4 = st.tabs(["📊 آمار پیشرفته", "🤖 هوش مصنوعی", "👥 مدیریت کاربران", "⚙️ تنظیمات سیستم"])

    with tab1:
        show_advanced_analytics()

    with tab2:
        show_ai_management()

    with tab3:
        show_user_management()

    with tab4:
        show_system_settings()

def show_advanced_analytics():
    """نمایش آمار و تحلیل‌های پیشرفته"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 📈 آمار و تحلیل‌های پیشرفته")
    
    conn = get_conn()
    
    # آمار کلی
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    users_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM properties WHERE status='published'")
    active_props = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM properties WHERE status='sold'")
    sold_props = c.fetchone()[0]
    
    c.execute("SELECT SUM(views) FROM properties")
    total_views = c.fetchone()[0] or 0
    
    # تحلیل‌های پیشرفته
    properties_df = pd.read_sql("""
        SELECT * FROM properties WHERE status='published'
    """, conn)
    
    user_activity_df = pd.read_sql("""
        SELECT * FROM user_activity ORDER BY id DESC LIMIT 1000
    """, conn)
    
    conn.close()
    
    # نمایش آمار
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 کاربران", users_count)
    col2.metric("🏠 املاک فعال", active_props)
    col3.metric("💰 املاک فروخته", sold_props)
    col4.metric("👀 بازدید کل", total_views)
    
    if not properties_df.empty:
        # تحلیل بازار
        st.markdown("#### 📊 تحلیل بازار")
        trends = market_analytics.get_market_trends(properties_df)
        
        col1, col2 = st.columns(2)
        col1.metric("💰 میانگین قیمت", f"{trends['avg_price']:,.0f} تومان")
        col2.metric("📐 قیمت هر متر", f"{trends['avg_price_per_meter']:,.0f} تومان")
        
        # نمودار فعالیت کاربران
        if not user_activity_df.empty:
            st.markdown("#### 📈 فعالیت کاربران")
            activity_counts = user_activity_df['action'].value_counts()
            fig_activity = px.bar(x=activity_counts.index, y=activity_counts.values,
                                title='توزیع فعالیت کاربران')
            st.plotly_chart(fig_activity, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_ai_management():
    """مدیریت سیستم‌های هوشمند"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 🤖 مدیریت سیستم‌های هوشمند")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 راه‌اندازی مجدد هوش مصنوعی", use_container_width=True):
            with st.spinner("در حال راه‌اندازی سیستم‌های هوشمند..."):
                if initialize_ai_systems():
                    st.success("✅ سیستم‌های هوشمند با موفقیت راه‌اندازی شدند")
                else:
                    st.error("❌ خطا در راه‌اندازی سیستم‌های هوشمند")
    
    with col2:
        if st.button("📊 به‌روزرسانی تحلیل بازار", use_container_width=True):
            with st.spinner("در حال به‌روزرسانی تحلیل‌ها..."):
                try:
                    conn = get_conn()
                    properties_df = pd.read_sql("SELECT * FROM properties WHERE status='published'", conn)
                    conn.close()
                    
                    market_analytics.train_price_model(properties_df)
                    st.success("✅ تحلیل بازار با موفقیت به‌روزرسانی شد")
                except Exception as e:
                    st.error(f"❌ خطا در به‌روزرسانی: {e}")
    
    # وضعیت سیستم‌های هوشمند
    st.markdown("#### 🟢 وضعیت سیستم‌ها")
    
    try:
        conn = get_conn()
        properties_count = pd.read_sql("SELECT COUNT(*) as count FROM properties WHERE status='published'", conn)['count'][0]
        users_count = pd.read_sql("SELECT COUNT(*) as count FROM users", conn)['count'][0]
        conn.close()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 داده‌های آموزشی", f"{properties_count} ملک")
        col2.metric("👥 کاربران فعال", users_count)
        col3.metric("🤖 وضعیت هوش مصنوعی", "فعال" if properties_count > 5 else "نیاز به داده بیشتر")
        
        if properties_count < 10:
            st.warning("⚠️ برای عملکرد بهتر هوش مصنوعی، داده‌های بیشتری نیاز است")
            
    except Exception as e:
        st.error(f"❌ خطا در بررسی وضعیت: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_user_management():
    """مدیریت کاربران"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 👥 مدیریت کاربران")
    
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT name,email,role,phone,rating,verified FROM users ORDER BY id DESC")
    users=c.fetchall()
    
    for name,email,role,phone,rating,verified in users:
        col1,col2,col3,col4,col5,col6,col7 = st.columns([2,2,1,1,1,1,1])
        col1.write(name); col2.write(email); col3.write(role); col4.write(phone or "—"); col5.write(f"⭐ {rating}")
        col6.write("✅" if verified else "❌")
        
        # فقط مدیر اصلی می‌تواند نقش‌ها را تغییر دهد
        if email != ADMIN_EMAIL:
            if col7.button(f"ارتقا", key=f"mk_{email}", use_container_width=True):
                cx=get_conn(); cc=cx.cursor(); cc.execute("UPDATE users SET role='agent' WHERE email=?", (email,)); cx.commit(); cx.close(); st.success("✅ انجام شد"); st.rerun()
    
    conn.close()
    st.markdown("</div>", unsafe_allow_html=True)

def show_system_settings():
    """تنظیمات سیستم"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ تنظیمات سیستم")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📦 ایجاد پشتیبان جدید", use_container_width=True):
            backup_file = create_backup()
            st.success(f"✅ پشتیبان با موفقیت ایجاد شد: {backup_file}")
    
    with col2:
        backup_files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.db') or f.endswith('.gz')]
        if backup_files:
            selected_backup = st.selectbox("انتخاب پشتیبان برای بازیابی", backup_files)
            if st.button("🔄 بازیابی پشتیبان", use_container_width=True):
                if restore_backup(f"{BACKUP_DIR}/{selected_backup}"):
                    st.success("✅ پشتیبان با موفقیت بازیابی شد. لطفاً صفحه را refresh کنید.")
                else:
                    st.error("❌ خطا در بازیابی پشتیبان")
        else:
            st.info("ℹ️ هیچ پشتیبانی موجود نیست")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ENHANCED PUBLIC PANEL - پنل عمومی بهبود یافته
# =========================
def public_panel(user: Dict[str, Any]):
    """پنل عمومی با قابلیت‌های هوشمند"""
    st.markdown("<div class='traditional-header'>", unsafe_allow_html=True)
    st.subheader(f"🌺 خوش آمدی {user['name']}")
    
    # نمایش وضعیت ویژه برای مدیر
    if user["email"] == ADMIN_EMAIL:
        st.markdown("<span class='badge-premium'>⭐ مدیر سیستم</span>", unsafe_allow_html=True)
    elif user.get('verified'):
        st.markdown("<span class='badge-verified'>✅ حساب تأیید شده</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # تب‌های پیشرفته
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 خانه", "📊 تحلیل بازار", "🤖 پیشنهادات", "👤 پروفایل"])

    with tab1:
        show_main_dashboard(user)

    with tab2:
        show_ai_market_analysis()

    with tab3:
        show_ai_recommendations(user["email"])

    with tab4:
        show_user_profile(user)

def show_main_dashboard(user: Dict[str, Any]):
    """نمایش داشبورد اصلی"""
    # اطلاع‌رسانی‌ها
    notification_count = get_unread_notifications_count(user["email"])
    unread_messages = get_unread_message_count(user["email"])
    
    if notification_count > 0 or unread_messages > 0:
        st.markdown("<div class='persian-pattern' style='text-align: center;'>", unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].markdown(f"### 🔔 {notification_count} نوتیفیکیشن")
        cols[1].markdown(f"### 📧 {unread_messages} پیام جدید")
        
        if st.button("مشاهده همه اعلان‌ها", use_container_width=True):
            st.session_state["show_notifications"] = True
        st.markdown("</div>", unsafe_allow_html=True)

    # جستجو و فیلترها
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 🔍 جستجوی هوشمند املاک")
    filt = property_filters()
    df = list_properties_df_cached(json.dumps(filt, ensure_ascii=False))
    
    # نمایش نقشه
    show_map(df)

    # نمایش نتایج
    st.markdown("### 📋 نتایج جستجو")
    df_page = paginator(df, page_size=8, key="pg_results")
    for _, row in df_page.iterrows():
        property_card(row, user)
        
        # تحلیل قیمت برای هر ملک
        if st.session_state.get("user"):
            with st.expander(f"💡 تحلیل قیمت برای {row['title']}"):
                show_smart_price_analysis(dict(row))
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_user_profile(user: Dict[str, Any]):
    """نمایش پروفایل کاربر"""
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 👤 پروفایل کاربری")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_image = st.file_uploader("🖼️ تغییر تصویر پروفایل", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            try:
                image_bytes = uploaded_image.read()
                conn = get_conn(); c = conn.cursor()
                c.execute("UPDATE users SET profile_image=? WHERE email=?", (image_bytes, user["email"]))
                conn.commit(); conn.close()
                st.success("✅ تصویر پروفایل به روز شد.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ خطا در آپلود تصویر: {e}")
    
    with col2:
        st.markdown(f"**نام:** {user['name']}")
        st.markdown(f"**ایمیل:** {user['email']}")
        if user.get('phone'):
            st.markdown(f"**تلفن:** {user['phone']}")
        if user.get('bio'):
            st.markdown(f"**بیوگرافی:** {user['bio']}")
        st.markdown(f"**امتیاز:** ⭐ {calculate_user_rating(user['email'])}")
        
        report = generate_user_report(user["email"])
        st.markdown(f"**املاک منتشر شده:** {report['published_properties']}")
        st.markdown(f"**املاک فروخته شده:** {report['sold_properties']}")
        if report['published_properties'] > 0:
            st.markdown(f"**نرخ موفقیت:** {report['success_rate']:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # فرم ثبت ملک جدید
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 🏡 ثبت ملک جدید")
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.expander("📋 فرم ثبت ملک", expanded=False):
        st.markdown("<div class='islamic-pattern'>", unsafe_allow_html=True)
        
        title = st.text_input("🏷️ عنوان ملک")
        price = st.number_input("💰 قیمت (تومان)", min_value=0, step=100000)
        area  = st.number_input("📏 متراژ (متر)", min_value=0, step=1)
        city  = st.text_input("🏙️ شهر")
        ptype = st.selectbox("🏠 نوع ملک", ["آپارتمان","ویلایی","مغازه","زمین","دفتر"])
        
        c1,c2 = st.columns(2)
        lat = c1.number_input("📍 عرض جغرافیایی", format="%.6f")
        lon = c2.number_input("📍 طول جغرافیایی", format="%.6f")
        
        address = st.text_input("🏠 آدرس")
        rooms = st.number_input("🚪 تعداد اتاق", min_value=0, step=1)
        age   = st.number_input("🏚️ سن بنا", min_value=0, step=1)
        facilities = st.text_area("⭐ امکانات (با کاما جدا کنید)")
        desc  = st.text_area("📝 توضیحات")
        video = st.text_input("🎥 لینک تور ویدئویی/۳۶۰ (اختیاری)")
        
        featured = st.checkbox("⭐ ملک ویژه", help="ملک ویژه در صدر نتایج جستجو نمایش داده می‌شود")
        
        uploaded = st.file_uploader(f"🖼️ تصاویر (حداکثر {MAX_UPLOAD_IMAGES} عدد)", type=["png","jpg","jpeg", "webp"], accept_multiple_files=True)
        
        if st.button("✅ ثبت ملک", use_container_width=True):
            if not title or price<=0 or not city or not uploaded:
                st.error("❌ موارد ضروری: عنوان، قیمت، شهر، حداقل یک تصویر.")
            else:
                try:
                    imgs_bytes = image_files_to_bytes(uploaded)
                    if not imgs_bytes:
                        st.error("❌ هیچ تصویر معتبر بارگذاری نشده.")
                        return
                    
                    pid = add_property_row({
                        "title": title, "price": int(price), "area": int(area), "city": city, "property_type": ptype,
                        "latitude": float(lat or 0), "longitude": float(lon or 0), "address": address,
                        "owner_email": user["email"], "description": desc, "rooms": int(rooms or 0),
                        "building_age": int(age or 0), "facilities": facilities, "video_url": video,
                        "featured": featured
                    }, images=imgs_bytes, publish=True)
                    
                    st.success(f"✅ ملک شما با موفقیت ثبت شد! شناسه ملک: {pid}")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ خطا در ثبت ملک: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# UI: STYLE - طراحی سنتی و ایرانی پیشرفته
# =========================
def custom_style():
    st.markdown("""
    <style>
      :root {
        --prim: #8B3A3A;      /* قرمز ایرانی */
        --prim-dark: #6f2e2e;
        --gold: #C5A572;      /* طلایی */
        --cream: #FBF5E6;     /* کرم */
        --turquoise: #40E0D0; /* فیروزه‌ای */
        --ink: #2e2e2e;
        --pattern-bg: #f8f3e6;
      }
      
      html, body, [class*="css"] { 
        font-family: Vazirmatn, Tahoma, sans-serif; 
        background: var(--cream); 
        color: var(--ink); 
      }
      
      .stApp { 
        background: linear-gradient(180deg, #fffaf1, #f9f1df 180px, #fffaf1);
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M50 50L0 25L50 0L100 25L50 50Z' fill='%23c5a57222'/%3E%3C/svg%3E");
        background-size: 300px;
      }
      
      .stButton > button { 
        background: var(--prim); 
        color: #fff; 
        border-radius: 12px; 
        padding: 12px 24px; 
        font-weight: 700;
        border: 2px solid var(--gold);
        box-shadow: 0 4px 8px rgba(139, 58, 58, 0.2);
        transition: all 0.3s ease;
      }
      
      .stButton > button:hover {
        background: var(--prim-dark);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(139, 58, 58, 0.3);
      }
      
      .stTextInput > div > input, 
      .stNumberInput > div > div > input, 
      textarea, 
      select { 
        border: 2px solid var(--gold) !important; 
        border-radius: 12px !important; 
        background: #fffaf6 !important; 
        box-shadow: inset 0 2px 4px rgba(197, 165, 114, 0.1);
      }
      
      .card { 
        background: #fff; 
        border: 2px solid #eadfc7; 
        border-radius: 20px; 
        padding: 20px; 
        margin: 15px 0; 
        box-shadow: 0 8px 30px rgba(197, 165, 114, 0.15);
        position: relative;
        overflow: hidden;
      }
      
      .card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--prim), var(--gold));
        border-radius: 10px 10px 0 0;
      }
      
      .pill { 
        display: inline-block;
        background: linear-gradient(145deg, var(--cream), #fff);
        border: 1px solid var(--gold); 
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px 8px;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(197, 165, 114, 0.1);
      }
      
      .traditional-header {
        background: linear-gradient(135deg, var(--prim), var(--prim-dark));
        color: white;
        padding: 20px;
        border-radius: 0 0 20px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(139, 58, 58, 0.3);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
      }
      
      .traditional-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.1;
      }
      
      .persian-pattern {
        background-color: var(--pattern-bg);
        background-image: url("data:image/svg+xml,%3Csvg width='52' height='26' viewBox='0 0 52 26' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c5a572' fill-opacity='0.1'%3E%3Cpath d='M10 10c0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6h2c0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4v2c-3.314 0-6-2.686-6-6 0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6zm25.464-1.95l8.486 8.486-1.414 1.414-8.486-8.486 1.414-1.414z' /%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        padding: 20px;
        border-radius: 15px;
        border: 2px solid rgba(197, 165, 114, 0.3);
        margin: 10px 0;
      }
      
      .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f3e6, #f1e8d5);
        border-right: 3px solid var(--gold);
      }
      
      .traditional-tab {
        background: linear-gradient(145deg, #fff, var(--cream));
        border: 2px solid var(--gold);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(197, 165, 114, 0.1);
      }
      
      .iranian-border {
        border: 3px solid var(--gold);
        border-radius: 20px;
        padding: 20px;
        background: #fffaf6;
        box-shadow: 0 8px 25px rgba(197, 165, 114, 0.15);
        position: relative;
        overflow: hidden;
      }
      
      .iranian-border::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, var(--prim), var(--gold), var(--turquoise));
      }
      
      .islamic-pattern {
        background: 
          radial-gradient(circle at 10% 20%, rgba(197, 165, 114, 0.1) 0%, transparent 20%),
          radial-gradient(circle at 90% 80%, rgba(64, 224, 208, 0.1) 0%, transparent 20%),
          radial-gradient(circle at 50% 50%, rgba(139, 58, 58, 0.1) 0%, transparent 30%);
        background-color: #fdf8f0;
        padding: 25px;
        border-radius: 20px;
        border: 2px dashed var(--gold);
      }
      
      .yellow-line {
        height: 5px;
        background: linear-gradient(90deg, #ffd700, #ffed4e, #ffd700);
        margin: 0 0 20px 0;
        border-radius: 3px;
        box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);
      }
      
      .badge-premium {
        background: linear-gradient(145deg, #ffd700, #ffed4e);
        color: #8B3A3A;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 0 5px;
      }
      
      .badge-verified {
        background: linear-gradient(145deg, #40E0D0, #20B2AA);
        color: white;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 0 5px;
      }
      
      .animation-pulse {
        animation: pulse 2s infinite;
      }
      
      @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
      }
      
      .featured-card {
        border: 3px solid #ffd700 !important;
        box-shadow: 0 8px 35px rgba(255, 215, 0, 0.3) !important;
      }
      
      .featured-card::before {
        background: linear-gradient(90deg, #ffd700, #ffed4e, #ffd700) !important;
      }
      
      /* بهبود استایل برای موبایل */
      @media (max-width: 768px){
        .stButton>button{ 
            width:100%; 
            font-size: 14px !important;
            padding: 10px 16px !important;
        }
        
        .card{ 
            padding:12px !important; 
            margin: 10px 0 !important;
        }
        
        /* بهبود فونت‌ها برای موبایل */
        .card h4, .card h3, .card h2 {
            font-size: 16px !important;
            color: #8B3A3A !important;  /* رنگ قرمز برای عناوین */
        }
        
        .card p, .card div {
            font-size: 14px !important;
            color: #2e2e2e !important;  /* رنگ مشکی برای متن‌ها */
            line-height: 1.5 !important;
        }
        
        /* بهبود badge ها برای موبایل */
        .pill {
            font-size: 12px !important;
            padding: 6px 12px !important;
            margin: 3px 6px !important;
            background: #8B3A3A !important;  /* پسورد قرمز */
            color: white !important;  /* فونت سفید */
            border: 1px solid #C5A572 !important;
        }
        
        /* بهبود input ها */
        .stTextInput > div > input,
        .stNumberInput > div > div > input,
        textarea,
        select {
            font-size: 14px !important;
            color: #2e2e2e !important;
            background: #fff !important;
        }
        
        /* هدرها رو قرمز کن */
        h1, h2, h3, h4, h5, h6 {
            color: #8B3A3A !important;
        }
        
        /* متن‌های عمومی */
        .stMarkdown, .stText {
            color: #2e2e2e !important;
        }
      }
    </style>
    """, unsafe_allow_html=True)

# =========================
# UI: AUTH PAGES - با طراحی سنتی پیشرفته
# =========================
def signup_page():
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("<div class='persian-pattern' style='text-align: center;'>", unsafe_allow_html=True)
    st.subheader("📝 ثبت‌نام در سامانه املاک شهرستان جرقویه")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    name = col1.text_input("👤 نام کامل")
    email = col2.text_input("📧 ایمیل")
    phone = col1.text_input("📞 شماره تماس (اختیاری)")
    password = col2.text_input("🔒 رمز عبور (حداقل ۶ کاراکتر)", type="password", 
                             help="رمز عبور می‌تواند هر ترکیبی از کاراکترها با حداقل ۶ حرف باشد")
    bio = st.text_area("📝 بیوگرافی (اختیاری)", help="در مورد خودتان و زمینه فعالیت در املاک توضیح دهید")
    
    if st.button("✨ ایجاد حساب کاربری", use_container_width=True):
        if not name or not valid_email(email) or not (strong_password(password) or simple_password(password)) or not valid_phone(phone):
            st.error("لطفاً نام، ایمیل معتبر، رمز (حداقل ۶ کاراکتر) و شماره صحیح وارد کنید.")
        else:
            ok = register_user(name, email, password, role="public", phone=phone, bio=bio)
            if ok: 
                st.success("✅ ثبت‌نام موفق. حالا وارد شوید.")
                st.balloons()
            else: 
                st.error("❌ این ایمیل قبلاً ثبت شده یا ورودی نامعتبر است.")
    st.markdown("</div>", unsafe_allow_html=True)

def login_page():
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("<div class='persian-pattern' style='text-align: center;'>", unsafe_allow_html=True)
    st.subheader("🔐 ورود به سامانه")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # اضافه کردن راهنمای پسورد ساده
    st.info("🔓 **ورود آسان:** پسورد می‌تواند حداقل ۴ کاراکتر باشد")
    
    email = st.text_input("📧 ایمیل")
    password = st.text_input("🔒 رمز عبور", type="password", 
                           help="پسورد می‌تواند هر ترکیبی از کاراکترها با حداقل ۴ حرف باشد")
    
    colA, colB = st.columns(2)
    if colA.button("🚪 ورود به سیستم", use_container_width=True):
        u = login_user(email, password)
        if u:
            st.session_state["user"] = u
            st.success(f"🌹 خوش آمدی {u['name']}")
            st.rerun()
        else:
            st.error("❌ ایمیل یا رمز عبور اشتباه است.")
    
    if colB.button("🔑 فراموشی رمز عبور", use_container_width=True):
        st.session_state["show_reset"] = True
    
    if st.session_state.get("show_reset"):
        st.markdown("<div class='traditional-tab'>", unsafe_allow_html=True)
        st.info("🔐 رمز جدید را تنظیم کنید (حداقل ۴ کاراکتر)")
        e = st.text_input("📧 ایمیل ثبت‌شده", key="rp_e")
        npw = st.text_input("🔒 رمز جدید", type="password", key="rp_p",
                          help="پسورد جدید می‌تواند حداقل ۴ کاراکتر باشد")
        if st.button("🔄 تغییر رمز عبور", use_container_width=True):
            if reset_password(e, npw):
                st.success("✅ رمز تغییر کرد. وارد شوید.")
                st.session_state["show_reset"] = False
            else:
                st.error("❌ ایمیل یا رمز معتبر نیست.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# UI: SEARCH / FILTERS - با طراحی سنتی پیشرفته
# =========================
@st.cache_data(ttl=3600)
def list_all(colname:str) -> List[str]:
    conn=get_conn(); c=conn.cursor()
    try:
        c.execute(f"SELECT DISTINCT {colname} FROM properties WHERE status='published' AND {colname} IS NOT NULL")
        vals = sorted({r[0] for r in c.fetchall() if r[0]})
    except Exception:
        vals=[]
    conn.close(); return vals

def property_filters():
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("<div class='persian-pattern' style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("### 🔍 فیلتر دقیق جستجو")
    st.markdown("</div>", unsafe_allow_html=True)
    
    cities = st.multiselect("🏙️ شهر", options=list_all("city"))
    types  = st.multiselect("🏠 نوع ملک", options=list_all("property_type"))
    
    c1, c2 = st.columns(2)
    min_price = c1.number_input("💰 حداقل قیمت (تومان)", min_value=0, step=100000, value=0)
    max_price = c2.number_input("💰 حداکثر قیمت (تومان)", min_value=0, step=100000, value=0)
    
    a1, a2 = st.columns(2)
    min_area = a1.number_input("📏 حداقل متراژ", min_value=0, step=1, value=0)
    max_area = a2.number_input("📏 حداکثر متراژ", min_value=0, step=1, value=0)
    
    r1, r2 = st.columns(2)
    min_rooms = r1.number_input("🚪 حداقل اتاق", min_value=0, step=1, value=0)
    max_rooms = r2.number_input("🚪 حداکثر اتاق", min_value=0, step=1, value=0)
    
    g1, g2 = st.columns(2)
    min_age = g1.number_input("🏚️ حداقل سن بنا", min_value=0, step=1, value=0)
    max_age = g2.number_input("🏚️ حداکثر سن بنا", min_value=0, step=1, value=0)
    
    facilities = st.multiselect("⭐ امکانات (شامل شود)", ["آسانسور","پارکینگ","انباری","بالکن","استخر","سونا","روف‌گاردن","کمد دیواری"])
    
    st.markdown("<div class='traditional-tab'>", unsafe_allow_html=True)
    st.markdown("#### 🗺️ جستجو بر اساس شعاع مکانی (اختیاری)")
    d1, d2, d3 = st.columns(3)
    center_lat = d1.number_input("📍 عرض جغرافیایی مرکز", format="%.6f")
    center_lon = d2.number_input("📍 طول جغرافیایی مرکز", format="%.6f")
    radius_km  = d3.number_input("📐 شعاع (کیلومتر)", min_value=0, step=1)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='traditional-tab'>", unsafe_allow_html=True)
    st.markdown("#### ⚡ فیلترهای پیشرفته")
    adv1, adv2, adv3 = st.columns(3)
    featured_only = adv1.checkbox("فقط املاک ویژه")
    verified_only = adv2.checkbox("فقط مالکین تأیید شده")
    has_images = adv3.checkbox("فقط دارای تصویر")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.get("user"):
        col_save, col_freq = st.columns(2)
        if col_save.button("💾 ذخیره این جستجو", use_container_width=True):
            filters = {
                "city": cities or None,
                "property_type": types or None,
                "min_price": min_price or None,
                "max_price": max_price if max_price>0 else None,
                "min_area": min_area or None,
                "max_area": max_area if max_area>0 else None,
                "min_rooms": min_rooms or None,
                "max_rooms": max_rooms if max_rooms>0 else None,
                "min_age": min_age or None,
                "max_age": max_age if max_age>0 else None,
                "facilities": facilities or None,
                "featured_only": featured_only,
                "verified_only": verified_only,
                "has_images": has_images
            }
            
            frequency = col_freq.selectbox("فرکانس اطلاع‌رسانی", ["instant", "daily", "weekly"])
            create_search_subscription(st.session_state["user"]["email"], filters, frequency)
            st.success("✅ جستجوی شما ذخیره شد! هنگام پیدا شدن ملک جدید به شما اطلاع می‌دهیم.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return {
        "city": cities or None,
        "property_type": types or None,
        "min_price": min_price or None,
        "max_price": max_price if max_price>0 else None,
        "min_area": min_area or None,
        "max_area": max_area if max_area>0 else None,
        "min_rooms": min_rooms or None,
        "max_rooms": max_rooms if max_rooms>0 else None,
        "min_age": min_age or None,
        "max_age": max_age if max_age>0 else None,
        "facilities": facilities or None,
        "center_lat": center_lat if center_lat else None,
        "center_lon": center_lon if center_lon else None,
        "radius_km": radius_km if radius_km else None,
        "featured_only": featured_only,
        "verified_only": verified_only,
        "has_images": has_images
    }

# =========================
# UI: COMPONENTS - با طراحی سنتی پیشرفته
# =========================
def property_card(row: pd.Series, user: Optional[Dict[str,Any]]):
    card_class = "card featured-card" if row.get('featured') else "card"
    
    with st.container():
        st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
        
        col_title, col_price = st.columns([2, 1])
        col_title.markdown(f"#### {row['title']}")
        col_price.markdown(f"### {format_price(row['price'])}")
        
        if row.get('owner_verified'):
            col_title.markdown("<span class='badge-verified'>✅ تأیید شده</span>", unsafe_allow_html=True)
        
        if row.get('featured'):
            col_price.markdown("<span class='badge-premium'>⭐ ویژه</span>", unsafe_allow_html=True)
        
        st.markdown("<div style='display: flex; flex-wrap: wrap; margin: 10px 0;'>", unsafe_allow_html=True)
        badge(f"🏠 {row['property_type']}")
        badge(f"🏙️ {row['city']}")
        badge(f"📏 {int(row['area'])} متر")
        if row.get('rooms'): badge(f"🚪 {int(row['rooms'])} اتاق")
        if row.get('building_age'): badge(f"🏚️ {int(row['building_age'])} سال")
        if row.get('views'): badge(f"👀 {int(row['views'])} بازدید")
        if row.get('owner_rating'): badge(f"⭐ {float(row['owner_rating']):.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if row.get("address"):
            st.caption(f"📍 {row['address']}")
        
        if row.get("description"):
            st.markdown(f"<div style='background:#f8f5ee;padding:15px;border-radius:12px;border-right:3px solid var(--gold);margin:10px 0;font-size:14px;line-height:1.6'>{row['description'][:200]}{'...' if len(row['description']) > 200 else ''}</div>", unsafe_allow_html=True)
        
        imgs = property_images(int(row["id"]))
        if imgs:
            try:
                st.image(io.BytesIO(imgs[0]), use_column_width=True, caption="تصویر اصلی ملک")
            except Exception:
                try:
                    st.image(base64.b64decode(imgs[0]), use_column_width=True, caption="تصویر اصلی ملک")
                except Exception:
                    pass
        
        cols = st.columns(5)
        if user:
            if cols[0].button("❤️ علاقه‌مندی", key=f"fav_{row['id']}", use_container_width=True):
                _ = toggle_fav(int(row['id']), user["email"])
                st.success("✅ به علاقه‌مندی‌ها اضافه/حذف شد.")
        
        if row.get("video_url"):
            cols[1].markdown(f"[🎥 تور ویدئویی]({row['video_url']})")
        
        if cols[2].button("🗺️ نمایش روی نقشه", key=f"map_{row['id']}", use_container_width=True):
            st.map(pd.DataFrame([[row["latitude"],row["longitude"]]], columns=["lat","lon"]))
        
        if cols[3].button("📄 جزئیات", key=f"view_{row['id']}", use_container_width=True):
            st.query_params["pg"] = "view"
            st.query_params["pid"] = int(row['id'])
            st.rerun()
        
        if user and user["email"] != row["owner_email"]:
            if cols[4].button("💬 گفتگو", key=f"chat_{row['id']}", use_container_width=True):
                st.session_state["chat_pid"]=int(row['id']); st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def image_files_to_bytes(uploaded_files)->List[bytes]:
    out = []
    for f in uploaded_files[:MAX_UPLOAD_IMAGES]:
        if f.type not in ALLOWED_IMAGE_TYPES:
            st.warning(f"فرمت نامجاز: {f.name}")
            continue
        size_mb = (getattr(f, "size", None) or 0) / (1024*1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            st.warning(f"حجم زیاد: {f.name} (حداکثر {MAX_IMAGE_SIZE_MB}MB)")
            continue
        out.append(f.read())
    return out

def paginator(df: pd.DataFrame, page_size:int=8, key:str="pg")->pd.DataFrame:
    if df.empty: return df
    total = len(df)
    pages = (total + page_size - 1)//page_size
    col1,col2 = st.columns([3,2])
    with col1:
        st.caption(f"نتایج: {total} | صفحات: {pages}")
    with col2:
        p = st.number_input("صفحه", min_value=1, max_value=max(pages,1), value=1, step=1, key=key)
    start = (p-1)*page_size
    end = start+page_size
    return df.iloc[start:end]

# =========================
# AGENT PANEL - پنل مشاور
# =========================
def agent_panel(user: Dict[str,Any]):
    st.markdown("<div class='traditional-header'>", unsafe_allow_html=True)
    st.subheader("👔 پنل مشاور املاک")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.info("📊 مدیریت آگهی‌های خودت")
    
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT * FROM properties WHERE owner_email=?", (user["email"],))
    rows=c.fetchall(); cols=[d[0] for d in c.description]; conn.close()
    df=pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
    if df.empty:
        st.info("ℹ️ هنوز ملکی اضافه نکرده‌ای.")
    else:
        st.dataframe(df[["id","title","price","city","property_type","status","views","featured"]])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not df.empty:
        st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
        st.subheader("📈 گزارش‌های عملکرد")
        
        selected_property = st.selectbox("انتخاب ملک برای گزارش", df["id"].tolist(), format_func=lambda x: f"{x} - {df[df['id']==x]['title'].iloc[0]}")
        
        if selected_property:
            report = generate_property_report(selected_property)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("👀 بازدیدها", report["total_views"])
            col2.metric("❤️ علاقه‌مندی‌ها", report["total_favorites"])
            col3.metric("⭐ امتیاز", report["average_rating"])
            col4.metric("📊 امتیاز عملکرد", report["performance_score"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.subheader("❤️ علاقه‌مندی‌های کاربران برای املاک تو")
    
    conn=get_conn(); c=conn.cursor()
    c.execute("""SELECT p.id,p.title,COUNT(f.id) as favs
                 FROM properties p LEFT JOIN favorites f ON p.id=f.property_id
                 WHERE p.owner_email=? GROUP BY p.id,p.title ORDER BY favs DESC""", (user["email"],))
    fav_rows = c.fetchall(); conn.close()
    if fav_rows:
        for pid, title, favs in fav_rows:
            st.write(f"🏠 {pid} | {title} — ❤️ {favs}")
    else:
        st.info("ℹ️ فعلاً علاقه‌مندی ثبت نشده.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MAIN APPLICATION - برنامه اصلی
# =========================
def main():
    st.set_page_config(
        page_title="املاک هوشمند جرقویه - نسخه حرفه‌ای", 
        layout="wide", 
        page_icon="🏡",
        initial_sidebar_state="expanded"
    )
    
    # اعمال استایل
    custom_style()
    
    # راه‌اندازی دیتابیس و سیستم‌ها
    migrate_db()
    initialize_ai_systems()
    
    # خط زرد در بالای صفحه
    st.markdown("<div class='yellow-line'></div>", unsafe_allow_html=True)

    if "user" not in st.session_state:
        st.session_state["user"] = None

    # مدیریت احراز هویت
    if not st.session_state["user"]:
        show_auth_pages()
    else:
        show_main_application()

def show_auth_pages():
    """نمایش صفحات احراز هویت"""
    st.sidebar.markdown("<div class='persian-pattern' style='text-align: center; padding: 20px;'>", unsafe_allow_html=True)
    st.sidebar.title("🏡 املاک هوشمند جرقویه")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("📄 صفحه", ["خانه", "ورود", "ثبت‌نام"])
    
    if page == "ثبت‌نام":
        signup_page()
    elif page == "ورود":
        login_page()
    else:
        show_landing_page()

def show_landing_page():
    """نمایش صفحه اصلی برای کاربران مهمان"""
    st.markdown("<div class='traditional-header'>", unsafe_allow_html=True)
    st.title("🏡 املاک و مستغلات شهرستان جرقویه")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <h2>🌺 به سامانه هوشمند املاک شهرستان جرقویه خوش آمدید</h2>
        <p>برای استفاده از تمامی امکانات سامانه، لطفاً وارد شوید یا ثبت‌نام کنید.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 کاربران", "500+")
    col2.metric("🏠 املاک", "1200+")
    col3.metric("⭐ رضایت", "98%")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center;'>
        <h3>✨ امکانات ویژه سامانه</h3>
    </div>
    """, unsafe_allow_html=True)
    
    features = st.columns(3)
    features[0].markdown("""
    <div style='text-align: center; padding: 15px; background: #f8f5ee; border-radius: 15px; margin: 10px;'>
        <h4>🔍 جستجوی هوشمند</h4>
        <p>پیدا کردن ملک با فیلترهای پیشرفته</p>
    </div>
    """, unsafe_allow_html=True)
    
    features[1].markdown("""
    <div style='text-align: center; padding: 15px; background: #f8f5ee; border-radius: 15px; margin: 10px;'>
        <h4>🗺️ نقشه تعاملی</h4>
        <p>نمایش موقعیت مکانی ملک روی نقشه</p>
    </div>
    """, unsafe_allow_html=True)
    
    features[2].markdown("""
    <div style='text-align: center; padding: 15px; background: #f8f5ee; border-radius: 15px; margin: 10px;'>
        <h4>🤖 پیشنهادات هوشمند</h4>
        <p>سیستم پیشنهاد ملک بر اساس علاقه‌مندی‌ها</p>
    </div>
    """, unsafe_allow_html=True)
    
    features2 = st.columns(3)
    features2[0].markdown("""
    <div style='text-align: center; padding: 15px; background: #f8f5ee; border-radius: 15px; margin: 10px;'>
        <h4>📊 گزارشات پیشرفته</h4>
        <p>آنالیز عملکرد املاک و کاربران</p>
    </div>
    """, unsafe_allow_html=True)
    
    features2[1].markdown("""
    <div style='text-align: center; padding: 15px; background: #f8f5ee; border-radius: 15px; margin: 10px;'>
        <h4>💬 چت و پیام‌رسانی</h4>
        <p>ارتباط مستقیم با مالکین و مشاورین</p>
    </div>
    """, unsafe_allow_html=True)
    
    features2[2].markdown("""
    <div style='text-align: center; padding: 15px; background: #f8f5ee; border-radius: 15px; margin: 10px;'>
        <h4>📱 پنل مدیریت</h4>
        <p>مدیریت کامل املاک و کاربران</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # نمایش برخی از املاک ویژه
    st.markdown("<div class='iranian-border'>", unsafe_allow_html=True)
    st.markdown("### 🏠 املاک ویژه")
    
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT * FROM properties WHERE status='published' AND featured=1 ORDER BY created_at DESC LIMIT 3")
    featured_props = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    
    if featured_props:
        featured_df = pd.DataFrame(featured_props, columns=cols)
        for _, row in featured_df.iterrows():
            property_card(row, None)
    else:
        st.info("ℹ️ فعلاً ملک ویژه‌ای وجود ندارد.")
    st.markdown("</div>", unsafe_allow_html=True)

def show_main_application():
    """نمایش برنامه اصلی"""
    user = st.session_state["user"]
    
    st.sidebar.markdown("<div class='persian-pattern' style='text-align: center; padding: 20px;'>", unsafe_allow_html=True)
    st.sidebar.title("🏡 منوی اصلی")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # اطلاعات کاربر
    st.sidebar.markdown("<div class='traditional-tab'>", unsafe_allow_html=True)
    st.sidebar.write(f"👤 {user['name']}")
    st.sidebar.write(f"🎯 نقش: {user['role']}")
    
    if user["email"] == ADMIN_EMAIL:
        st.sidebar.markdown("<span class='badge-premium'>⭐ مدیر سیستم</span>", unsafe_allow_html=True)
    
    st.sidebar.write(f"⭐ امتیاز: {calculate_user_rating(user['email'])}")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # منوی اصلی بر اساس نقش کاربر
    if user["role"] == "admin" and user["email"] == ADMIN_EMAIL:
        admin_panel(user)
    elif user["role"] == "agent":
        agent_panel(user)
    else:
        public_panel(user)
    
    # دکمه خروج
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 خروج", use_container_width=True):
        st.session_state["user"] = None
        st.rerun()

if __name__ == "__main__":
    main()

