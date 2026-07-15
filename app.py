import os
import sys

# গ্রাফিক্স এরর এড়ানোর কনফিগারেশন
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# ১. প্রিমিয়াম থিম কনফিগারেশন
st.set_page_config(
    page_title="AI Face Golden Ratio Analyzer",
    page_icon="🧬",
    layout="wide"
)

# কাস্টম CSS দিয়ে UI অত্যন্ত আকর্ষণীয় ও আধুনিক করা
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3b82f6;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        margin-bottom: 15px;
        color: white;
    }
    .metric-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #38bdf8;
        margin: 5px 0;
    }
    .metric-desc {
        font-size: 13px;
        color: #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🧬 AI Face Golden Ratio Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 16px;'>অ্যাডভান্সড কম্পিউটার ভিশন ও বায়োমেট্রিক ল্যান্ডমার্কের সাহায্যে ফেসিয়াল সিমেট্রি ও গোল্ডেন রেশিও পরিমাপ</p>", unsafe_allow_html=True)
st.write("---")

# মিডিয়াপাইপ ফেস মেশ কনফিগারেশন
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# গোল্ডেন রেশিও এবং বায়োমেট্রিক ল্যান্ডমার্ক হিসাব করার ফাংশন
def analyze_face_symmetry(landmarks, width, height):
    # ল্যান্ডমার্ক কোঅর্ডিনেট ম্যাপিং
    # ১. সার্বিক ফেস ডাইমেনশন
    forehead = np.array([landmarks[10].x * width, landmarks[10].y * height])
    chin = np.array([landmarks[152].x * width, landmarks[152].y * height])
    left_cheek = np.array([landmarks[234].x * width, landmarks[234].y * height])
    right_cheek = np.array([landmarks[454].x * width, landmarks[454].y * height])
    
    # ২. নাক ও মুখের প্রস্থ (Nose & Mouth Base)
    nose_left = np.array([landmarks[129].x * width, landmarks[129].y * height])
    nose_right = np.array([landmarks[358].x * width, landmarks[358].y * height])
    mouth_left = np.array([landmarks[61].x * width, landmarks[61].y * height])
    mouth_right = np.array([landmarks[291].x * width, landmarks[291].y * height])
    
    # দূরত্ব হিসাব
    face_length = np.linalg.norm(forehead - chin)
    face_width = np.linalg.norm(left_cheek - right_cheek)
    
    nose_width = np.linalg.norm(nose_left - nose_right)
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    
    # রেশিও হিসাব
    face_ratio = face_length / face_width if face_width > 0 else 0
    nose_mouth_ratio = mouth_width / nose_width if nose_width > 0 else 0
    
    points = {
        'forehead': forehead, 'chin': chin, 
        'left_cheek': left_cheek, 'right_cheek': right_cheek,
        'nose_left': nose_left, 'nose_right': nose_right,
        'mouth_left': mouth_left, 'mouth_right': mouth_right
    }
    
    return face_ratio, nose_mouth_ratio, points

# UI কলাম বিন্যাস
col1, col2 = st.columns([1.1, 1.0], gap="large")

with col1:
    st.subheader("📸 ইনপুট সোর্স")
    option = st.radio("ছবি আপলোডের মাধ্যম বেছে নিন:", ("ছবি আপলোড করুন", "ক্যামেরা ব্যবহার করুন"))
    
    image_file = None
    if option == "ছবি আপলোড করুন":
        image_file = st.file_uploader("পরিষ্কার সোজা মুখের ছবি দিন (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    else:
        image_file = st.camera_input("সরাসরি ফ্রন্ট ক্যামেরা ব্যবহার করুন")

if image_file is not None:
    # ইমেজ লোড করা
    image = Image.open(image_file)
    img_array = np.array(image)
    
    # কালার স্পেস ফিক্স করা
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
    h, w, _ = img_array.shape
    results = face_mesh.process(img_array)
    
    with col2:
        st.subheader("🖥️ বায়োমেট্রিক স্ক্যান রিপোর্ট")
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # রেশিও এবং পয়েন্ট সমূহ বিশ্লেষণ
                face_ratio, nose_mouth_ratio, pts = analyze_face_symmetry(face_landmarks.landmark, w, h)
                
                # আউটপুট ছবি কাস্টমাইজড ড্রয়িং (হাই-টেক স্ক্যানার লুক)
                annotated_img = img_array.copy()
                
                # লাইন কালার কোড (BGR)
                color_cyan = (248, 189, 56)   # সায়ান / আকাশী
                color_magenta = (255, 50, 150) # ম্যাজেন্টা
                color_green = (0, 255, 100)   # নিয়ন গ্রিন
                
                # ১. সার্বিক দৈর্ঘ্য ও প্রস্থের রেখা
                cv2.line(annotated_img, tuple(pts['forehead'].astype(int)), tuple(pts['chin'].astype(int)), color_cyan, 3)
                cv2.line(annotated_img, tuple(pts['left_cheek'].astype(int)), tuple(pts['right_cheek'].astype(int)), color_cyan, 3)
                
                # ২. নাক ও মুখের রেখা
                cv2.line(annotated_img, tuple(pts['nose_left'].astype(int)), tuple(pts['nose_right'].astype(int)), color_magenta, 2)
                cv2.line(annotated_img, tuple(pts['mouth_left'].astype(int)), tuple(pts['mouth_right'].astype(int)), color_magenta, 2)
                
                # ৩. জংশনগুলোতে স্ক্যানার নোড আঁকা
                for key, pt in pts.items():
                    cv2.circle(annotated_img, tuple(pt.astype(int)), 7, color_green, -1)
                    cv2.circle(annotated_img, tuple(pt.astype(int)), 10, (255, 255, 255), 1)
                
                # প্রসেসড ছবি প্রদর্শন
                st.image(annotated_img, caption="সরাসরি এআই বায়োমেট্রিক ট্র্যাকিং", use_container_width=True)
                
                # আদর্শ মান ১.৬১৮ এর সাথে নিখুঁত সিমেট্রি তুলনা
                ideal_ratio = 1.618
                
                # ফেস একুরেসি
                face_accuracy = (1 - abs(face_ratio - ideal_ratio) / ideal_ratio) * 100
                face_accuracy = max(0.0, min(100.0, face_accuracy))
                
                # নাক-মুখের একুরেসি
                nm_accuracy = (1 - abs(nose_mouth_ratio - ideal_ratio) / ideal_ratio) * 100
                nm_accuracy = max(0.0, min(100.0, nm_accuracy))
                
                # ড্যাশবোর্ড কার্ড শো করা (HTML কাস্টম ডিজাইন)
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">১. সার্বিক ফেস স্ট্রাকচার অনুপাত</div>
                        <div class="metric-value">{face_ratio:.3f} <span style="font-size: 16px; color: #10b981;">(Φ টার্গেট: 1.618)</span></div>
                        <div class="metric-desc">আপনার ফেস লেন্থ ও উইডথের সিমেট্রি গোল্ডেন রেশিওর সাথে <b>{face_accuracy:.2f}%</b> সামঞ্জস্যপূর্ণ।</div>
                    </div>
                    
                    <div class="metric-card" style="border-color: #ec4899;">
                        <div class="metric-title" style="color: #f472b6;">২. ঠোঁট বনাম নাকের অনুপাত (Nose-to-Mouth Width)</div>
                        <div class="metric-value" style="color: #f472b6;">{nose_mouth_ratio:.3f} <span style="font-size: 16px; color: #10b981;">(Φ টার্গেট: 1.618)</span></div>
                        <div class="metric-desc">ঠোঁট এবং নাকের চওড়ার অনুপাত গোল্ডেন রেশিওর সাথে <b>{nm_accuracy:.2f}%</b> সামঞ্জস্যপূর্ণ।</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # এআই কনক্লুশন ও ফিডব্যাক
                avg_score = (face_accuracy + nm_accuracy) / 2
                st.write("### 🧠 AI সৌন্দর্য ও জ্যামিতিক বিশ্লেষণ:")
                if avg_score > 90:
                    st.balloons()
                    st.success(f"👑 অসাধারণ! আপনার মুখের অনুপাত অত্যন্ত বিরল ও সুন্দর। সমন্বিত স্কোর: **{avg_score:.2f}%**!")
                elif avg_score > 75:
                    st.info(f"✨ চমৎকার! সুষম এবং অত্যন্ত সামঞ্জস্যপূর্ণ ফেসিয়াল সিমেট্রি। সমন্বিত স্কোর: **{avg_score:.2f}%**")
                else:
                    st.warning(f"🙂 আপনার ফেসিয়াল স্ট্রাকচার অত্যন্ত ইউনিক এবং বৈচিত্র্যময়। সমন্বিত স্কোর: **{avg_score:.2f}%**")
        else:
            st.error("⚠️ ছবিতে কোনো মুখমণ্ডল সনাক্ত করা যায়নি! অনুগ্রহ করে আলো উজ্জ্বল রেখে সোজা হয়ে আরেকটি ছবি তুলুন।")
