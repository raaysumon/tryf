# ১. ওএস ও ওপেনসিভি কনফ্লিক্ট এড়ানোর জন্য ম্যাজিক লাইন (অবশ্যই সবার উপরে থাকবে)
import os
import sys
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# স্ট্রিমলিট পেজ কনফিগারেশন
st.set_page_config(
    page_title="Face Golden Ratio Detector",
    page_icon="👑",
    layout="centered"
)

st.title("👑 Face Golden Ratio Detector")
st.write("আপনার মুখের গোল্ডেন রেশিও (Φ - 1.618) পরিমাপ করুন।")

# মিডিয়াপাইপ ফেস মেশ (Face Mesh) সেটআপ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# গোল্ডেন রেশিও হিসাব করার ফাংশন
def calculate_golden_ratio(landmarks, width, height):
    # নির্দিষ্ট ল্যান্ডমার্কের কোঅর্ডিনেট বের করা (মিডিয়াপাইপ ইনডেক্স অনুযায়ী)
    # ১০ = কপালের ওপরের অংশ, ১৫২ = থুতনি
    # ২৩৪ = বাম গাল (সর্ববামে), ৪৫৪ = ডান গাল (সর্বডানে)
    
    forehead = np.array([landmarks[10].x * width, landmarks[10].y * height])
    chin = np.array([landmarks[152].x * width, landmarks[152].y * height])
    left_cheek = np.array([landmarks[234].x * width, landmarks[234].y * height])
    right_cheek = np.array([landmarks[454].x * width, landmarks[454].y * height])
    
    # মুখের দৈর্ঘ্য এবং প্রস্থ হিসাব
    face_length = np.linalg.norm(forehead - chin)
    face_width = np.linalg.norm(left_cheek - right_cheek)
    
    # রেশিও বের করা
    ratio = face_length / face_width if face_width > 0 else 0
    return ratio, forehead, chin, left_cheek, right_cheek

# ইনপুট অপশন: ফাইল আপলোড নাকি ক্যামেরা
option = st.radio("ছবি ইনপুট করার মাধ্যম বেছে নিন:", ("ছবি আপলোড করুন", "ক্যামেরা ব্যবহার করুন"))

image_file = None

if option == "ছবি আপলোড করুন":
    image_file = st.file_uploader("আপনার একটি পরিষ্কার সোজা মুখের ছবি আপলোড করুন (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
else:
    image_file = st.camera_input("সরাসরি ক্যামেরা দিয়ে ছবি তুলুন")

if image_file is not None:
    # PIL Image কে OpenCV ফর্ম্যাটে কনভার্ট করা
    image = Image.open(image_file)
    img_array = np.array(image)
    
    # RGB ফর্ম্যাটে কনভার্ট (কারন MediaPipe RGB ব্যবহার করে)
    if img_array.shape[2] == 4: # RGBA হলে RGB করা
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    h, w, _ = img_array.shape
    
    # ফেস মেশ প্রসেসিং
    results = face_mesh.process(img_array)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # রেশিও এবং মূল পয়েন্টগুলো হিসাব করা
            ratio, top, bottom, left, right = calculate_golden_ratio(face_landmarks.landmark, w, h)
            
            # ছবিতে ড্রয়িং করা (লাইন টানা)
            annotated_img = img_array.copy()
            
            # দৈর্ঘ্য মাপার লাইন (কপাল থেকে থুতনি - নীল রঙ)
            cv2.line(annotated_img, tuple(top.astype(int)), tuple(bottom.astype(int)), (0, 120, 255), 3)
            # প্রস্থ মাপার লাইন (বাম গাল থেকে ডান গাল - লাল রঙ)
            cv2.line(annotated_img, tuple(left.astype(int)), tuple(right.astype(int)), (255, 50, 50), 3)
            
            # ল্যান্ডমার্ক পয়েন্টগুলোতে ছোট গোল বৃত্ত আঁকা
            for pt in [top, bottom, left, right]:
                cv2.circle(annotated_img, tuple(pt.astype(int)), 6, (0, 255, 0), -1)
                
            # ফলাফল স্ক্রিনে দেখানো
            st.image(annotated_img, caption="প্রসেসড ছবি", use_column_width=True)
            
            st.subheader("📊 পরিমাপের ফলাফল:")
            st.write(f"**আপনার মুখের দৈর্ঘ্য ও প্রস্থের অনুপাত (Ratio):** `{ratio:.3f}`")
            
            # গোল্ডেন রেশিও ১.৬১৮ এর সাথে তুলনা
            ideal_ratio = 1.618
            accuracy = (1 - abs(ratio - ideal_ratio) / ideal_ratio) * 100
            accuracy = max(0, min(100, accuracy)) # ০ থেকে ১০০ এর মধ্যে রাখা
            
            st.info(f"🧬 আদর্শ গোল্ডেন রেশিও হলো **1.618**। আপনার মুখমণ্ডল আদর্শ অনুপাতের তুলনায় **{accuracy:.2f}%** সামঞ্জস্যপূর্ণ!")
            
            # ফিডব্যাক
            if accuracy > 90:
                st.success("🎉 চমৎকার! আপনার ফেস কাটিং প্রায় নিখুঁত গোল্ডেন রেশিও মেনে চলে।")
            elif accuracy > 75:
                st.success("✨ দারুণ! আপনার মুখের অনুপাত অত্যন্ত আকর্ষণীয় এবং সুষম।")
            else:
                st.warning("🙂 আপনার মুখের অনুপাত বেশ ইউনিক ও স্বাভাবিক।")
                
    else:
        st.error("⚠️ ছবিতে কোনো মুখমণ্ডল সনাক্ত করা যায়নি! দয়া করে সোজা হয়ে ক্যামেরার দিকে তাকিয়ে আরেকটি ছবি দিন।")
