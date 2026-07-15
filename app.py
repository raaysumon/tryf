import os
import sys

# গ্রাফিক্স কনফ্লিক্ট এড়ানোর কনফিগারেশন
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# পেজ কনফিগারেশন
st.set_page_config(
    page_title="AI Facial Landmark & Symmetry Analyzer",
    page_icon="🤖",
    layout="wide"
)

# প্রিমিয়াম ডার্ক থিম ও রেডিমেড কার্ড স্টাইল
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .report-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        padding: 22px;
        border-radius: 15px;
        border: 1.5px solid #4f46e5;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.15);
        margin-bottom: 20px;
        color: #f3f4f6;
    }
    .report-title {
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #818cf8;
    }
    .report-value {
        font-size: 38px;
        font-weight: 900;
        color: #34d399;
        margin: 8px 0;
    }
    .report-desc {
        font-size: 13.5px;
        color: #9ca3af;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #818cf8;'>🤖 AI Facial Landmark & Symmetry Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 16px;'>৪৬৮টি ত্রিমাত্রিক (3D) বায়োমেট্রিক নোডের সমন্বয়ে পূর্ণাঙ্গ ফেস প্রোফাইল বিশ্লেষণ</p>", unsafe_allow_html=True)
st.write("---")

# মিডিয়াপাইপ ফেস ডিক্টেশন মডিউল
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True, # চোখ ও ঠোঁটের ভেতরের নিখুঁত ট্র্যাক নিশ্চিত করতে এটি True রাখা হয়েছে
    min_detection_confidence=0.5
)

# বায়োমেট্রিক পরিমাপের প্রধান ফাংশন
def analyze_comprehensive_face(landmarks, w, h):
    # কোঅর্ডিনেট পয়েন্ট বের করার হেল্পার ফাংশন
    def get_pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    # ল্যান্ডমার্ক ডাটা পয়েন্ট ম্যাপিং (মিডিয়াপাইপ ৩D ডায়াগ্রাম অনুযায়ী)
    pts = {
        'forehead': get_pt(10),
        'chin': get_pt(152),
        'left_cheek': get_pt(234),
        'right_cheek': get_pt(454),
        'nose_top': get_pt(168),   # নাকের ওপরের অংশ
        'nose_tip': get_pt(1),     # নাকের ডগা
        'nose_left': get_pt(129),
        'nose_right': get_pt(358),
        'mouth_left': get_pt(61),
        'mouth_right': get_pt(291),
        'eye_left_outer': get_pt(33),
        'eye_left_inner': get_pt(133),
        'eye_right_inner': get_pt(362),
        'eye_right_outer': get_pt(263),
    }

    # ১. সার্বিক মুখাবয়ব (Overall Face Ratio)
    face_len = np.linalg.norm(pts['forehead'] - pts['chin'])
    face_wid = np.linalg.norm(pts['left_cheek'] - pts['right_cheek'])
    face_ratio = face_len / face_wid if face_wid > 0 else 0

    # ২. নাকের জ্যামিতি (Nose Ratio)
    nose_len = np.linalg.norm(pts['nose_top'] - pts['nose_tip'])
    nose_wid = np.linalg.norm(pts['nose_left'] - pts['nose_right'])
    nose_ratio = nose_len / nose_wid if nose_wid > 0 else 0

    # ৩. ঠোঁট ও নাকের অনুপাত (Lip to Nose Ratio)
    mouth_wid = np.linalg.norm(pts['mouth_left'] - pts['mouth_right'])
    mouth_nose_ratio = mouth_wid / nose_wid if nose_wid > 0 else 0

    # ৪. চোখের প্রতিসাম্য অনুপাত (Eye Proportions)
    left_eye_wid = np.linalg.norm(pts['eye_left_outer'] - pts['eye_left_inner'])
    right_eye_wid = np.linalg.norm(pts['eye_right_outer'] - pts['eye_right_inner'])
    eye_gap = np.linalg.norm(pts['eye_left_inner'] - pts['eye_right_inner'])
    
    # চোখের উইডথ এবং চোখের মধ্যকার দূরত্বের অনুপাত (আদর্শ মান: ১.০)
    eye_gap_ratio = eye_gap / left_eye_wid if left_eye_wid > 0 else 0

    return {
        'face_ratio': face_ratio,
        'nose_ratio': nose_ratio,
        'mouth_nose_ratio': mouth_nose_ratio,
        'eye_gap_ratio': eye_gap_ratio,
        'points': pts
    }

# ২টি কলামের স্ট্রাকচার
col1, col2 = st.columns([1.1, 1.0], gap="large")

with col1:
    st.subheader("📸 ক্যামেরা ও কন্ট্রোল প্যানেল")
    
    # ৩D গ্রিড অন বা অফ করার কন্ট্রোল অপশন
    show_mesh_grid = st.checkbox("💻 ৪৬৮-পয়েন্ট এআই গ্রিড (Face Mesh) চালু করুন", value=True)
    
    option = st.radio("ইনপুট সোর্স সিলেক্ট করুন:", ("ফাইল আপলোড", "লাইভ ক্যামেরা স্ন্যাপ"))
    
    image_file = None
    if option == "ফাইল আপলোড":
        image_file = st.file_uploader("আপনার একটি সোজা ছবি ড্রপ করুন", type=['jpg', 'jpeg', 'png'])
    else:
        image_file = st.camera_input("ছবি তোলার সময় সোজা তাকান")

if image_file is not None:
    image = Image.open(image_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
    h, w, _ = img_array.shape
    results = face_mesh.process(img_array)
    
    with col2:
        st.subheader("📊 মাল্টি-পয়েন্ট এআই রিপোর্ট")
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                analysis = analyze_comprehensive_face(face_landmarks.landmark, w, h)
                pts = analysis['points']
                
                annotated_img = img_array.copy()
                
                # যদি ব্যবহারকারী ৩D মেশ গ্রিড অন রাখে
                if show_mesh_grid:
                    mp_drawing.draw_landmarks(
                        image=annotated_img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                # ল্যান্ডমার্ক কানেক্টিং কালার ও ড্রয়িং
                color_cyan = (0, 255, 242)      # প্রধান মুখমণ্ডল (Cyan)
                color_magenta = (234, 0, 255)   # চোখ ও কপাল (Magenta)
                color_yellow = (255, 226, 0)    # নাক ও ঠোঁট (Yellow)
                
                # প্রধান রেখাগুলো ড্র করা
                cv2.line(annotated_img, tuple(pts['forehead'].astype(int)), tuple(pts['chin'].astype(int)), color_cyan, 3)
                cv2.line(annotated_img, tuple(pts['left_cheek'].astype(int)), tuple(pts['right_cheek'].astype(int)), color_cyan, 3)
                cv2.line(annotated_img, tuple(pts['nose_top'].astype(int)), tuple(pts['nose_tip'].astype(int)), color_yellow, 3)
                cv2.line(annotated_img, tuple(pts['nose_left'].astype(int)), tuple(pts['nose_right'].astype(int)), color_yellow, 3)
                cv2.line(annotated_img, tuple(pts['mouth_left'].astype(int)), tuple(pts['mouth_right'].astype(int)), color_yellow, 3)
                cv2.line(annotated_img, tuple(pts['eye_left_outer'].astype(int)), tuple(pts['eye_right_outer'].astype(int)), color_magenta, 2)
                
                # ল্যান্ডমার্ক জংশনগুলোতে সার্কেল করা
                for key, pt in pts.items():
                    cv2.circle(annotated_img, tuple(pt.astype(int)), 6, (0, 255, 0), -1)
                    cv2.circle(annotated_img, tuple(pt.astype(int)), 9, (255, 255, 255), 1)
                
                # আউটপুট ইমেজ শো করা
                st.image(annotated_img, caption="কম্পিউটার ভিশন এনালাইসিস ভিউ", use_container_width=True)
                
                # আদর্শ বৈজ্ঞানিক মানসমূহ
                phi = 1.618
                ideal_eye_gap = 1.000
                
                # নির্ভুলতা পরিমাপের ফর্মুলা
                def get_accuracy(val, target):
                    acc = (1 - abs(val - target) / target) * 100
                    return max(0.0, min(100.0, acc))
                
                acc_face = get_accuracy(analysis['face_ratio'], phi)
                acc_nose = get_accuracy(analysis['nose_ratio'], phi)
                acc_lip_nose = get_accuracy(analysis['mouth_nose_ratio'], phi)
                acc_eye = get_accuracy(analysis['eye_gap_ratio'], ideal_eye_gap)
                
                # ড্যাশবোর্ড কার্ড শো করা (সাই সায়েন্স গ্রাফিক্স)
                st.markdown(f"""
                    <div class="report-card" style="border-color: #00f2fe;">
                        <div class="report-title" style="color: #00f2fe;">১. ফেস স্ট্রাকচার সিমেট্রি (Face Ratio)</div>
                        <div class="report-value" style="color: #00f2fe;">{analysis['face_ratio']:.3f} <span style="font-size: 16px; color: #9ca3af;">(আদর্শ: 1.618)</span></div>
                        <div class="report-desc">মুখমণ্ডলের অনুপাত গোল্ডেন রেশিওর সাথে <b>{acc_face:.2f}%</b> সামঞ্জস্যপূর্ণ।</div>
                    </div>
                    
                    <div class="report-card" style="border-color: #ffd200;">
                        <div class="report-title" style="color: #ffd200;">২. নাকের গঠন অনুপাত (Nose Bridge Symmetry)</div>
                        <div class="report-value" style="color: #ffd200;">{analysis['nose_ratio']:.3f} <span style="font-size: 16px; color: #9ca3af;">(আদর্শ: 1.618)</span></div>
                        <div class="report-desc">নাকের দৈর্ঘ্য ও প্রস্থের অনুপাত গোল্ডেন রেশিওর সাথে <b>{acc_nose:.2f}%</b> সামঞ্জস্যপূর্ণ।</div>
                    </div>

                    <div class="report-card" style="border-color: #38bdf8;">
                        <div class="report-title" style="color: #38bdf8;">৩. লিপস-টু-নোজ অনুপাত (Nose-to-Mouth Width)</div>
                        <div class="report-value" style="color: #38bdf8;">{analysis['mouth_nose_ratio']:.3f} <span style="font-size: 16px; color: #9ca3af;">(আদর্শ: 1.618)</span></div>
                        <div class="report-desc">নাকের চেয়ে ঠোঁট কতখানি চওড়া তার অনুপাত গোল্ডেন রেশিওর সাথে <b>{acc_lip_nose:.2f}%</b> সামঞ্জস্যপূর্ণ।</div>
                    </div>

                    <div class="report-card" style="border-color: #ec4899;">
                        <div class="report-title" style="color: #ec4899;">৪. চোখের সমন্বয় ও গ্যাপ অনুপাত (Interpupillary Ratio)</div>
                        <div class="report-value" style="color: #ec4899;">{analysis['eye_gap_ratio']:.3f} <span style="font-size: 16px; color: #9ca3af;">(আদর্শ: 1.000)</span></div>
                        <div class="report-desc">দুই চোখের ভেতরের মধ্যবর্তী দূরত্ব এক চোখের সাইজের প্রায় <b>{acc_eye:.2f}%</b> নিখুঁত পরিমাপ।</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # ওভারঅল এআই বায়োমেট্রিক স্কোর
                overall_score = (acc_face + acc_nose + acc_lip_nose + acc_eye) / 4
                
                st.write("### 🧬 AI বায়োমেট্রিক সারসংক্ষেপ:")
                if overall_score > 90:
                    st.balloons()
                    st.success(f"🏆 অসাধারণ ফেসিয়াল হারমোনি! সামগ্রিক সিমেট্রি স্কোর: **{overall_score:.2f}%**!")
                elif overall_score > 75:
                    st.info(f"✨ চমৎকার ও সুষম মুখের গঠন। সামগ্রিক সিমেট্রি স্কোর: **{overall_score:.2f}%**")
                else:
                    st.warning(f"🙂 বেশ ইউনিক এবং বৈচিত্র্যময় ফেসিয়াল ফিচার। সামগ্রিক সিমেট্রি স্কোর: **{overall_score:.2f}%**")
        else:
            st.error("⚠️ মুখমণ্ডল ঠিকভাবে স্ক্যান করা যাচ্ছে না। দয়া করে সোজা তাকিয়ে পর্যাপ্ত আলোতে আরেকটি পরিষ্কার ছবি নিন।")
