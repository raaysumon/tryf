import os
import sys

# গ্রাফিক্স কনফ্লিক্ট এড়ানোর কনফিগারেশন
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import face_recognition

# পেজ কনফিগারেশন
st.set_page_config(
    page_title="AI Face Recognition & Symmetry Analyzer",
    page_icon="🧠",
    layout="wide"
)

# সেশন স্টেট (ডাটাবেজ) তৈরি করা - যেখানে নাম ও ফেস ডাটা সেভ থাকবে
if 'known_face_encodings' not in st.session_state:
    st.session_state['known_face_encodings'] = []
if 'known_face_names' not in st.session_state:
    st.session_state['known_face_names'] = []

# প্রিমিয়াম সায়েন্স থিম CSS
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    .report-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1.5px solid #4f46e5;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.15);
        margin-bottom: 15px;
        color: #f3f4f6;
    }
    .report-title { font-size: 13px; font-weight: bold; text-transform: uppercase; color: #818cf8; }
    .report-value { font-size: 32px; font-weight: 900; color: #34d399; margin: 5px 0; }
    .welcome-banner {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #818cf8;'>🧠 AI Face Recognition & Symmetry Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 16px;'>ফেসিয়াল রিকগনিশন মেমোরি ও ৩D বায়োমেট্রিক অ্যানালাইজার</p>", unsafe_allow_html=True)
st.write("---")

# মিডিয়াপাইপ মডিউল
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# পোজ চেকার ফাংশন
def check_face_pose(landmarks):
    il = landmarks[33]
    ir = landmarks[263]
    nose = landmarks[1]
    mid_eye_x = (il.x + ir.x) / 2
    turn_ratio = abs(nose.x - mid_eye_x) / abs(il.x - ir.x)
    tilt = abs(il.y - ir.y)
    return turn_ratio <= 0.08 and tilt <= 0.04

# বায়োমেট্রিক অ্যানালাইসিস ফাংশন
def analyze_comprehensive_face(landmarks, w, h):
    def get_pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float64)

    pts = {
        'forehead': get_pt(10), 'chin': get_pt(152),
        'left_cheek': get_pt(234), 'right_cheek': get_pt(454),
        'nose_top': get_pt(168), 'nose_tip': get_pt(1),
        'nose_left': get_pt(129), 'nose_right': get_pt(358),
        'mouth_left': get_pt(61), 'mouth_right': get_pt(291),
        'eye_left_outer': get_pt(33), 'eye_left_inner': get_pt(133),
        'eye_right_inner': get_pt(362), 'eye_right_outer': get_pt(263),
    }
    face_ratio = np.linalg.norm(pts['forehead'] - pts['chin']) / np.linalg.norm(pts['left_cheek'] - pts['right_cheek'])
    nose_ratio = np.linalg.norm(pts['nose_top'] - pts['nose_tip']) / np.linalg.norm(pts['nose_left'] - pts['nose_right'])
    mouth_nose_ratio = np.linalg.norm(pts['mouth_left'] - pts['mouth_right']) / np.linalg.norm(pts['nose_left'] - pts['nose_right'])
    eye_gap_ratio = np.linalg.norm(pts['eye_left_inner'] - pts['eye_right_inner']) / np.linalg.norm(pts['eye_left_outer'] - pts['eye_left_inner'])

    return {'face_ratio': face_ratio, 'nose_ratio': nose_ratio, 'mouth_nose_ratio': mouth_nose_ratio, 'eye_gap_ratio': eye_gap_ratio, 'points': pts}

# ২টি কলাম লেআউট
col1, col2 = st.columns([1.1, 1.0], gap="large")

with col1:
    st.subheader("📸 এআই স্ক্যানার ক্যামেরা")
    show_mesh_grid = st.checkbox("💻 ৪৬৮-পয়েন্ট এআই গ্রিড চালু রাখুন", value=True)
    image_file = st.camera_input("আপনার মুখ স্ক্যান করুন")

if image_file is not None:
    image = Image.open(image_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    h, w, _ = img_array.shape
    
    # ------------------ ১. ফেস রিকগনিশন পার্ট (Face Recognition) ------------------
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)
    
    identified_name = "Unknown"
    current_face_encoding = None
    
    if len(face_encodings) > 0:
        current_face_encoding = face_encodings[0]
        # আগের সেভ করা ফেসের সাথে মিলানো
        if len(st.session_state['known_face_encodings']) > 0:
            matches = face_recognition.compare_faces(st.session_state['known_face_encodings'], current_face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(st.session_state['known_face_encodings'], current_face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
            
            if best_match_index is not None and matches[best_match_index]:
                identified_name = st.session_state['known_face_names'][best_match_index]

    # ------------------ ২. মিডিয়াপাইপ ল্যান্ডমার্ক ট্র্যাকিং ------------------
    results = face_mesh.process(img_array)
    
    with col2:
        st.subheader("📊 এআই রিকগনিশন ও বায়োমেট্রিক রিপোর্ট")
        
        # নাম সনাক্তকরণের স্বাগতম ব্যানার
        if identified_name != "Unknown":
            st.markdown(f'<div class="welcome-banner">👋 স্বাগতম ফিরে আসার জন্য, {identified_name}!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="welcome-banner" style="background: #ef4444;">👤 নতুন মুখ সনাক্ত করা হয়েছে!</div>', unsafe_allow_html=True)
            
            # নতুন নাম রেজিস্টার করার ইনপুট বক্স
            new_name = st.text_input("এআই আপনাকে চিনতে পারছে না। আপনার নাম লিখে সেভ করুন:")
            if st.button("💾 এআই মেমোরিতে নাম সেভ করুন"):
                if new_name.strip() != "" and current_face_encoding is not None:
                    st.session_state['known_face_encodings'].append(current_face_encoding)
                    st.session_state['known_face_names'].append(new_name.strip())
                    st.success(f"🎉 সফলতা! এআই মেমোরিতে '{new_name}' নামটি সেভ করা হয়েছে। পরেরবার স্ক্যান করলে আপনাকে চিনে ফেলবে।")
                    st.rerun()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                is_straight = check_face_pose(face_landmarks.landmark)
                if not is_straight:
                    st.warning("⚠️ সঠিক স্কোরের জন্য অনুগ্রহ করে ক্যামেরার দিকে একদম সোজা তাকান।")
                
                analysis = analyze_comprehensive_face(face_landmarks.landmark, w, h)
                pts = analysis['points']
                annotated_img = img_array.copy()
                
                # ফেসের উপরে সনাক্তকৃত নামটি লিখে দেওয়া (ওপেনসিভি টেক্সট)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(annotated_img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(annotated_img, identified_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if show_mesh_grid:
                    mp_drawing.draw_landmarks(
                        image=annotated_img, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                # ড্রয়িং লাইন
                color_cyan = (0, 255, 242)
                cv2.line(annotated_img, tuple(pts['forehead'].astype(int)), tuple(pts['chin'].astype(int)), color_cyan, 3)
                cv2.line(annotated_img, tuple(pts['left_cheek'].astype(int)), tuple(pts['right_cheek'].astype(int)), color_cyan, 3)
                
                for key, pt in pts.items():
                    pt_int = (int(round(pt[0])), int(round(pt[1])))
                    cv2.circle(annotated_img, pt_int, 5, (0, 255, 0), -1)
                
                st.image(annotated_img, caption="এআই ট্র্যাকিং ভিউ", use_container_width=True)
                
                # সিমেট্রি ক্যালকুলেশন
                phi = 1.618
                def get_accuracy(val, target):
                    return max(0.0, min(100.0, (1 - abs(val - target) / target) * 100))
                
                acc_face = get_accuracy(analysis['face_ratio'], phi)
                acc_nose = get_accuracy(analysis['nose_ratio'], phi)
                
                st.markdown(f"""
                    <div class="report-card" style="border-color: #00f2fe;">
                        <div class="report-title">১. ফেস স্ট্রাকচার অনুপাত</div>
                        <div class="report-value">{analysis['face_ratio']:.3f}</div>
                        <div class="report-desc">গোল্ডেন রেশিও সিমেট্রি: <b>{acc_face:.2f}%</b></div>
                    </div>
                    <div class="report-card" style="border-color: #ffd200;">
                        <div class="report-title">২. নাকের জ্যামিতিক অনুপাত</div>
                        <div class="report-value">{analysis['nose_ratio']:.3f}</div>
                        <div class="report-desc">গোল্ডেন রেশিও সিমেট্রি: <b>{acc_nose:.2f}%</b></div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("⚠️ মুখমণ্ডল সনাক্ত করা যায়নি।")
