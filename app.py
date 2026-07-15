import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# ১. পেজ সেটআপ
st.set_page_config(page_title="Golden Ratio Face Analyzer", page_icon="✨", layout="centered")

st.title("✨ Face Golden Ratio Analyzer")
st.write("আপনার ছবি আপলোড করুন অথবা ক্যামেরা দিয়ে ছবি তুলুন। বিজ্ঞানের গোল্ডেন রেশিও (1.618) অনুযায়ী আপনার মুখের সিমেট্রি ও গঠন কতখানি নিখুঁত, তা রিয়েল-টাইমে পরীক্ষা করুন!")

# ২. MediaPipe ও ধ্রুবক সেটআপ
mp_face_mesh = mp.solutions.face_mesh
GOLDEN_RATIO = 1.618

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ৩. ছবি আপলোড বা ক্যামেরা ইনপুট অপশন
option = st.radio("ইনপুট অপশন সিলেক্ট করুন:", ("ছবি আপলোড করুন", "ক্যামেরা ব্যবহার করুন"))

image_file = None
if option == "ছবি আপলোড করুন":
    image_file = st.file_uploader("একটি ছবি সিলেক্ট করুন (JPG/PNG):", type=["jpg", "png", "jpeg"])
else:
    image_file = st.camera_input("ক্যামেরার সামনে সোজা হয়ে দাঁড়িয়ে ছবি তুলুন:")

if image_file is not None:
    # PIL ইমেজকে OpenCV ফরম্যাটে কনভার্ট করা
    image = Image.open(image_file)
    img_array = np.array(image)
    
    # RGB থেকে BGR কনভার্ট (OpenCV প্রসেসিংয়ের জন্য)
    if img_array.shape[2] == 4: # RGBA হলে RGB করা
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    h, w, _ = img_array.shape
    
    # MediaPipe দিয়ে ফেস ল্যান্ডমার্ক ডিটেকশন
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_array)
        
        if results.multi_face_landmarks:
            # ছবিটির ওপর আঁকার জন্য একটি কপি তৈরি
            annotated_img = img_array.copy()
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))

                # ফেস পরিমাপ (উচ্চতা ও প্রস্থ)
                forehead_top = landmarks[10]
                chin_bottom = landmarks[152]
                face_height = get_distance(forehead_top, chin_bottom)

                left_cheek = landmarks[234]
                right_cheek = landmarks[454]
                face_width = get_distance(left_cheek, right_cheek)

                # নাক ও ঠোঁটের পরিমাপ
                nose_bottom = landmarks[2]
                lips_center = landmarks[0]
                dist_nose_to_lips = get_distance(nose_bottom, lips_center)
                dist_lips_to_chin = get_distance(lips_center, chin_bottom)

                if face_width > 0 and dist_lips_to_chin > 0:
                    ratio_face = face_height / face_width
                    diff_face = abs(ratio_face - GOLDEN_RATIO)
                    score_face = max(0.0, min(100.0, 100.0 - (diff_face / GOLDEN_RATIO) * 100))

                    ratio_lips = dist_lips_to_chin / (dist_nose_to_lips if dist_nose_to_lips > 0 else 1)
                    diff_lips = abs(ratio_lips - GOLDEN_RATIO)
                    score_lips = max(0.0, min(100.0, 100.0 - (diff_lips / GOLDEN_RATIO) * 100))

                    total_score = (score_face + score_lips) / 2

                    # লাইভ লাইন ড্র করা
                    cv2.line(annotated_img, forehead_top, chin_bottom, (0, 255, 0), 4)  # সবুজ লাইন
                    cv2.line(annotated_img, left_cheek, right_cheek, (255, 0, 0), 4)   # নীল লাইন

                    # রেজাল্ট প্রদর্শন (Streamlit UI-তে)
                    st.image(annotated_img, caption="Processed Image", use_container_width=True)
                    
                    st.subheader("📊 আপনার ফেস অ্যানালাইসিস রেজাল্ট:")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Face Ratio", value=f"{ratio_face:.2f}", delta=f"{ratio_face - GOLDEN_RATIO:.2f} (vs 1.618)")
                    with col2:
                        st.metric(label="Lips Ratio", value=f"{ratio_lips:.2f}", delta=f"{ratio_lips - GOLDEN_RATIO:.2f} (vs 1.618)")
                    with col3:
                        st.metric(label="Overall Score", value=f"{total_score:.1f}%")
                    
                    # স্কোরের মন্তব্য ও প্রগ্রেস বার
                    if total_score >= 90:
                        st.success("🏆 **Golden Ratio Match!** আপনার মুখের জ্যামিতিক সামঞ্জস্য অসাধারণ ও নিখুঁত!")
                    elif total_score >= 75:
                        st.info("⭐ **High Symmetry!** আপনার ফেসের সিমেট্রি খুবই আকর্ষণীয় এবং ব্যালেন্সড।")
                    else:
                        st.warning("🙂 **Average Symmetry.** সাধারণ মানের সিমেট্রি। (সঠিক ফলাফলের জন্য ক্যামেরার দিকে একদম সোজা তাকিয়ে ছবি তুলুন)।")
                    
                    st.progress(int(total_score))
        else:
            st.error("দুঃখিত! ছবিতে কোনো মুখ সনাক্ত করা যায়নি। দয়া করে ভালো আলোতে তোলা পরিষ্কার ছবি ব্যবহার করুন।")
