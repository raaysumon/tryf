# ✨ Face Golden Ratio Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/MediaPipe-00C7B7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe" />
</p>

---

## 📖 প্রজেক্ট পরিচিতি
**Face Golden Ratio Analyzer** হলো একটি রিয়েল-টাইম লাইভ ওয়েব অ্যাপ্লিকেশন, যা বিজ্ঞানের গোল্ডেন রেশিও ($\phi \approx 1.618$) এবং কৃত্রিম বুদ্ধিমত্তা (AI) ব্যবহার করে মানুষের মুখের জ্যামিতিক গঠন এবং সামঞ্জস্যতা (Symmetry) পরিমাপ করে। 

ব্যবহারকারী চাইলে গ্যালারি থেকে যেকোনো ছবি আপলোড করতে পারেন অথবা সরাসরি ডিভাইস ক্যামেরা ব্যবহার করে রিয়েল-টাইমে নিজের বিউটি বা সিমেট্রি স্কোর দেখে নিতে পারেন।

---

## 🚀 চমৎকার ফিচারসমূহ

- 📸 **ডুয়াল ইনপুট সিস্টেম:** ডিভাইস থেকে ছবি আপলোড অথবা সরাসরি ওয়েবক্যাম দিয়ে তাৎক্ষণিক স্ন্যাপ নেওয়ার সুবিধা।
- 📐 **অ্যাডভান্সড ফেসিয়াল অ্যানালাইসিস:**
  - **ফেস রেশিও:** কপাল থেকে থুতনির দৈর্ঘ্য এবং দুই গালের শেষ সীমানার প্রস্থের অনুপাত।
  - **লিপস রেশিও:** নাক থেকে ঠোঁট এবং ঠোঁট থেকে থুতনির দূরত্বের পারস্পরিক অনুপাত।
- 📊 **স্মার্ট মেট্রিক ড্যাশবোর্ড:** গোল্ডেন রেশিও ১.৬১৮ এর সাথে আপনার ফেসের নিখুঁত তুলনা ও লাইভ ডেটা কার্ড।
- 🏆 **ডাইনামিক রেটিং ও প্রোগ্রেস বার:** আপনার স্কোরের ওপর ভিত্তি করে ডায়নামিক কালার কোড ও চমৎকার মন্তব্য (যেমন: Golden Ratio Match, High Symmetry)।

---

## 🛠️ ব্যবহৃত টেকনোলজি (Tech Stack)

- **ব্যাকএন্ড ও ওয়েব ফ্রেমওয়ার্ক:** [Streamlit](https://streamlit.io/) (পাইথনের দ্রুততম ওয়েব ফ্রেমওয়ার্ক)
- **এআই ও ফেস ট্র্যাকিং:** [Google MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- **ইমেজ প্রসেসিং:** [OpenCV](https://opencv.org/) ও [Pillow](https://python-pillow.org/)
- **গাণিতিক গণনা:** [NumPy](https://numpy.org/)

---

## 📂 ফাইল স্ট্রাকচার

```text
├── app.py                  # ওয়েবসাইটের মূল পাইথন রানার কোড
├── requirements.txt        # সার্ভারে হোস্ট করার জন্য প্রয়োজনীয় লাইব্রেরির তালিকা
└── README.md               # প্রজেক্টের চমৎকার ডকুমেন্টেশন ফাইল
