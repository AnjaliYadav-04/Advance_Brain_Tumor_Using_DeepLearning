## 🧠 Advance Brain Tumor Classification using Deep Learning
A robust Deep Learning based medical imaging project that **classifies brain MRI** scans into tumor categories using **Convolutional Neural Networks (CNN)**.
This project automates tumor detection and classification to **assist radiologists in faster and more accurate diagnosis.**
**
### 📌 Project Overview
- Brain tumor diagnosis through MRI analysis is a critical and time-sensitive medical task. Manual interpretation is error-prone and time-consuming.
- This project introduces an AI-powered automated tumor classification system that learns discriminative features from MRI images and classifies them into tumor types using Deep Learning models.
- The system improves diagnostic reliability and supports early detection for better treatment planning.

### ✨ Key Features
-  Automatic brain tumor detection from MRI scans
- Multi-class tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- CNN based Deep Learning architecture
- High accuracy model with visual performance evaluation
- Confusion matrix & classification reports
-  Training/validation loss & accuracy visualization
-  Clean, reproducible Jupyter Notebook pipeline

### 🧱 Repository Structure
Advance_Brain_Tumor_Classification/
│
├── Advance DL Project Brain Tumor Image Classification.ipynb
├── archive.zip
├── README.md
└── .gitignore

### 🛠️ Technologies Used
1. Technology	Purpose
2. Python	Core language
3. Jupyter Notebook	Model development
4. TensorFlow / Keras	Deep Learning framework
5. NumPy & Pandas	Data handling
6. OpenCV	Image preprocessing
7. Matplotlib & Seaborn	Visualization
8. Kaggle MRI Dataset	Training data

### 📂 Dataset
- Dataset is not included due to size limits.
- Use a public Brain MRI dataset (e.g., Kaggle) containing:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor
- Images should be placed in structured folders for training, validation, and testing.

### 🚀 Getting Started
**1. Clone Repository**
- git clone https://github.com/Chando0185/Advance_Brain_Tumor_Classification.git
- cd Advance_Brain_Tumor_Classification

**2. Install Dependencies**
- pip install -r requirements.txt

**3. Add Dataset**
- Create:
 data/
   ├── train/
   ├── validation/
   └── test/
- Place class-wise MRI images inside.

**4. Run Notebook**
- jupyter notebook

**Open: Advance DL Project Brain Tumor Image Classification.ipynb**

### 📊 Output & Results
- Trained CNN model
- Classification accuracy & loss graphs
- Confusion matrix
- Prediction visualization on test images
- Expected Accuracy: 90%+ (depends on dataset & training epochs)

### ⚙️ Future Improvements
- Transfer Learning (VGG16, ResNet50, EfficientNet)
- Data augmentation pipeline
- Model deployment using Flask / FastAPI / Streamlit
- Web & mobile diagnostic interface
- Real-time MRI upload & prediction system

### 📜 License
Open-source — free for academic & research use.

## 💬 Contact
**Author: Anjali Yadav**
🔗 GitHub: https://github.com/AnjaliYadav-04
