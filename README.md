# 🎶 Music-Genre-Detection  

## 📌 Overview  
A machine learning project to automatically classify music into genres using both **supervised and unsupervised learning algorithms**. The system leverages advanced audio feature extraction to identify unique patterns across genres and provides insights into clustering and classification performance.  

---

## 📂 Dataset  
- **GTZAN Music Genre Dataset** containing **1000 audio tracks** (each 30 seconds long).  
- **10 genres** represented (100 tracks per genre).  
- Widely used benchmark dataset for **music information retrieval** tasks.  

---

## 🛠️ Feature Extraction  
- **Librosa** library was used for audio signal processing.  
- Extracted features:  
  - MFCC (Mel-Frequency Cepstral Coefficients)  
  - Chroma-STFT  
  - Spectral Contrast  
  - Tonnetz features  

---

## 🤖 Techniques Used  
### Supervised Learning  
- KNN (K-Nearest Neighbors)  
- Random Forest  
- Decision Tree  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Neural Networks  

### Unsupervised Learning  
- K-Means Clustering  
- Hierarchical Clustering  

---

## 🚀 Project Highlights  
- Designed a **music genre detection system** using supervised and unsupervised approaches.  
- Achieved structured classification by leveraging **Librosa-based feature extraction**.  
- Compared performance of multiple algorithms for accuracy and clustering efficiency.  
- Gained insights into **genre similarities and separability** through clustering analysis.  

---

## 📊 Results (Optional – if you have)  
- Best performing model: (e.g., **Random Forest – XX% accuracy**)  
- Visualization of clusters and confusion matrix for classification.  

---

## 🔮 Future Work  
- Experiment with **deep learning models** (CNNs, RNNs for spectrograms).  
- Use **larger and more diverse datasets** beyond GTZAN.  
- Deploy the model as a **web application** for real-time genre prediction. 
