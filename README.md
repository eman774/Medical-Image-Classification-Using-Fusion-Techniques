# 🧠 Medical Image Classification Using Fusion Techniques

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project explores deep learning approaches for medical image classification using different feature fusion strategies. The main focus is comparing **Early Fusion** and **Intermediate Fusion** methods to understand how combining features at different stages affects model performance.

---

## 📌 Project Objective

Medical images often contain complex patterns that require powerful representation learning. This project investigates:

* How feature fusion improves medical image classification
* The difference between **Early Fusion** and **Intermediate Fusion**
* Which fusion strategy provides better performance for respiratory disease detection

The system classifies four types of respiratory conditions:
- ✅ **Healthy**
- ✅ **Lung Cancer**
- ✅ **Pneumonia**
- ✅ **COVID-19**

Using three medical imaging modalities:
- 🔬 **CT Scans** (Computed Tomography)
- 🩻 **Chest X-Rays (CXR)**
- 🎤 **Cough Sound Spectrograms**

---

## 🧪 Fusion Techniques Used

### 🔹 Early Fusion
* Features are combined at the **input level**
* The model learns from merged data from the start
* Simpler pipeline but may lose modality-specific details
* All modalities are concatenated before entering the network

```
[CT + CXR + Cough] → [Deep Network] → [Classification]
```

### 🔹 Intermediate Fusion
* Features are **extracted separately** first
* Fusion happens at **feature level** inside the network
* Preserves richer representations before merging
* Each modality has its own encoder

```
[CT] → [Encoder] ┐
[CXR] → [Encoder] ├→ [Fusion Layer] → [Classifier] → [Output]
[Cough] → [Encoder] ┘
```

---

## 🧠 Deep Learning Workflow

The project follows a complete ML pipeline:

1. **Data Loading & Preprocessing**
   - Dataset splitting (train/validation/test)
   - Image normalization and preparation
   - Multimodal data alignment

2. **Model Building**
   - Convolutional Neural Networks (CNNs)
   - Custom fusion architectures
   - Batch normalization and dropout layers

3. **Fusion Strategy Implementation**
   - Early fusion model
   - Intermediate fusion model

4. **Model Training**
   - Adam optimizer
   - Categorical crossentropy loss
   - 50 epochs with validation monitoring

5. **Evaluation & Comparison**
   - Accuracy metrics
   - Loss curves
   - Confusion matrices
   - Classification reports
   - Performance analysis

---

## 📊 Dataset

The project uses a balanced multimodal dataset:

- **Training set**: 400 samples per class (1,600 total)
- **Validation set**: 50 samples per class (200 total)
- **Test set**: 50 samples per class (200 total)

**Total**: 2,000 multimodal samples across 4 disease categories

### 📥 Dataset Source

**Kaggle Dataset**: [CXR-CT-Cough Dataset](https://www.kaggle.com/datasets/hossamfakher/cxr-ct-cough)

> 💡 **Note**: Download the dataset from Kaggle and organize it according to the structure below before running the notebook.

### Data Organization
```
processed_dataset_split/
├── train/
│   ├── CT/
│   │   ├── Healthy/
│   │   ├── LungsCancer/
│   │   ├── Pneumonia/
│   │   └── covid/
│   ├── CXR/
│   └── Cough sound/
├── valid/
└── test/
```

---

## 📈 Results

The project compares model performance using:

* ✅ **Training & Validation Accuracy**
* ✅ **Loss Curves**
* ✅ **Confusion Matrices**
* ✅ **Per-class Performance Metrics**
* ✅ **Precision, Recall, F1-Score**

The results show how fusion level impacts classification capability in medical imaging tasks. Detailed visualizations include:

- Training progress plots
- Sample predictions
- Class distribution analysis
- Model comparison charts

---

## 🛠 Technologies Used

* **Python** - Programming language
* **TensorFlow / Keras** - Deep learning framework
* **NumPy** - Numerical computing
* **Pandas** - Data manipulation
* **Matplotlib** - Visualization
* **Scikit-learn** - Evaluation metrics
* **PIL (Pillow)** - Image processing
* **Jupyter Notebook** - Development environment

---

## 📁 Project Structure

```
medical-image-fusion-classification/
│
├── early-vs-intermediate-fusion-for-medical-image-cla.ipynb
├── README.md

```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn pillow
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/eman774/medical-image-fusion-classification.git
cd medical-image-fusion-classification
```

2. Download the dataset from Kaggle:
   - Visit the [CXR-CT-Cough Dataset](https://www.kaggle.com/datasets/hossamfakher/cxr-ct-cough) on Kaggle
   - Download and extract the dataset
   - Organize it according to the structure shown below

3. Prepare your dataset according to the structure shown above

4. Open and run the Jupyter notebook:
```bash
jupyter notebook early-vs-intermediate-fusion-for-medical-image-cla.ipynb
```

### Usage

Run all cells in the notebook sequentially to:
- Load and visualize the multimodal dataset
- Train both fusion models
- Compare their performance
- Generate evaluation reports and visualizations

---

## 🎯 Applications

This work is relevant to:

* 🏥 **Medical AI systems**
* 🔍 **Disease detection from images**
* 💻 **Computer-aided diagnosis**
* 🧬 **Multi-modal medical data analysis**
* 🌍 **Healthcare accessibility in remote areas**
* 📊 **Clinical decision support systems**

---

## 🔬 Key Features

- ✨ **Multimodal Learning**: Combines CT, CXR, and cough sound data
- ✨ **Fusion Comparison**: Side-by-side evaluation of fusion strategies
- ✨ **Comprehensive Metrics**: Multiple evaluation perspectives
- ✨ **Visualization Tools**: Rich data and results visualization
- ✨ **Reproducible**: Consistent results with seed setting
- ✨ **Balanced Dataset**: Equal representation across classes

---

## 👩‍💻 Author

**Eman Hisham**  
AI Student | Deep Learning & Computer Vision Enthusiast

📧 Contact: [emanhisham471@gmail.com]  
🔗 LinkedIn: [@EmanHisham](https://www.linkedin.com/in/eman-hisham-607411283/) 
🐙 GitHub: [@EmanHisham](https://github.com/eman774)

---

## 🚀 Future Work

- [ ] Try **Late Fusion** techniques
- [ ] Use larger medical datasets (e.g., MIMIC-CXR, COVID-CT)
- [ ] Apply **transfer learning** with pre-trained models (ResNet, DenseNet)
- [ ] Optimize model architecture with hyperparameter tuning
- [ ] Deploy as a **web-based medical AI tool** (Flask/Streamlit)
- [ ] Implement **attention mechanisms** for better fusion
- [ ] Add **explainability** features (Grad-CAM, LIME)
- [ ] Cross-validation for robust evaluation
- [ ] Real-time inference optimization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🎓 **Dataset source**: [Kaggle CXR-CT-Cough Dataset](https://www.kaggle.com/datasets/hossamfakher/cxr-ct-cough)
- 💡 TensorFlow and Keras teams for the deep learning framework
- 🏥 Medical imaging community for insights on multimodal fusion
- 👥 Open-source contributors and researchers

---

## 📚 References

- Deep Learning for Medical Image Analysis
- Multimodal Fusion Strategies in Healthcare AI
- COVID-19 Detection using Deep Learning
- Computer-Aided Diagnosis Systems

---

## 🔗 Citation

If you use this work in your research, please cite:

```bibtex
@misc{medical-fusion-classification,
  author = {Eman Hisham},
  title = {Medical Image Classification Using Fusion Techniques},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/medical-image-fusion-classification}
}
```

---

**⭐ If you find this project helpful, please consider giving it a star!**
