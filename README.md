# AICTE_Internship_potato_leaf_disease_detector_project

# 🌱 Potato Leaf Disease Detection

##  Project Overview

Potato crops are highly susceptible to diseases that can severely impact yield and quality. This project leverages **Artificial Intelligence (AI) and Deep Learning** techniques to detect diseases in potato leaves using image analysis. By analyzing leaf images, the system can classify the leaf as **healthy** or suffering from diseases such as **Early Blight** and **Late Blight**.

With this tool, farmers can take timely actions to prevent crop loss, ensuring better productivity and sustainability in agriculture.

---

##  Key Features

 1)**AI-based Image Classification:** Uses a Convolutional Neural Network (CNN) to detect diseases.
 
 2)**Streamlit Web Application:** Provides an interactive and user-friendly interface. 
 
 3)**Pretrained Model Integration:** Uses a trained deep learning model for predictions. 
 
 4)**Multi-Orientation Image Processing:** Improves accuracy by analyzing images from different angles. 

 5)**Fast & Efficient:** Provides instant disease detection results.

---

##  Technologies Used

- **Python** - Programming language
- **TensorFlow** - Deep learning framework for model training
- **PyTorch** - Used for experimentation and initial model building
- **Matplotlib & Seaborn** - Data visualization
- **OpenCV** - Image preprocessing and enhancement
- **Streamlit** - Web app development
- **Jupyter Notebook** - Model training and testing

---

##  Project Structure

```
📂 Potato-Disease-Detection
│-- 📂 datasets/                # Training, Validation & Test Images
│-- 📂 models/                  # Saved AI Model
│-- 📜 Train_potato_disease.ipynb  # Model Training Script (Jupyter Notebook)
│-- 📜 web.py                   # Streamlit Web App Script
│-- 📜 README.md                # Project Documentation
│-- 📜 requirements.txt         # Required dependencies
```

---

##  Installation & Setup

### 1️) **Clone the Repository**

```sh
 git clone https://github.com/Pranavnath77/AICTE_Internship_potato_leaf_disease_detector_project.git
cd AICTE_Internship_potato_leaf_disease_detector_project

```

### 2️) **Create Virtual Environment (Optional, Recommended)**

```sh
 python -m venv venv
 source venv/bin/activate  # For Mac/Linux
 venv\Scripts\activate     # For Windows
```

### 3️) **Install Required Libraries**

```sh
 pip install -r requirements.txt
```

### 4️) **Run the Streamlit Web App**

```sh
 streamlit run web.py
```

After running the above command, open **[http://localhost:8501](http://localhost:8501)** in your browser to access the application.

---

##  How to Use the Application?

1. Open the **Streamlit app** in your browser.
2. Upload an image of a potato leaf.
3. Click on the **Predict** button.
4. The model will analyze the image and display the disease classification.

---

## 🛠  Model Training Process

1. **Dataset Preparation:** Images of potato leaves are collected and labeled as **Healthy, Early Blight, and Late Blight**.
2. **Preprocessing:** Images are resized, normalized, and augmented to improve model performance.
3. **CNN Architecture:**
   - Multiple convolutional layers with ReLU activation.
   - Max pooling layers to reduce dimensionality.
   - Fully connected dense layers for classification.
4. **Training & Evaluation:**
   - Model trained using **Categorical Crossentropy Loss** and **Adam Optimizer**.
   - Training history visualized using **Matplotlib**.
   - Model saved as `trained_plant_disease_model.keras`.

---

##  Accuracy & Performance

- **Training Accuracy:** **X%**
- **Validation Accuracy:** **Y%**

Performance graphs for accuracy and loss trends are available in **Train\_potato\_disease.ipynb**.

---

##  Future Improvements

🔹 Extend support for multiple crops and diseases.\
🔹 Improve model accuracy using advanced architectures.\
🔹 Deploy as a mobile application for on-field usage.

---

##  Contributors

- **Pranav Nath** - Developer
- **[Aadharsh P, Jay Rathod]** - Project Guide

---

##  License

This project is **open-source** and available for academic and research purposes.

---

##  Acknowledgment

Special thanks to **[Dataset Source]** for providing the images used for training.






