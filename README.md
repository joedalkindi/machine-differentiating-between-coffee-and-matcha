# ☕️🍵 Coffee vs Matcha Classifier

A simple machine learning project that classifies images as either **coffee** or **matcha** using a Convolutional Neural Network (CNN) built with PyTorch.

---

## 📁 Dataset Structure

Images should be placed like this:

data/  
└── raw/  
  ├── coffee/  
  └── matcha/  

After preprocessing:

data/  
└── processed/  
  ├── train/  
  │  ├── coffee/  
  │  └── matcha/  
  └── val/  
    ├── coffee/  
    └── matcha/  

- All images resized to 224x224  
- 80% training / 20% validation split  

---

## 🧠 Model Overview

- CNN with 3 convolutional layers  
- ReLU activations + MaxPooling  
- Fully connected layers  
- Output: softmax (2 classes: coffee, matcha)  
- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Framework: PyTorch  

---

## ⚙️ How to Use

### 1. Clone the Repository

git clone https://github.com/joedalkindi/machine-differentiating-between-coffee-and-matcha.git

cd machine-differentiating-between-coffee-and-matcha

---

### 2. Install Requirements

pip install -r requirements.txt

---

### 3. Preprocess the Data

python scripts/preprocess.py --input data/raw --output data/processed --img-size 224 --val-split 0.2

---

### 4. Train the Model

python train.py --data data/processed --epochs 15 --batch-size 32 --save-path models/model.pth

---

### 5. Evaluate the Model

python evaluate.py --model models/model.pth --data data/processed/val

---

### 6. Run Inference

python infer.py --model models/model.pth --image samples/test.jpg

---

## 📊 Results

- ~95% accuracy on validation set  
- Confusion matrix and training plots saved in the `results/` folder  
- Sample output: "Prediction: Matcha (Confidence: 92.3%)"

---

## 📄 Project Structure

├── data/  
├── models/  
├── results/  
├── samples/  
├── scripts/  
│  ├── preprocess.py  
│  ├── train.py  
│  ├── evaluate.py  
│  └── infer.py  
├── requirements.txt  
└── README.md  

---

## 👤 Author

**Joe D. Al-Kindi**  
GitHub: [@joedalkindi](https://github.com/joedalkindi)

---

## 📄 License

This project is licensed under the MIT License.
