# DCapNet (Deep Capsule Network)

**DCapNet** is a unified deep learning framework proposed for **segmentation and classification of hand gestures** captured in **uncontrolled environments**.  
The proposed end-to-end architecture operates in three main stages:  
(i) **Segmentation** using a U-Net backbone,  
(ii) **Feature extraction and prediction** using multiple deep convolutional models, and  
(iii) **Confidence-based ensemble classification strategy** for robust decision-making.

---

## 🔑 Salient Features of DCapNet

- End-to-end **segmentation + classification** pipeline  
- **U-Net-based segmentation** for precise hand region extraction  
- **Multi-model ensemble** including ResNet, EfficientNet, Inception, and Xception  
- **Confidence-based decision fusion** for final gesture classification  
- Compatible with both **CPU and GPU** systems  
- Modular structure for easy training, evaluation, and extension  
- Supports visualization of segmentation masks, confusion matrix, and learning curves  

---

## ⚙️ Setup and Configuration

The framework is implemented in **Python (PyTorch)** and has been trained and evaluated on the following configuration:

- **Processor:** AMD Ryzen 7 (3.80 GHz)  
- **RAM:** 16 GB  
- **GPU:** NVIDIA GeForce RTX (6 GB VRAM)  
- **Framework:** PyTorch (latest stable version)  
- **Environment:** Linux / Windows / macOS compatible  

### Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/DCapNet.git
cd DCapNet
pip install -r requirements.txt
```

---

## 🧠 Usage

### Training and Evaluation

To train and evaluate the full DCapNet pipeline:

```bash
python DCapNet.py --train_image_dir data/train/images                   --train_mask_dir data/train/masks                   --test_image_dir data/test/images                   --test_mask_dir data/test/masks                   --out_dir dcapnet_output
```

### Output

The framework automatically:
- Trains the **segmentation** and **classification** models  
- Computes performance metrics (IoU, Dice Score,mAP50, Accuracy, Precision, Recall, F1, RMSE, Error Rate)  
- Saves:
  - Best model weights (`*_best.pth`)
  - Metrics (`metrics.json`)
  - Training plots and confusion matrix (`plots/` directory)

---

## 📁 Repository Structure

```
DCapNet/
├── DCapNet.py                 # Main unified training and evaluation script
├── models/                    # Contains U-Net and ensemble model definitions
│   ├── unet.py
│   ├── ensemble.py
├── utils/                     # Utility scripts for metrics, visualization, and data handling
│   ├── data.py
│   ├── metrics.py
│   ├── visualization.py
├── train.py                   # Training loops for segmentation and classification
├── requirements.txt           # Dependencies list
└── README.md                  # Project documentation
```

---

## 📦 Dataset

The model is trained and validated on the publicly available **OUHANDS dataset**, which is available at:  
🔗 [http://www.ouhands.oulu.fi](http://www.ouhands.oulu.fi)

---

## 🧑‍💻 Contributors

**Dr. Ayatullah Faruk Mollah**  
Assistant Professor, Department of Computer Science and Engineering,  
Aliah University, Kolkata, India  

**Taniya Sahana**  
Research Scholar, Department of Computer Science and Engineering,  
Aliah University, Kolkata, India  

---

## 📬 Contact

- **Dr. Ayatullah Faruk Mollah**, Email: [afmollah@aliah.ac.in](mailto:afmollah@aliah.ac.in)  
- **Taniya Sahana**, Email: [taniyaswork@gmail.com](mailto:taniyaswork@gmail.com)

---
