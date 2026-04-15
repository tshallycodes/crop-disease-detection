# 🌿 Crop Disease Detector

A deep learning web app that identifies plant diseases from leaf photos. Upload a photo of a tomato, potato, or pepper leaf and get an instant diagnosis with treatment advice — powered by EfficientNetV2 trained on 41,000+ plant images.

## Demo

Upload any plant leaf photo → AI identifies the disease → Get treatment advice instantly.

## Results

| Metric | Score |
|--------|-------|
| Dataset | PlantVillage (41,274 images) |
| Classes | 15 disease categories |
| Model | EfficientNetV2B0 (Transfer Learning) |
| Tomato Late Blight | 99.9% confidence |

## Supported Plants & Diseases

| Plant | Conditions |
|-------|-----------|
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |

## Features

- Upload any leaf photo and get instant diagnosis
- Shows top 3 predictions with confidence bars
- Treatment advice for every detected disease
- Green theme — clean agricultural UI
- Trained on 41,000+ real plant images from PlantVillage dataset

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Model | EfficientNetV2B0 (pre-trained on ImageNet) |
| Training Method | Transfer Learning + Data Augmentation |
| Web Framework | Flask |
| Image Processing | Pillow |
| Frontend | HTML, CSS, JavaScript |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/crop-disease-detector.git
cd crop-disease-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the PlantVillage dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and place it at:
```
~/Documents/PlantVillage/PlantVillage/
```

**4. Train the model** *(only needed once)*
```bash
python3 train.py
```

**5. Start the app**
```bash
python3 app.py
```

**6. Open in browser**
```
http://127.0.0.1:5006
```

## Project Structure

```
crop-disease-detector/
├── app.py               # Flask server + prediction logic
├── train.py             # Model training script
├── class_names.json     # List of disease class names
├── templates/
│   └── index.html       # Upload page and results
├── static/
│   └── style.css        # Green agricultural theme
├── uploads/             # Temporary image storage (auto-created)
└── requirements.txt
```

## How It Works

1. User uploads a leaf photo through the web interface
2. Image is resized to 224×224 pixels
3. **EfficientNetV2B0** (pre-trained on ImageNet) extracts features from the leaf
4. Custom classification head identifies the disease from 15 categories
5. Top 3 predictions shown with confidence bars
6. Treatment advice displayed based on the diagnosis

## Training Details

- **Base model:** EfficientNetV2B0 frozen (transfer learning)
- **Data augmentation:** Random flip, rotation, zoom
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical crossentropy
- **Early stopping:** Patience of 4 epochs
- **Dataset split:** 80% train / 20% validation

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Wildfire Detection | YOLOv8 on Sentinel-2 satellite imagery | [wildfire-detection](https://github.com/manny2341/wildfire-detection-and-monitoring-) |
| Emotion Detection | Real-time CNN webcam emotion recognition | [Emotion-Detection](https://github.com/manny2341/Emotion-Detection) |
| Image Classifier | EfficientNetV2 — 1,000 categories | [image-classifier](https://github.com/manny2341/image-classifier) |
| Breast Cancer Analysis | 96.49% accuracy healthcare AI | [breast-cancer-analysis](https://github.com/manny2341/breast-cancer-analysis) |

## Author

[@manny2341](https://github.com/manny2341)
