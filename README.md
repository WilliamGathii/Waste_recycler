# ♻️ Waste Management AI Classifier

A **Mobile AI-powered waste classification system** built using **TensorFlow/Keras**, **TFLite**, and **Flutter**.  
It detects whether a waste item is **Recyclable** or **Non-Recyclable** and provides **recycling guidance** for different material types.  

## 🚀 Features
- **Real-time classification** of waste images (via camera or gallery)
- **High-accuracy model** trained on multiple waste categories
- **Recyclable / Non-Recyclable detection**
- **Recycling tips** for each class
- **Offline support** (TFLite model runs directly on device)
- **Classification history** saved locally

---

## 📂 Dataset
The dataset is organized into two main categories:

```
Recycle Dataset/
│── recyclable/
│   ├── alluminium/
│   ├── cardboard/
│   ├── glass/
│   ├── paper/
│   ├── plastic/
│
└── non_recyclable/
    ├── diaper/
    ├── organic_waste/
    ├── pizza_box/
    ├── styrofoam/
    ├── tissue/
```

### **Classes & Categories**
| Class           | Category         |
|-----------------|------------------|
| Alluminium      | Recyclable       |
| Cardboard       | Recyclable       |
| Glass           | Recyclable       |
| Paper           | Recyclable       |
| Plastic         | Recyclable       |
| Diaper          | Non-Recyclable   |
| Organic Waste   | Non-Recyclable   |
| Pizza Box*      | Conditional      |
| Styrofoam       | Non-Recyclable   |
| Tissue          | Non-Recyclable   |

> **Note:** Pizza boxes can be recycled if clean; greasy parts should be discarded.

---

## 🛠️ Model Development

### 1️⃣ **Model Architecture**
The model uses **Transfer Learning** with **MobileNetV2/MobileNetV3**:
- Input shape: `224x224x3`
- Frozen base layers
- Global Average Pooling
- Dropout (0.5)
- Dense layer with Softmax for classification

Example Keras setup:
```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

---

### 2️⃣ **Training**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Callbacks:
  - `ModelCheckpoint`
  - `EarlyStopping`
  - `ReduceLROnPlateau`
- Epochs: **30**
- Batch size: **32**
- Validation split: **20%**

---

### 3️⃣ **TFLite Conversion**
The trained `.h5` model is converted to `.tflite`:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('recycle_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 📱 Flutter App Integration
- **Framework**: Flutter
- **Plugin**: [`tflite_flutter`](https://pub.dev/packages/tflite_flutter)
- **Assets**:
  ```
  assets/
  ├── recycle_classifier.tflite
  ├── labels.txt
  ```
- **Image Picker**: For capturing or selecting an image
- **Local Storage**: `shared_preferences` for storing classification history

---

## 🖼️ App Flow
1. **User selects/captures an image**
2. **Image is resized and preprocessed** to match model input
3. **Model predicts class & confidence score**
4. **App displays category, confidence, and recycling tips**
5. **Prediction is saved to history**

---

## 📋 Example Output
**Prediction:**
```
Plastic (Recyclable) — 85%
```

**Guidance:**
- Bottles and containers
- Textile fibers (clothing/carpets)
- Packaging films and wraps

---

## 🏗️ Installation & Setup

### **Model Training (Python)**
```bash
pip install tensorflow keras matplotlib
jupyter notebook Waste_recycler.ipynb
```

### **Flutter App**
```bash
flutter pub get
flutter run
```

**Add assets to `pubspec.yaml`:**
```yaml
flutter:
  assets:
    - assets/recycle_classifier.tflite
    - assets/labels.txt
```

---

## 📌 Future Improvements
- Improve dataset balance & diversity
- Add more waste categories
- Support multi-label classification for mixed waste
- Cloud sync for history
- Multilingual recycling guidance

---

## 📜 License
This project is licensed under the MIT License.
