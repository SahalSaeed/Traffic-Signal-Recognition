# Traffic Sign Classification Using Classical Digital Image Processing

A complete implementation of traffic sign classification using only classical digital image processing techniques - no machine learning or pre-trained models involved.

## 📋 Project Overview

This project implements an end-to-end traffic sign classifier that operates entirely using classical image processing techniques. The system identifies traffic sign classes from cropped images using color, shape, and geometric features through a rule-based approach.

### Key Features
- **Classical DIP Only**: No machine learning - purely rule-based classification
- **Complete Pipeline**: From raw images to final classification results
- **Manual Implementation**: All algorithms implemented from scratch using NumPy
- **Multi-class Classification**: Handles 6-8 different traffic sign classes
- **Performance Evaluation**: Comprehensive metrics and confusion matrix analysis

## 🗂️ Project Structure

```
├── project.ipynb          # Main implementation notebook
├── metrics.txt           # Overall accuracy and class-wise metrics
├── predictions.csv       # Predicted vs ground-truth labels
└── README.md            # This file
```

## 🚀 Implementation Pipeline

### 1. **Image Preprocessing & Filtering**
- Mean Filter (3×3)
- Gaussian Filter with standard deviation
- Median Filter
- Adaptive Median Filter
- Unsharp Masking/High-Boost Filtering

### 2. **Color Space Conversion & Segmentation**
- Manual RGB to HSV conversion
- Color-based segmentation:
  - **Red signs**: Hue [0-15°] or [165-180°]
  - **Blue signs**: Hue [100-130°]
- Morphological operations (erosion, dilation, opening)
- Connected component filtering and hole filling

### 3. **Edge Detection**
- Complete Canny edge detection implementation:
  - Sobel gradient computation
  - Non-maximum suppression
  - Double thresholding and edge tracking

### 4. **Geometric Normalization**
- Affine transformations for rotation and scaling
- Perspective correction (optional)
- Standardization to 200×200 pixels

### 5. **Feature Extraction**
- **Corner Count**: Harris Corner Detection
- **Circularity**: C = 4π×Area/(Perimeter)²
- **Aspect Ratio**: Width/Height of bounding box
- **Extent**: Region area to bounding box ratio
- **Average Hue**: Color feature extraction

### 6. **Rule-Based Classification**
- If-else decision tree using extracted features
- Color and shape feature combinations
- Handles visually similar signs (Stop vs. Yield)

## 📊 Dataset Information

- **Classes**: 6-8 traffic sign categories
- **Images per Class**: ~100 images
- **Total Dataset**: 600-800 images
- **Format**: Pre-cropped PNG images
- **Labels**: Integer ClassId mapping

## 🔧 Technical Requirements

### Libraries Used
- **NumPy**: Core array operations and manual algorithm implementation
- **OpenCV/PIL**: Image loading only (no processing operations)
- **Matplotlib**: Visualization and confusion matrix plotting
- **Pandas**: Data handling for CSV files

### Key Constraints
- No machine learning libraries (scikit-learn, TensorFlow, etc.)
- No built-in image processing functions beyond basic I/O
- All algorithms implemented manually using NumPy

## 📈 Results & Evaluation

### Output Files
- **`predictions.csv`**: Contains filename, ground truth, predicted labels, and correctness
- **`metrics.txt`**: Overall accuracy and per-class precision/recall/F1-score
- **Confusion Matrix**: Visual representation of classification performance

### Metrics Calculated
- Overall Classification Accuracy
- Per-class Precision and Recall
- Confusion Matrix Analysis
- Class-wise Performance Breakdown

## 🚦 Traffic Sign Classes

The classifier handles multiple traffic sign categories including:
- Speed Limit signs (various speeds)
- Regulatory signs (Stop, Yield, etc.)
- Warning signs
- Informational signs

*Note: Specific classes selected are detailed in the implementation notebook*

## 💻 Usage

1. **Setup Environment**:
   ```bash
   pip install numpy opencv-python pillow matplotlib pandas
   ```

2. **Run Classification**:
   ```bash
   jupyter notebook project.ipynb
   ```

3. **View Results**:
   - Check `metrics.txt` for performance summary
   - Examine `predictions.csv` for detailed predictions
   - Analyze confusion matrix visualization

## 🎯 Key Achievements

- **Pure Classical Approach**: No deep learning or pre-trained models
- **Complete Implementation**: Every algorithm built from scratch
- **Robust Pipeline**: Handles real-world traffic sign variations
- **Interpretable Results**: Rule-based decisions are fully explainable
- **Comprehensive Evaluation**: Detailed performance analysis

## 🔍 Algorithm Details

### Color Segmentation Thresholds
- **Red Detection**: HSV ranges optimized for traffic sign red
- **Blue Detection**: HSV ranges for regulatory blue signs
- **Adaptive Thresholding**: Handles varying lighting conditions

### Shape Analysis Features
- **Circularity**: Distinguishes circular from triangular/rectangular signs
- **Aspect Ratio**: Separates square from rectangular signs
- **Corner Detection**: Identifies geometric shape characteristics

### Classification Rules
Decision tree logic combining:
- Color dominance (red/blue/yellow classification)
- Shape characteristics (circular/triangular/rectangular)
- Size and proportion features
- Texture and edge density metrics

## 📝 Academic Context

This project demonstrates the power and limitations of classical digital image processing techniques in computer vision tasks. By implementing everything from scratch, it provides deep insights into:

- Fundamental image processing operations
- Feature engineering for classification
- Trade-offs between classical and modern approaches
- Importance of domain knowledge in rule-based systems

## 🏆 Project Outcomes

Successfully demonstrates that classical image processing techniques can achieve reasonable performance on structured classification tasks like traffic sign recognition, while highlighting the challenges that led to the adoption of machine learning approaches in computer vision.

---

*This project was completed as part of a Digital Image Processing course, focusing on classical techniques and manual implementation of core algorithms.*