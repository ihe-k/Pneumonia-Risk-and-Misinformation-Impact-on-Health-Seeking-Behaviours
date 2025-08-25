# Pneumonia Detection from Chest X-Ray Images

A comprehensive machine learning project for detecting pneumonia from chest X-ray images using multiple ML algorithms and a Streamlit web interface.

## 🏥 Project Overview

This project implements an automated pneumonia detection system that analyzes chest X-ray images to classify them as either normal or pneumonia-positive. The system uses both traditional machine learning algorithms (Logistic Regression, XGBoost) and provides a user-friendly web interface for real-time inference.

## ✨ Features

- **Multi-Model Approach**: Combines Logistic Regression and XGBoost classifiers for robust predictions
- **Streamlit Web Interface**: Interactive web application for easy image upload and analysis
- **Real-time Inference**: Instant pneumonia detection results with confidence scores
- **Data Visualization**: Comprehensive analysis with confusion matrices, ROC curves, and performance metrics
- **Model Persistence**: Pre-trained models that can be loaded without retraining
- **Flexible Input**: Support for both local file uploads and URL-based image analysis

## 🏗️ Project Structure

```
pneumonia_final/
├── streamlit_inference.py      # Main Streamlit application
├── train_pneumonia.py          # Model training script
├── inference.py                # Standalone inference script
├── requirements.txt            # Python dependencies
├── chest_xray/                 # Dataset directory
│   ├── train/                 # Training images
│   ├── val/                   # Validation images
│   └── test/                  # Test images
├── saved_trained_model/        # Pre-trained models
│   ├── pneumonia_log_reg.pkl  # Logistic Regression model
│   └── pneumonia_xgb.pkl      # XGBoost model
└── Copy_of_train_pneumonia_model.ipynb  # Jupyter notebook for training
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pneumonia_final
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit, tensorflow, sklearn, xgboost; print('All packages installed successfully!')"
   ```

## 📊 Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset, which contains:
- **Training set**: Chest X-ray images for model training
- **Validation set**: Images for hyperparameter tuning
- **Test set**: Images for final model evaluation

The dataset includes:
- Normal chest X-rays
- Pneumonia-positive chest X-rays (bacterial and viral)

## 🎯 Usage

### Web Interface (Recommended)

1. **Launch the Streamlit app**
   ```bash
   streamlit run streamlit_inference.py
   ```

2. **Open your browser** and navigate to the provided local URL

3. **Upload an image** or provide an image URL

4. **View results** including:
   - Prediction (Normal/Pneumonia)
   - Confidence scores
   - Model performance metrics
   - Data visualizations

### Command Line Inference

```bash
python inference.py --image path/to/your/image.jpg
```

### Model Training

```bash
python train_pneumonia.py
```

## 🤖 Models

### 1. Logistic Regression
- **Type**: Linear classifier
- **Features**: Flattened image pixels
- **Use case**: Baseline model, interpretable results

### 2. XGBoost
- **Type**: Gradient boosting classifier
- **Features**: Flattened image pixels
- **Use case**: High-performance predictions

## 📈 Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🔧 Configuration

### Model Parameters

- **Image Size**: 150x150 pixels (configurable in `train_pneumonia.py`)
- **Batch Size**: 32 (for training)
- **Model Save Directory**: `saved_trained_model/`

### Streamlit Configuration

- **Page Title**: "Pneumonia Detection from Chest X-Ray Images"
- **Layout**: Wide layout for better visualization
- **Caching**: Model loading is cached for performance

## 🛠️ Technical Details

### Dependencies

**Core ML Libraries:**
- TensorFlow/Keras for image preprocessing
- Scikit-learn for traditional ML algorithms
- XGBoost for gradient boosting

**Web Framework:**
- Streamlit for the web interface

**Data Processing:**
- NumPy for numerical operations
- Pandas for data manipulation
- PIL for image processing

**Visualization:**
- Matplotlib and Seaborn for static plots
- Plotly for interactive visualizations

### Architecture

1. **Data Preprocessing**: Image resizing, normalization, and flattening
2. **Feature Extraction**: Pixel values as features
3. **Model Training**: Supervised learning on labeled data
4. **Model Persistence**: Joblib serialization for model storage
5. **Inference Pipeline**: Real-time prediction on new images
6. **Web Interface**: User-friendly Streamlit application

## 📝 API Reference

### Main Functions

- `load_pretrained_models(model_dir)`: Load saved ML models
- `predict_pneumonia(image_path, models)`: Perform inference on images
- `create_confusion_matrix(y_true, y_pred)`: Generate confusion matrix
- `plot_roc_curve(y_true, y_pred_proba)`: Plot ROC curve

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure models exist in `saved_trained_model/` directory
   - Check file permissions and paths

2. **Dependency Issues**
   - Verify all packages are installed: `pip list`
   - Check Python version compatibility

3. **Memory Issues**
   - Reduce batch size in training
   - Use smaller image dimensions

### Error Messages

- **"No module named 'sklearn'"**: Install scikit-learn
- **"Model not found"**: Check model file paths
- **"Image loading failed"**: Verify image format and path

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Chest X-Ray Images dataset contributors
- Streamlit development team
- Open-source ML community

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section
- Review the code comments for implementation details

## 🔮 Future Enhancements

- [ ] Deep learning models (CNN, ResNet)
- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] API endpoints for integration
- [ ] Mobile application
- [ ] Cloud deployment
- [ ] Real-time video analysis

---

**Note**: This project is for educational and research purposes. Always consult healthcare professionals for medical diagnoses.
