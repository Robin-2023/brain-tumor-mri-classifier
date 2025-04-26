# Brain Tumor MRI Classifier

![Brain Tumor Classification Demo](https://raw.githubusercontent.com/Robin-2023/brain-tumor-mri-classifier/main/assets/header.png)
A deep learning-powered web application for detecting and classifying brain tumors from MRI scans with explainable AI features.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

## Features

**Comprehensive Analysis Pipeline**  
The application provides end-to-end analysis of brain MRI scans, from image upload to classification with visual explanations. The multi-branch architecture combines different neural network approaches to improve reliability.

Key capabilities:
- **Three-class tumor detection**: Glioma, Meningioma, Pituitary
- **Model interpretability**: Grad-CAM heatmaps show decision regions
- **Confidence metrics**: Percentage-based prediction scores
- **Consistency checks**: Multiple inference verification
- **Device optimization**: Automatic GPU/CPU switching

## Installation

**Environment Setup**  
The application requires Python 3.7+ with PyTorch. Using a virtual environment is strongly recommended to avoid dependency conflicts.

Step-by-step setup:

### 1. Clone repository
```bash
git clone https://github.com/yourusername/brain-tumor-classifier.git
cd brain-tumor-classifier
```
### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Critical components:
- Core: `streamlit torch torchvision`
- Visualization: `opencv-python matplotlib`
- Image processing: `Pillow numpy`

## Usage

**Interactive Diagnostic Interface**  
The web interface guides users through a simple workflow while handling complex model operations in the background. The system is designed for ease of use while providing advanced options for technical users.

Workflow overview:
1. **Launch application**:
   ```bash
   streamlit run app.py
   ```
2. **Upload MRI scan** (JPG/PNG format)
3. **View results**:
   - Classification prediction
   - Confidence percentages
   - Attention heatmap overlay
4. **Optional**: Run consistency tests

## Model Architecture

**Innovative Multi-Branch Design**  
The model combines three parallel feature extraction pathways to leverage different aspects of convolutional and attention-based learning:

- **Branch 1 (EfficientNet)**:
  - Modified EfficientNet-B0 backbone
  - Custom grayscale adaptation
  - 512-unit feature embedding

- **Branch 2 (Window Attention)**:
  - 4x4 strided convolutions
  - Layer normalization
  - Adaptive spatial pooling

- **Branch 3 (Transformer)**:
  - 16x16 patch embedding
  - Learnable class token
  - Positional encodings

**Fusion & Classification**:

`
``mermaid

graph LR
    B1 -->|512-d| Fusion
    B2 -->|512-d| Fusion
    B3 -->|512-d| Fusion
    Fusion -->|1536-d| FC128
    FC128 -->|128-d| FC3
```

## Technical Details

**Robust Implementation**  
The system incorporates multiple reliability features to ensure consistent performance across different hardware environments.

Key technical aspects:
- **Preprocessing**:
  - Fixed 256×256 resolution
  - Grayscale conversion
  - Tensor normalization
- **Inference**:
  - Automatic mixed precision
  - CUDA/CPU fallback
  - Memory cleanup protocols
- **Reproducibility**:
  - Seed locking (42)
  - Deterministic algorithms
  - Gradient checkpointing

## Limitations

**Application Boundaries**  
While the model shows strong performance on benchmark datasets, real-world deployment requires understanding of its constraints.

Current limitations:
- **Data Scope**:
  - Trained on specific MRI protocols
  - Limited to three tumor classes
  - Performance varies by tumor stage

- **Technical Constraints**:
  - Requires clear axial-view MRIs
  - Sensitive to image artifacts
  - Batch processing not implemented

- **Clinical Relevance**:
  - Not validated for pediatric cases
  - Limited post-treatment samples
  - No multi-modal fusion (CT/PET)

## Disclaimer

**Model File Preparation**  
The model requires MRI images from the Brain Tumor Classification dataset, which is available on [Kaggle](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/data). After training on this dataset, generate the model file named **multi_branch_brain_tumor_model.pth** and place it in the same directory to run the Streamlit app.

**Research Use Only**  
This application represents experimental technology and should not be used for clinical decision-making without extensive further validation.

Critical notices:
- **Not a medical device**: Outputs are probabilistic predictions
- **Experimental nature**: False positives/negatives expected
- **Ethical considerations**: Potential biases in training data

**Required Actions**:
1. Always consult licensed radiologists
2. Disclose AI-assisted nature of results
3. Validate findings with clinical protocols

For research collaborations or technical inquiries, please contact: [haquerobin161@gmail.com](mailto:haquerobin161@gmail.com)
