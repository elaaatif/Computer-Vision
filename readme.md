# Faster R-CNN for Wildfire Detection using Satellite Images

## Overview
This project focuses on detecting  wildfire-affected areas using satellite  imagery. Wildfires pose a significant threat to ecosystems, human lives, and infrastructure, making early detection and accurate mapping of affected areas critical for disaster management. By leveraging advanced deep learning techniques, this project aims to develop a robust system for identifying and localizing wildfire-affected regions with high precision.

The project is divided into two main components: **object detection** . For object detection, we use the **Faster R-CNN** (Region-based Convolutional Neural Network) model, a state-of-the-art deep learning model known for its accuracy and efficiency in detecting objects in complex scenes.

---

## Utility
The utility of this project lies in its ability to:
1. **Early Detection**: Quickly identify wildfire-affected areas, enabling faster response times for emergency services.
2. **Accurate Mapping**: Provide precise localization of affected regions, which is crucial for resource allocation and damage assessment.
3. **Automation**: Reduce reliance on manual inspection, saving time and resources while improving scalability.
4. **Disaster Management**: Assist in planning evacuation routes, allocating firefighting resources, and assessing environmental impact.

This system can be integrated into disaster management frameworks, enabling authorities to make data-driven decisions during wildfire events.

---

## Techniques Used
The project employs the following techniques and methodologies:

### 1. **Faster R-CNN for Object Detection**
   - **Faster R-CNN** is a deep learning model designed for object detection tasks. It combines a Region Proposal Network (RPN) with a detection network to accurately localize objects in images.
   - The model is fine-tuned on annotated wildfire datasets to detect wildfire-affected areas in satellite and UAV imagery.
   - Pre-trained models like **ResNet-50** and **ResNet-101** are used as backbones for feature extraction, ensuring high accuracy and efficiency.

### 2. **Data Preprocessing**
   - Multiple datasets from Roboflow are merged to create a comprehensive training dataset, ensuring the model generalizes well to diverse scenarios.

### 3. **Evaluation Metrics**
   - The model's performance is evaluated using metrics such as **Precision**, **Recall**, **F1-Score**, and **mAP** (mean Average Precision). These metrics ensure the model is both accurate and reliable.

### 5. **Deep Learning Frameworks**
   - The project is built using **PyTorch**, a popular deep learning framework known for its flexibility and ease of use.
   - Pre-trained models and libraries like **torchvision** are utilized to streamline development and improve performance.

---

## Datasets
The project uses multiple datasets from Roboflow, which include satellite and UAV imagery of wildfires. These datasets were preprocessed and merged into a single, comprehensive dataset to ensure the model is trained on diverse and representative data. The merged dataset includes:
- **Wildfire Dataset 1**: Preprocessed with auto-orientation and resized to 640x640.
- **Wildfire Dataset 2**: Preprocessed with auto-orientation, resized to 416x416, and augmented with horizontal and vertical flips.
- **Wildfire Dataset 3**: Preprocessed with auto-orientation and resized to 640x640.
- **Fire Forest Dataset**: No preprocessing or augmentations applied.
- **Fire Forest Gen Dataset**: Resized to 800x800.
- **Volcano Dataset**: Preprocessed with auto-orientation, resized to 416x416, and augmented with flips, blur, and noise.
- **Wildfire Detection Satellite Dataset**: Preprocessed with auto-orientation, resized to 640x640, and augmented with tiling, flips, and saturation adjustments.

By merging these datasets, we ensure the model is robust and capable of handling various wildfire scenarios.

---

## Key Features
- **High Accuracy**: Leverages state-of-the-art deep learning models to achieve precise detection and segmentation of wildfire-affected areas.
- **Scalability**: Can process large volumes of satellite and UAV imagery, making it suitable for real-world applications.
- **Flexibility**: Supports both object detection and segmentation, allowing users to choose the approach that best suits their needs.
- **Open-Source**: Built using open-source tools and frameworks, making it accessible to researchers and developers worldwide.

---

## Applications
This project has several real-world applications, including:
1. **Disaster Response**: Enabling rapid response to wildfires by providing accurate and timely information.
2. **Environmental Monitoring**: Tracking the spread of wildfires and assessing their impact on ecosystems.
3. **Urban Planning**: Identifying high-risk areas and implementing preventive measures.
4. **Research and Development**: Serving as a foundation for further research in wildfire detection and management.

---

## Future Enhancements
The project can be extended in the following ways:
1. **Real-Time Detection**: Implementing real-time processing capabilities for live satellite and UAV feeds.
2. **Multi-Spectral Analysis**: Incorporating multi-spectral and thermal imagery to improve detection accuracy.
3. **Integration with GIS**: Combining the system with Geographic Information Systems (GIS) for enhanced visualization and analysis.
4. **Deployment on Edge Devices**: Optimizing the model for deployment on edge devices, enabling on-site analysis in remote areas.

---
