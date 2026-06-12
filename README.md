# comparison-of-cnn-models-for-brain-mri
Brain tumor MRI classification using DenseNet121, ResNet50, and EfficientNetB0 with fine-tuning and Grad-CAM visualization.

This project presents a comparative analysis of three deep learning architectures DenseNet121, ResNet50, and EfficientNetB0 for brain tumor classification using MRI images. The study applies transfer learning and fine-tuning strategies to evaluate each model’s performance in multiclass classification of brain tumors.

The models were trained and evaluated on a Brain Tumor MRI dataset consisting of four classes: glioma, meningioma, pituitary, and healthy brain MRI images. Image preprocessing techniques such as resizing, model-specific preprocessing input, and light geometric augmentation were applied to improve model generalization.

Six experimental scenarios were conducted:
- DenseNet121 Baseline
- DenseNet121 Fine-Tuning
- ResNet50 Baseline
- ResNet50 Fine-Tuning
- EfficientNetB0 Baseline
- EfficientNetB0 Fine-Tuning

Experimental results show that fine-tuning significantly improves classification performance across all architectures. Among all models, ResNet50 Fine-Tuning achieved the best overall performance with:
- Test Accuracy: 97%
- Precision: 97%
- Recall: 97%
- F1-Score: 97%
- Test Loss: 0.10

To enhance interpretability, Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented to visualize important regions influencing model predictions. The system was further integrated into an interactive GUI application that allows users to upload MRI images, perform classification, and visualize Grad-CAM heatmaps.

## Features
- Brain tumor MRI classification
- Transfer learning & fine-tuning
- Comparative CNN model analysis
- Grad-CAM visualization
- Interactive GUI application
- Confusion matrix and evaluation metrics

Dataset Source:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Best Model
ResNet50 with Fine-Tuning demonstrated the highest performance and the most stable classification results among all tested architectures.
