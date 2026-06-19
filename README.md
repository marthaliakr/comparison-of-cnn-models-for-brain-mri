# Analisis Perbandingan Performa Densenet121, Resnet50, Dan Efficientnetb0 Dengan Fine-Tuning Pada Klasifikasi Tumor Otak Menggunakan Citra MRI Dan Grad-CAM

Proyek ini membahas analisis perbandingan tiga arsitektur deep learning, yaitu DenseNet121, ResNet50, dan EfficientNetB0 untuk melakukan klasifikasi tumor otak berdasarkan citra MRI. Penelitian ini menerapkan metode transfer learning dan fine-tuning untuk mengevaluasi performa setiap model dalam melakukan klasifikasi multiclass pada citra MRI tumor otak.

Model dilatih dan diuji menggunakan dataset Brain Tumor MRI yang terdiri dari empat kelas, yaitu glioma, meningioma, pituitary, dan healthy brain MRI. Tahapan preprocessing citra yang dilakukan meliputi proses resizing, preprocessing input sesuai kebutuhan masing-masing model, serta augmentasi geometrik sederhana untuk meningkatkan kemampuan generalisasi model.

Terdapat enam skenario eksperimen yang dilakukan, yaitu:
- DenseNet121 Baseline
- DenseNet121 Fine-Tuning
- ResNet50 Baseline
- ResNet50 Fine-Tuning
- EfficientNetB0 Baseline
- EfficientNetB0 Fine-Tuning

Hasil eksperimen menunjukkan bahwa penerapan fine-tuning mampu meningkatkan performa klasifikasi pada seluruh arsitektur yang diuji. Dari seluruh model yang dibandingkan, **ResNet50 dengan Fine-Tuning** menghasilkan performa terbaik dengan hasil klasifikasi yang paling stabil.

Untuk meningkatkan interpretabilitas model, penelitian ini juga menerapkan **Grad-CAM (Gradient-weighted Class Activation Mapping)** untuk memvisualisasikan area penting pada citra MRI yang memengaruhi hasil prediksi model. Selain itu, sistem dikembangkan dalam bentuk aplikasi GUI interaktif yang memungkinkan pengguna untuk mengunggah citra MRI, melakukan proses klasifikasi, serta melihat visualisasi heatmap Grad-CAM.

## Fitur
- Klasifikasi tumor otak berdasarkan citra MRI
- Penerapan transfer learning dan fine-tuning
- Analisis perbandingan beberapa model CNN
- Visualisasi Grad-CAM untuk interpretasi model
- Aplikasi GUI interaktif
- Evaluasi menggunakan confusion matrix dan berbagai metrik performa

## Dataset
Sumber dataset:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Model Terbaik
Berdasarkan hasil pengujian, **ResNet50 dengan Fine-Tuning** menunjukkan performa terbaik dibandingkan model lainnya dengan tingkat akurasi yang lebih tinggi serta hasil klasifikasi yang lebih konsisten.

---

**Dibuat oleh:**  
Kelompok 4  
1. Nabila Yudhitya Larasati		(23083010086)
2. Salsabila Wardah			(23083010092)
3. Marthalia Kusumarima		(23083010100)
