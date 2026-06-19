# Import Library
import numpy as np

# Class Label
class_names = [
    "glioma",
    "healthy",
    "meningioma",
    "pituitary"
]

# Predict All Models 
def predict_all_models(
    models,
    image_array,
    preprocess_function
):

    # Menyimpan hasil prediksi
    results = []

    # Loop semua model
    for model_name, model in models.items():

        # Preprocessing sesuai model
        processed_image = preprocess_function(
            image_array.copy(),
            model_name
        )

        # Prediksi model
        prediction = model.predict(
            processed_image
        )

        # Ambil index probabilitas tertinggi
        predicted_index = np.argmax(
            prediction
        )

        # Ambil nama kelas
        predicted_class = class_names[
            predicted_index
        ]

        # Confidence score
        confidence = np.max(
            prediction
        ) * 100

        # Simpan hasil
        results.append({
            "Model": model_name,
            "Prediction": predicted_class,
            "Confidence":
            round(confidence, 2)
        })

    return results