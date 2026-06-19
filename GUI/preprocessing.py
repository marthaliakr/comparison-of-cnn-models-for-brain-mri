# =========================
# IMPORT LIBRARY
# =========================
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

# =========================
# IMAGE SIZE
# =========================
IMG_SIZE = (224, 224)

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(uploaded_image):
    # Membuka gambar
    image = Image.open(
        uploaded_image
    ).convert("RGB")
    # Resize gambar
    image = image.resize(
        IMG_SIZE
    )
    # Convert ke NumPy
    image_array = np.array(
        image,
        dtype=np.float32
    )
    # Menambahkan batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )
    return image_array, image

# =========================
# PREPROCESS SESUAI MODEL
# =========================
def preprocess_for_model(
    image_array,
    model_name
):
    # DenseNet121
    if "DenseNet" in model_name:
        image_array = densenet_preprocess(
            image_array
        )

    # ResNet50
    elif "ResNet" in model_name:
        image_array = resnet_preprocess(
            image_array
        )

    # EfficientNetB0
    elif "EfficientNet" in model_name:
        image_array = efficientnet_preprocess(
            image_array
        )

    return image_array