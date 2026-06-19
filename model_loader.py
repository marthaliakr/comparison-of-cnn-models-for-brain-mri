# IMPORT LIBRARY
from tensorflow.keras.models import load_model

# LOAD ALL MODELS
models = {
    # DenseNet121 Fine-Tuning
    "DenseNet121 FT":
    load_model("models/densenet_ft.h5"),

    # DenseNet121 Baseline
    "DenseNet121 Non-FT":
    load_model("models/densenet_non_ft.h5"),

    # ResNet50 Fine-Tuning
    "ResNet50 FT":
    load_model("models/resnet_ft.h5"),

    # ResNet50 Baseline
    "ResNet50 Non-FT":
    load_model("models/resnet_non_ft.h5"),

    # EfficientNetB0 Fine-Tuning
    "EfficientNet FT":
    load_model("models/efficientnet_ft.h5"),

    # EfficientNetB0 Baseline
    "EfficientNet Non-FT":
    load_model("models/efficientnet_non_ft.h5")
}