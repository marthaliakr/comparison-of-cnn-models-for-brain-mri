# =========================
# IMPORT LIBRARY
# =========================

import tensorflow as tf      # Framework deep learning
import numpy as np           # Operasi numerik dan array
import cv2                   # Pengolahan citra dan visualisasi heatmap


# =========================
# IDENTIFIKASI BACKBONE
# =========================

def get_gradcam_config(model_name):

    # Menentukan nama backbone CNN dan layer konvolusi terakhir
    # yang akan digunakan sebagai sumber feature map pada Grad-CAM
    if "ResNet" in model_name:
        return {
            "backbone_name": "resnet50",
            "last_conv_layer_name": "conv5_block3_out"
        }

    elif "DenseNet" in model_name:
        return {
            "backbone_name": "densenet121",
            "last_conv_layer_name": "conv5_block16_concat"
        }
    
    elif "EfficientNet" in model_name:
        return {
            "backbone_name": "efficientnetb0",
            "last_conv_layer_name": "top_conv"
        }

    else:
        raise ValueError(
            f"Model tidak dikenali: {model_name}"
        )


# =========================
# MEMBUAT HEATMAP GRAD-CAM
# =========================

def make_gradcam_heatmap(
    img_array,
    model,
    backbone_name,
    last_conv_layer_name
):

    # Mengambil backbone CNN dari model yang telah dilatih
    base_model = model.get_layer(
        backbone_name
    )

    # Mengambil layer konvolusi terakhir sebagai sumber feature map
    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    # Membuat feature extractor untuk menghasilkan output
    # dari layer konvolusi terakhir
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Mendefinisikan input classifier sesuai ukuran output
    # layer konvolusi terakhir
    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    # Mengambil seluruh layer classifier setelah backbone
    # Cari posisi backbone di dalam model
    backbone_index = model.layers.index(
        base_model
    )

    # Ambil seluruh layer setelah backbone
    classifier_layers = model.layers[
        backbone_index + 1:
    ]
    
    # Menyusun ulang classifier untuk menghasilkan prediksi kelas
    for layer in classifier_layers:
        x = layer(
            x,
            training=False
        )
    classifier_model = tf.keras.Model(
        classifier_input,
        x
    )

    # Menghitung gradien menggunakan GradientTape
    with tf.GradientTape() as tape:
        # Menghasilkan feature map dari layer konvolusi terakhir
        conv_outputs = feature_extractor(
            img_array,
            training=False
        )
        # Memantau feature map agar dapat dihitung gradiennya
        tape.watch(conv_outputs)
        # Menghasilkan prediksi kelas
        predictions = classifier_model(
            conv_outputs,
            training=False
        )
        # Mengambil indeks kelas dengan probabilitas tertinggi
        pred_index = tf.argmax(
            predictions[0]
        )
        # Mengambil skor prediksi dari kelas target
        class_channel = predictions[
            :,
            pred_index
        ]
    # Menghitung gradien skor kelas terhadap feature map
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )
    # Melakukan global average pooling pada gradien
    # untuk memperoleh bobot tiap feature map
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )
    conv_outputs = conv_outputs[0]

    # Mengalikan feature map dengan bobot gradien
    # lalu menjumlahkannya untuk membentuk heatmap
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Menghilangkan nilai negatif
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalisasi heatmap ke rentang 0–1
    heatmap /= (
        tf.reduce_max(
            heatmap
        ) + 1e-8
    )
    heatmap_np = heatmap.numpy()

    # Mengembalikan heatmap dalam bentuk array NumPy
    return heatmap.numpy()


# =========================
# OVERLAY HEATMAP
# =========================

def overlay_gradcam(
    original_image,
    heatmap
):

    # Mengubah gambar menjadi array NumPy
    img = np.array(
        original_image
    )

    # Menyesuaikan ukuran heatmap dengan ukuran gambar asli
    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    # Mengubah heatmap menjadi rentang 0–255
    heatmap = np.uint8(
        255 * heatmap
    )

    # Memberikan warna menggunakan colormap JET
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # TAMBAHAN INI
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        img,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return heatmap_color, overlay