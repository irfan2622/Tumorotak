import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model('Model1.h5')

# Class names sesuai urutan pada training
class_names = ['no', 'yes']

# Preprocessing function
def preprocess_image(image):
    # Pastikan gambar memiliki 3 channel
    if len(image.shape) == 2:  # Jika gambar grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:  # Jika gambar grayscale dengan channel eksplisit
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Ubah ukuran gambar ke 224x224 sesuai dengan input model
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0  # Normalisasi pixel (0-1)
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

# Streamlit app
def main():
    st.title("Brain Tumor Detection")
    st.write("Unggah gambar MRI untuk mendeteksi tumor otak.")

    uploaded_file = st.file_uploader("Unggah gambar MRI", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar yang diunggah
        image = Image.open(uploaded_file).convert("RGB")  # Pastikan gambar selalu RGB
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        # Preprocessing gambar
        image_array = np.array(image)
        preprocessed_image = preprocess_image(image_array)

        # Prediksi menggunakan model
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** {predicted_class}")
        st.write(f"**Kepercayaan:** {confidence:.2f}%")

if __name__ == "__main__":
    main()
