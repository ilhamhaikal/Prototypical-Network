from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

# Memuat gambar
image_path = 'C:/Users/PH315-53/ProtoNet/split_dataset/train/mentah/mentah_1.jpeg'
image = Image.open(image_path)  

# Mengubah ukuran gambar menjadi 224x224
image = image.resize((224, 224))

# Mengonversi gambar menjadi array numpy
pixel_values = np.array(image)

# Mengonversi array numpy menjadi DataFrame untuk visualisasi yang lebih baik
# Menampilkan channel pertama (R) sebagai contoh
pixel_df_r = pd.DataFrame(pixel_values[:, :, 0])  # Channel merah
pixel_df_g = pd.DataFrame(pixel_values[:, :, 1])  # Channel hijau
pixel_df_b = pd.DataFrame(pixel_values[:, :, 2])  # Channel biru

# Mengatur pandas untuk menampilkan semua 224 baris
pd.set_option('display.max_rows', 224)

# Menampilkan semua baris dari DataFrame untuk channel merah sebagai contoh
print("Channel R (Merah):")
print(pixel_df_r)

print("Channel G (Hijau):")
print(pixel_df_g)

print("Channel B (Biru):")
print(pixel_df_b)

# Menambahkan transformasi ke tensor dan normalisasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor_image = transform(image)

# Menampilkan bentuk dari tensor untuk memastikan transformasi berhasil
print(tensor_image.shape)

# Menampilkan nilai tensor yang sudah dinormalisasi
# Perlu mengubah tensor ke numpy untuk tampilan yang lebih baik
normalized_array = tensor_image.numpy()

# Menampilkan sebagian dari tensor yang sudah dinormalisasi (9x9) untuk melihat hasilnya
print("Nilai-nilai tensor setelah normalisasi (9x9) - Channel pertama (R):")
print(normalized_array[0, :9, :9])  # Menampilkan channel pertama (R)

print("Nilai-nilai tensor setelah normalisasi (9x9) - Channel kedua (G):")
print(normalized_array[1, :9, :9])  # Menampilkan channel kedua (G)

print("Nilai-nilai tensor setelah normalisasi (9x9) - Channel ketiga (B):")
print(normalized_array[2, :9, :9])  # Menampilkan channel ketiga (B)
