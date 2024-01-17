import cv2
import numpy as np
import os
import zipfile

def add_noise(image, noise_factor):
    row, col, _ = image.shape
    mean = 0
    sigma = noise_factor
    gauss = np.random.normal(mean, sigma, (row, col, 3))
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

input_folder = "input_images"
output_folder = "noisy_images"

os.makedirs(output_folder, exist_ok=True)

noisy_image_paths = []

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        print(f'Processing file: {filename}')

        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        noise_factor = 20 
        noisy_image = add_noise(original_image, noise_factor)

        noisy_filename = f"noisy_{filename}"
        noisy_image_path = os.path.join(output_folder, noisy_filename)
        cv2.imwrite(noisy_image_path, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
        noisy_image_paths.append(noisy_image_path)

zip_filename = "COVID NOISE (20).zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for noisy_path in noisy_image_paths:
        zipf.write(noisy_path, os.path.basename(noisy_path))

print(f'Noisy images saved in: {zip_filename}')
