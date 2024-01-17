import cv2
import numpy as np
import os
import zipfile

def adjust_brightness(image, brightness_factor):
    adjusted_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return adjusted_image

input_folder = "input_images"
output_folder = "brightness_adjusted_images"

os.makedirs(output_folder, exist_ok=True)

brightness_adjusted_image_paths = []

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        print(f'Processing file: {filename}')

        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        brightness_factor = 1.5 
        adjusted_image = adjust_brightness(original_image, brightness_factor)

        adjusted_filename = f"brightness_adjusted_{filename}"
        adjusted_image_path = os.path.join(output_folder, adjusted_filename)
        cv2.imwrite(adjusted_image_path, cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR))
        brightness_adjusted_image_paths.append(adjusted_image_path)

zip_filename = "1.5 BRIGHTNESS.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for adjusted_path in brightness_adjusted_image_paths:
        zipf.write(adjusted_path, os.path.basename(adjusted_path))

print(f'Brightness adjusted images saved in: {zip_filename}')
