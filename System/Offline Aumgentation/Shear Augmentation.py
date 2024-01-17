import cv2
import numpy as np
import os
import zipfile

def apply_shear(image, shear_factor):
    rows, cols, _ = image.shape
    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])

    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return sheared_image

input_folder = "input_images"
output_folder = "sheared_images"

os.makedirs(output_folder, exist_ok=True)

sheared_image_paths = []

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        print(f'Processing file: {filename}')

        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        shear_factor = 0.4
        sheared_image = apply_shear(original_image, shear_factor)

        sheared_filename = f"sheared_{filename}"
        sheared_image_path = os.path.join(output_folder, sheared_filename)
        cv2.imwrite(sheared_image_path, cv2.cvtColor(sheared_image, cv2.COLOR_RGB2BGR))
        sheared_image_paths.append(sheared_image_path)

zip_filename = "COVID (0.4 + 20).zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for sheared_path in sheared_image_paths:
        zipf.write(sheared_path, os.path.basename(sheared_path))

print(f'Sheared images saved in: {zip_filename}')
