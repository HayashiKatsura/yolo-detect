import os

def save_image_and_label(img, label, img_path, label_path):
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    img.save(img_path)
    with open(label_path, 'w') as f:
        f.write(label + '\n')
