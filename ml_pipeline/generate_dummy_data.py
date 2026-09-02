import os
import json
from PIL import Image
import numpy as np

def create_dummy_dataset(base_dir, split, num_images):
    images_dir = os.path.join(base_dir, split, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    qa_pairs = []
    
    for i in range(num_images):
        # Create a random noisy image simulating a satellite image
        img_name = f"image_{i:03d}.tif"
        img_path = os.path.join(images_dir, img_name)
        
        # 256x256 random RGB image
        random_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(random_image)
        img.save(img_path)
        
        # Create dummy QA pair
        qa_pairs.append({
            "image_id": img_name,
            "question": "Is there a water body visible in this image?",
            "answer": "yes" if i % 2 == 0 else "no"
        })
        
    qa_path = os.path.join(base_dir, split, f"{split}_qa_pairs.json")
    with open(qa_path, 'w') as f:
        json.dump(qa_pairs, f, indent=4)
        
    print(f"Generated {num_images} dummy images and QA pairs for {split} split at {base_dir}")

if __name__ == "__main__":
    base_dir = "./dummy_dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    print("Generating Dummy Dataset for Training and Evaluation...")
    create_dummy_dataset(base_dir, "train", 10)
    create_dummy_dataset(base_dir, "val", 5)
    
    # Generate ISRO test set format for evaluation script
    isro_test_data = []
    val_images_dir = os.path.join(base_dir, "val", "images")
    for img_name in os.listdir(val_images_dir):
        isro_test_data.append({
            "image_path": os.path.join(val_images_dir, img_name),
            "question": "Is there a water body visible in this image?",
            "answer": "yes"
        })
        
    with open("isro_evaluation_set.json", "w") as f:
        json.dump(isro_test_data, f, indent=4)
    print("Generated isro_evaluation_set.json")
