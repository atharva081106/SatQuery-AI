import os
import json
from PIL import Image
from torch.utils.data import Dataset

class BigEarthNetDataset(Dataset):
    """
    Standard PyTorch Dataset for BigEarthNet.
    Assumes the dataset is structured with imagery and a JSON file containing metadata/labels.
    """
    def __init__(self, data_dir, processor, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.processor = processor
        
        # Load labels (dummy mapping here; adjust based on actual BigEarthNet format)
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
            
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.tif', '.png', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Find corresponding labels
        record = next((item for item in self.metadata if item["filename"] == img_name), None)
        labels = record["labels"] if record else []
        
        # We can format BigEarthNet for VLM by creating a pseudo-prompt:
        # e.g., "What is the land cover in this image?" -> ", ".join(labels)
        query = "Describe the land cover and major objects visible in this image."
        answer = ", ".join(labels) if labels else "Unknown land cover."
        
        # Prepare for BLIP or LLaVA
        encoding = self.processor(image, query, return_tensors="pt", padding="max_length", truncation=True)
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # The labels for computing loss (target answer)
        labels_encoding = self.processor.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True)
        encoding["labels"] = labels_encoding["input_ids"].squeeze(0)
        
        return encoding


class RSVQADataset(Dataset):
    """
    Dataset loader for RSVQA (Remote Sensing Visual Question Answering).
    """
    def __init__(self, data_dir, processor, split="train"):
        self.data_dir = os.path.join(data_dir, split, "images")
        self.processor = processor
        
        qa_path = os.path.join(data_dir, split, f"{split}_qa_pairs.json")
        with open(qa_path, 'r') as f:
            self.qa_data = json.load(f)

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        item = self.qa_data[idx]
        img_path = os.path.join(self.data_dir, item["image_id"])
        image = Image.open(img_path).convert("RGB")
        
        query = item["question"]
        answer = item["answer"]
        
        encoding = self.processor(image, query, return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        labels_encoding = self.processor.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True)
        encoding["labels"] = labels_encoding["input_ids"].squeeze(0)
        
        return encoding
