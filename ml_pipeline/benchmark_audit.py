import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from sat_seg_model import SatSegNet, SatelliteSegDataset

CLASS_NAMES = [
    "Background / Other",
    "Water Bodies",
    "Vegetation & Canopy",
    "Built-up Infrastructure",
    "Bare Ground & Soil",
    "Cloud Obscuration"
]

COLOR_MAP = {
    0: [40, 40, 40],       # Background: Dark Gray
    1: [245, 180, 25],     # Water: Luminous Cyan-Blue
    2: [40, 220, 90],      # Vegetation: Emerald Green
    3: [30, 140, 255],     # Built-up: Orange/Amber
    4: [140, 180, 210],    # Bare Soil: Sandy Brown
    5: [255, 255, 255]     # Cloud: White
}

def colorize_mask(mask_2d):
    h, w = mask_2d.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        colored[mask_2d == cls] = color
    return colored

def evaluate_model(data_dir, checkpoint_path, output_dir):
    print("============================================================")
    print("🛰️  SatQuery AI: Rigorous Remote Sensing Benchmark Audit")
    print(f"    Evaluating Checkpoint: {checkpoint_path}")
    print(f"    Dataset: {data_dir} (Test Split)")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    model = SatSegNet(n_channels=3, n_classes=6).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully. Trained at Epoch: {ckpt.get('epoch', 'N/A')}")
    
    # 2. Test Dataset & Loader
    test_dataset = SatelliteSegDataset(data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f"Evaluating {len(test_dataset)} unseen test satellite tiles...")
    
    num_classes = 6
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_pixels = 0
    correct_pixels = 0
    
    sample_images = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            p_flat = preds.cpu().numpy().flatten()
            m_flat = masks.cpu().numpy().flatten()
            
            for p, t in zip(p_flat, m_flat):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    confusion_matrix[t, p] += 1
                    
            correct_pixels += np.sum(p_flat == m_flat)
            total_pixels += len(p_flat)
            
            # Save visual samples for inspection
            if len(sample_images) < 4:
                for b in range(min(2, images.size(0))):
                    raw_img = (images[b].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    gt_color = colorize_mask(masks[b].cpu().numpy())
                    pred_color = colorize_mask(preds[b].cpu().numpy())
                    # Montage: Raw | Ground Truth | SatSegNet Prediction
                    row = np.hstack([cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR), gt_color, pred_color])
                    sample_images.append(row)
                    
    # Metrics Computation
    overall_accuracy = correct_pixels / float(max(1, total_pixels))
    
    per_class_results = {}
    ious = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = np.sum(confusion_matrix[:, cls]) - tp
        fn = np.sum(confusion_matrix[cls, :]) - tp
        
        union = tp + fp + fn
        iou = tp / float(union) if union > 0 else 1.0
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if union > 0:
            ious.append(iou)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            
        per_class_results[CLASS_NAMES[cls]] = {
            "class_id": cls,
            "iou": round(float(iou), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "total_ground_truth_pixels": int(tp + fn)
        }
        
    miou = float(np.mean(ious))
    mean_f1 = float(np.mean(f1_scores))
    mean_precision = float(np.mean(precisions))
    mean_recall = float(np.mean(recalls))
    
    results = {
        "mission": "SatQuery AI Remote Sensing Benchmark Audit",
        "checkpoint_evaluated": os.path.basename(checkpoint_path),
        "overall_accuracy": round(overall_accuracy, 4),
        "mean_iou (mIoU)": round(miou, 4),
        "mean_f1_score": round(mean_f1, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "total_test_samples": len(test_dataset),
        "total_pixels_evaluated": total_pixels,
        "per_class_metrics": per_class_results,
        "confusion_matrix": confusion_matrix.tolist()
    }
    
    # Save visual montage
    if sample_images:
        montage = np.vstack(sample_images[:3])
        # Add legend banner
        banner = np.zeros((40, montage.shape[1], 3), dtype=np.uint8)
        banner[:] = (15, 20, 25)
        cv2.putText(banner, "RAW SATELLITE TILE  |  GROUND TRUTH MASK  |  SATSEGNET PREDICTION", (20, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        final_montage = np.vstack([banner, montage])
        montage_path = os.path.join(output_dir, "test_predictions_montage.png")
        cv2.imwrite(montage_path, final_montage)
        print(f"Visual validation montage saved to: {montage_path}")
        
    # Save JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n============================================================")
    print(f"📊  AUDIT RESULTS SUMMARY (Unseen Test Satellite Tiles):")
    print(f"    Mean IoU (mIoU):       {miou * 100:.2f}%")
    print(f"    Pixel Accuracy:        {overall_accuracy * 100:.2f}%")
    print(f"    Macro F1-Score:        {mean_f1 * 100:.2f}%")
    print(f"    Water Bodies IoU:      {per_class_results['Water Bodies']['iou'] * 100:.2f}%")
    print(f"    Vegetation IoU:        {per_class_results['Vegetation & Canopy']['iou'] * 100:.2f}%")
    print(f"    Built-up IoU:          {per_class_results['Built-up Infrastructure']['iou'] * 100:.2f}%")
    print(f"    Full Audit Report:     {json_path}")
    print(f"============================================================")
    
    return results

if __name__ == "__main__":
    d_dir = r"d:\sih26167\ml_pipeline\dataset"
    ckpt = r"d:\sih26167\ml_pipeline\checkpoints\best_satsegnet.pth"
    out = r"d:\sih26167\ml_pipeline"
    evaluate_model(d_dir, ckpt, out)
