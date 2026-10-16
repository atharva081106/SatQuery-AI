import argparse
import json
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from tqdm import tqdm
import evaluate

def main(args):
    print(f"Loading model from {args.model_path}")
    processor = BlipProcessor.from_pretrained(args.model_path)
    model = BlipForQuestionAnswering.from_pretrained(args.model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load evaluation dataset (e.g., ISRO/SAC format or RSVQA format)
    print(f"Loading evaluation dataset from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Image-Shuffle Ablation Mode (per docs/eval_harness_spec.md)
    if args.shuffle_images:
        import random
        print(f"[IMAGE-SHUFFLE ABLATION] Shuffling images across queries (Seed: {args.seed})...")
        random.seed(args.seed)
        all_imgs = [item["image_path"] for item in test_data]
        shuffled_imgs = all_imgs.copy()
        # Ensure derangement / mismatch
        while True:
            random.shuffle(shuffled_imgs)
            if len(all_imgs) <= 1 or any(s != o for s, o in zip(shuffled_imgs, all_imgs)):
                break
        for idx, item in enumerate(test_data):
            item["image_path"] = shuffled_imgs[idx]

    predictions = []
    references = []
    
    print("Running evaluation...")
    for item in tqdm(test_data):
        img_path = item["image_path"]
        query = item["question"]
        ground_truth = item["answer"]
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, query, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
                
            pred_answer = processor.decode(out[0], skip_special_tokens=True).lower().strip()
            gt_answer = ground_truth.lower().strip()
            
            predictions.append(pred_answer)
            references.append(gt_answer)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # Calculate metrics
    # Simple Exact Match Accuracy
    correct = sum([1 for p, r in zip(predictions, references) if p == r])
    accuracy = correct / len(predictions) if predictions else 0.0
    
    # You can also plug in BLEU/ROUGE using HuggingFace Evaluate
    print(f"\n--- Evaluation Results ---")
    print(f"Total Evaluated: {len(predictions)}")
    print(f"Exact Match Accuracy: {accuracy * 100:.2f}%")
    if args.shuffle_images:
        print("[IMAGE-SHUFFLE ABLATION] Performance reflects language prior under visual mismatch.")
    
    # Save results
    results = {
        "accuracy": accuracy,
        "is_image_shuffled": args.shuffle_images,
        "details": [{"pred": p, "gt": r} for p, r in zip(predictions, references)]
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned VLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the JSON file containing test data (format: [{'image_path': '...', 'question': '...', 'answer': '...'}])")
    parser.add_argument("--shuffle-images", dest="shuffle_images", action="store_true", help="Execute Image-Shuffle Ablation (per docs/eval_harness_spec.md)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image shuffling")
    args = parser.parse_args()
    main(args)
