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
    
    # Save results
    results = {
        "accuracy": accuracy,
        "details": [{"pred": p, "gt": r} for p, r in zip(predictions, references)]
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned VLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the JSON file containing test data (format: [{'image_path': '...', 'question': '...', 'answer': '...'}])")
    args = parser.parse_args()
    main(args)
