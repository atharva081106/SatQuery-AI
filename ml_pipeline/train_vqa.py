import argparse
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, TrainingArguments, Trainer
from datasets import RSVQADataset # Adjust to BigEarthNetDataset as needed

def main(args):
    model_id = "Salesforce/blip-vqa-base"
    
    print(f"Loading processor and model: {model_id}")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)
    
    print(f"Loading datasets from {args.data_dir}")
    train_dataset = RSVQADataset(data_dir=args.data_dir, processor=processor, split="train")
    eval_dataset = RSVQADataset(data_dir=args.data_dir, processor=processor, split="val")
    
    # Configure HuggingFace Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(), # Use mixed precision if on GPU
        save_total_limit=2,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VLM on Remote Sensing Data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to save the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    main(args)
