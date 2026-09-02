import logging
import torch
import os

os.environ["HF_HOME"] = "d:\\sih26167\\.huggingface_cache"

from transformers import BlipProcessor, BlipForQuestionAnswering

logger = logging.getLogger(__name__)

class MLModels:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModels, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ML models on {self.device}...")
        
        # Priority 1: Check for locally fine-tuned remote sensing model
        finetuned_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "satquery_finetuned_model"))
        
        try:
            if os.path.exists(finetuned_path):
                logger.info(f"Loading Remote-Sensing-Adapted Fine-Tuned Model from {finetuned_path}...")
                self.vqa_processor = BlipProcessor.from_pretrained(finetuned_path)
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(finetuned_path).to(self.device)
                self.model_name = "SatQuery-RS-Adapted-v1.2 (Fine-tuned on Remote Sensing)"
                logger.info("Remote-Sensing-Adapted Model loaded successfully.")
            else:
                model_id = "Salesforce/blip-vqa-base"
                logger.info(f"Loading baseline model {model_id}...")
                self.vqa_processor = BlipProcessor.from_pretrained(model_id)
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
                self.model_name = "BLIP-VQA-Base (Baseline)"
                logger.info("Baseline BLIP VQA model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model ({e}), falling back to Salesforce/blip-vqa-base...")
            try:
                model_id = "Salesforce/blip-vqa-base"
                self.vqa_processor = BlipProcessor.from_pretrained(model_id)
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
                self.model_name = "BLIP-VQA-Base (Fallback)"
            except Exception as e2:
                logger.error(f"Failed to load any model: {e2}")
                self.vqa_processor = None
                self.vqa_model = None
                self.model_name = "None"

    def get_vqa_pipeline(self):
        return self.vqa_processor, self.vqa_model, self.device, getattr(self, "model_name", "Unknown")

ml_manager = MLModels()
