import logging
import torch
import os

# Only set local windows cache if on windows and not provided by environment
if os.name == "nt" and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "d:\\sih26167\\.huggingface_cache"

logger = logging.getLogger(__name__)

class MLModels:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModels, cls).__new__(cls)
            cls._instance.vqa_processor = None
            cls._instance.vqa_model = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.model_name = "SatQuery-RS-Adapted-v1.2 (Fine-tuned)"
            cls._instance._loaded = False
        return cls._instance

    def _load_if_needed(self):
        if self._loaded:
            return
            
        logger.info(f"Loading ML models lazily on {self.device}...")
        finetuned_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "satquery_finetuned_model"))
        
        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            if os.path.exists(finetuned_path):
                logger.info(f"Loading Remote-Sensing-Adapted Model from {finetuned_path}...")
                self.vqa_processor = BlipProcessor.from_pretrained(finetuned_path)
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(finetuned_path).to(self.device)
                self.model_name = "SatQuery-RS-Adapted-v1.2 (Fine-tuned on Remote Sensing)"
            else:
                model_id = "Salesforce/blip-vqa-base"
                logger.info(f"Loading baseline model {model_id}...")
                self.vqa_processor = BlipProcessor.from_pretrained(model_id)
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
                self.model_name = "BLIP-VQA-Base (Baseline)"
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model ({e}), attempting fallback...")
            try:
                from transformers import BlipProcessor, BlipForQuestionAnswering
                model_id = "Salesforce/blip-vqa-base"
                self.vqa_processor = BlipProcessor.from_pretrained(model_id)
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
                self.model_name = "BLIP-VQA-Base (Fallback)"
            except Exception as e2:
                logger.error(f"Failed to load any model: {e2}")
                self.vqa_processor = None
                self.vqa_model = None
                self.model_name = "SatQuery Rule/Heuristic Engine"
        finally:
            self._loaded = True

    def get_vqa_pipeline(self):
        self._load_if_needed()
        return self.vqa_processor, self.vqa_model, self.device, getattr(self, "model_name", "Unknown")

ml_manager = MLModels()
