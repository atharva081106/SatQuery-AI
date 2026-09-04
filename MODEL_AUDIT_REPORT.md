# Model Audit Report

| Field | Value |
|---|---|
| Model architecture | BlipForQuestionAnswering (Salesforce/blip-vqa-base) |
| Task type (classification/detection/regression/generation/etc.) | Visual Question Answering |
| Number of classes / output dimensionality | UNKNOWN — not found (Text generation) |
| Fine-tuning method | full fine-tune |
| Base pretrained weights | Salesforce/blip-vqa-base |
| Training dataset(s) | RSVQADataset (dummy_dataset) |
| Training samples (train/val/test) | Train: 10, Val: 5, Test: 0 |
| Epochs completed | 1 |
| Best recorded validation metric (name + value) | accuracy 0.0 |
| Does it currently run without errors? (yes/no) | yes |
| Input image size/format expected | torch.Size([1, 3, 384, 384]), 3 channels |
| Framework + version | PyTorch (UNKNOWN — not found) |

---

## 1. Discovery — find every relevant artifact

| File | Path | Size (bytes) |
|---|---|---|
| `requirements.txt` | `backend/requirements.txt` | 157 |
| `requirements.txt` | `ml_pipeline/requirements.txt` | 98 |
| `train_vqa.py` | `ml_pipeline/train_vqa.py` | 2263 |
| `config.json` | `ml_pipeline/satquery_finetuned_model/config.json` | 2531 |
| `model.safetensors` | `ml_pipeline/satquery_finetuned_model/model.safetensors` | 1445022200 |
| `config.json` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/config.json` | 2531 |
| `model.safetensors` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/model.safetensors` | 1445022200 |
| `optimizer.pt` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/optimizer.pt` | 2890480156 |
| `rng_state.pth` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/rng_state.pth` | 14455 |
| `scheduler.pt` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/scheduler.pt` | 1465 |
| `trainer_state.json` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/trainer_state.json` | 1074 |
| `training_args.bin` | `ml_pipeline/satquery_finetuned_model/checkpoint-5/training_args.bin` | 5265 |
| `evaluation_results.json` | `ml_pipeline/evaluation_results.json` | 401 |

The default model loaded is `ml_pipeline/satquery_finetuned_model/model.safetensors`.

## 2. Environment
- Framework and exact version: PyTorch, version `UNKNOWN — not found`.
- CUDA/GPU availability at training time: `UNKNOWN — not found` (Logs were empty).
- Python version: 3.10.
- Pinned versions of key ML libraries: `UNKNOWN — not found` (Unpinned versions).

## 3. Architecture
- Model class/type name: `BlipForQuestionAnswering` (wrapped around `Salesforce/blip-vqa-base`).
- Total parameter count: 361,230,140 total parameters. 361,230,140 trainable parameters.
- This is a full fine-tune.
- Full layer/module summary:
```
BlipForQuestionAnswering(
  (vision_model): BlipVisionModel(
    (embeddings): BlipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (encoder): BlipEncoder(
      (layers): ModuleList(
        (0-11): 12 x BlipEncoderLayer(...)
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (text_encoder): BlipTextModel(
    (embeddings): BlipTextEmbeddings(...)
    (encoder): BlipTextEncoder(
      (layer): ModuleList(
        (0-11): 12 x BlipTextLayer(...)
      )
    )
  )
  (text_decoder): BlipTextLMHeadModel(
    (bert): BlipTextModel(...)
    (cls): BlipTextOnlyMLMHead(
      (predictions): BlipTextLMPredictionHead(
        (transform): BlipTextPredictionHeadTransform(...)
        (decoder): Linear(in_features=768, out_features=30524, bias=True)
      )
    )
  )
)
```
- Input shape/format the model expects: Image size 384, RGB (3 channels), processed into `torch.Size([1, 3, 384, 384])`.
- Output shape/format: Generates text outputs of shape `torch.Size([1, 3])` mapping to token vocabularies.

## 4. Training provenance
- Dataset(s) used: `RSVQADataset` sourced locally from `dummy_dataset`.
- Number of training samples: 10 train, 5 val, 0 test.
- Class balance: `UNKNOWN — not found`.
- Image resolution used during training: `UNKNOWN — not found`.
- Batch size: 2. Epochs configured: 10. Epochs completed: 1 (Stopped early at step 5).
- Optimizer: `UNKNOWN — not found`.
- Learning rate: 5e-05.
- Loss function used: `UNKNOWN — not found`.
- Data augmentation applied: None.
- Fine-tuning method: Full fine-tune (361M/361M params).
- Pretrained initialization: `Salesforce/blip-vqa-base`.
- Random seed: `UNKNOWN — not found`.
- Total wall-clock training time: `UNKNOWN — not found`.

## 5. Recorded metrics
- Final training loss: `UNKNOWN — not found`.
- Final validation loss: 10.261253356933594.
- Accuracy: 0.0 (Recorded in `evaluation_results.json`).
- Confusion matrix: All 5 validation samples had a Ground Truth of "yes", and Model Prediction of "no".
- Explicit train/val gap: `UNKNOWN — not found`.

## 6. Live inference capability check
- The model runs flawlessly with no errors end-to-end.
- 1 test image evaluated: `ml_pipeline/dummy_dataset/val/images/image_000.tif`
- Raw output tensor shape: `torch.Size([1, 3])`
- Top predicted class/value: "no"
- Confidence scores: `UNKNOWN — not found`
- Total inference latency: 4453.77 ms

## 7. Domain and generalization signal
- Visual domain: Remote Sensing, inferred from code contexts, though model inference ran on a dummy `.tif` dataset lacking real metadata or features.
- Image types/conditions: `UNKNOWN — not found` (dummy datasets used).
- Held-out test performed: No out-of-distribution (OOD) tests performed; evaluating the limited validation split yielded 0.0 accuracy, indicating complete inability to generalize.
