import os
import logging
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

# --- CONFIGURATIONS ---
# Base settings provided from your config
CLASSES = ['saya', 'kamu', 'senang', 'bingung', 'marah', 'apa_kabar', 'halo', 'tidur']
BASE_CLASSES = CLASSES[:6]      # First 6 classes for Stage 1
NEW_CLASSES = CLASSES[6:]       # 2 new classes for Stage 2 (halo, tidur)

# Paths
LOCAL_WORK_DIR = os.getcwd()
DATASET_DIR = os.path.join(LOCAL_WORK_DIR, "04_dataset")
# Assume your YOLO data YAML is generated or located here:
DATA_YAML = os.path.join(DATASET_DIR, "bisindo.yaml")

# Pilot Config (adapted for local GPU training)
PILOT_CONFIG = dict(
    epochs=30, imgsz=640, batch=16, seed=42, deterministic=True,
    optimizer='SGD', lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005,
    warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1, cos_lr=False, patience=0,
    nbs=16, device=0, workers=0, amp=False, compile=False, pretrained=True, resume=False,
    single_cls=False, fraction=1., cache=False, rect=False, multi_scale=0.,
    degrees=0., translate=0., scale=0., shear=0., perspective=0., fliplr=0., flipud=0.,
    hsv_h=0., hsv_s=0., hsv_v=0., bgr=0., mosaic=0., mixup=0., cutmix=0.,
    copy_paste=0., close_mosaic=0, erasing=0., auto_augment=None,
    val=True, save=True, plots=True, exist_ok=False, verbose=True
)

def run_stage_1():
    LOG.info("=== STAGE 1: Training Base Model on Base Classes ===")
    # Load base model (e.g., YOLO26/YOLOv8 pre-trained weights)
    model = YOLO("yolov8n.pt") # or your specific model path
    
    # Train stage 1
    results = model.train(
        data=DATA_YAML,
        name="bisindo_stage1_base",
        **PILOT_CONFIG
    )
    
    stage1_weights = os.path.join("runs", "detect", "bisindo_stage1_base", "weights", "best.pt")
    LOG.info(f"Stage 1 complete. Best weights saved at: {stage1_weights}")
    return stage1_weights

def run_stage_2(base_weights_path):
    LOG.info("=== STAGE 2: Incremental Disassemble and Freeze (Method 3) ===")
    
    # Load the trained base weights from Stage 1
    model = YOLO(base_weights_path)
    
    # Freeze backbone/earlier layers to protect learned representations for base classes
    # In Ultralytics YOLO, freezing can target specific layer indexes (e.g., first 10-15 layers)
    LOG.info("Freezing backbone layers for incremental adaptation...")
    for name, param in model.model.named_parameters():
        if "model.22" not in name:  # Keep detection head unfrozen, freeze feature extractor backbone
            param.requires_grad = False

    # Fine-tune with new/combined dataset configuration
    stage2_config = PILOT_CONFIG.copy()
    stage2_config['epochs'] = 15  # Typically fewer epochs for fine-tuning
    stage2_config['lr0'] = 0.001  # Lower learning rate to prevent catastrophic forgetting
    
    results = model.train(
        data=DATA_YAML,
        name="bisindo_stage2_incremental",
        **stage2_config
    )
    
    stage2_weights = os.path.join("runs", "detect", "bisindo_stage2_incremental", "weights", "best.pt")
    LOG.info(f"Stage 2 incremental training complete. Final weights: {stage2_weights}")
    return stage2_weights

if __name__ == "__main__":
    # Ensure dataset yaml exists before kicking off
    if not os.path.exists(DATA_YAML):
        LOG.warning(f"Dataset YAML not found at {DATA_YAML}. Please ensure your YOLO data configuration file is generated.")
    else:
        # Execute Method 3 pipeline
        best_base_weights = run_stage_1()
        final_weights = run_stage_2(best_base_weights)
        LOG.info(f"Pipeline finished successfully! Final model ready for testing at: {final_weights}")