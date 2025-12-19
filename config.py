import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

BASE_TRAIN_PATH = os.path.join(DATA_DIR, "base_train.jsonl")
ADDITIONAL_DATA_PATH = os.path.join(DATA_DIR, "additional_data.jsonl")
LOW_SCORE_PATH = os.path.join(DATA_DIR, "low_score.jsonl")

# Record yang dieksklusi dari training (mis. no-aspect/misc, atau terlalu pendek).
# Format: JSON lines (satu JSON object per baris) untuk memudahkan append.
TRAINING_EXCLUSION_PATH = os.path.join(DATA_DIR, "training_exclusion.json")

LLM_RECOMMENDATION_PATH = os.path.join(DATA_DIR, "llm_recommendation.jsonl")
RELABELLED_LOG_PATH = os.path.join(DATA_DIR, "relabelled.jsonl")
DELETED_REVIEW_PATH = os.path.join(DATA_DIR, "deleted_review.jsonl")

PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
LATEST_PRED_PATH = os.path.join(PREDICTIONS_DIR, "latest_predictions.jsonl")
PRED_GLOB = os.path.join(PREDICTIONS_DIR, "pred_*.jsonl")
# Label aspek untuk UI/output (misc = fallback display ketika tidak ada aspek terdeteksi)
ASPECT_LABELS = ["food", "price", "service", "place_ambience", "misc"]

# Label aspek untuk model (tanpa misc; no-aspect direpresentasikan sebagai list kosong)
MODEL_ASPECT_LABELS = ["food", "price", "service", "place_ambience"]

COMMENT_TYPE_LABELS = ["complaint", "praise", "suggestion"]

# minimal total record untuk training otomatis
MIN_TRAIN_RECORDS_DEFAULT = 100

# default confidence threshold untuk routing ke base_train
CONFIDENCE_THRESHOLD_DEFAULT = 0.6

