from pathlib import Path


# Project root = parent of the "scripts" folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main folders
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TEMP_DIR = PROJECT_ROOT / "temp"
FUSION_INPUTS_DIR = PROJECT_ROOT / "fusion_inputs"

# Subfolders
BRANCH1_FEATURES_DIR = FEATURES_DIR / "branch1"
BRANCH2_FEATURES_DIR = FEATURES_DIR / "branch2"
BRANCH3_FEATURES_DIR = FEATURES_DIR / "branch3"

CLIP_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "clip"

BRANCH1_MODELS_DIR = MODELS_DIR / "branch1"
BRANCH2_MODELS_DIR = MODELS_DIR / "branch2"
BRANCH3_MODELS_DIR = MODELS_DIR / "branch3"
FUSION_MODELS_DIR = MODELS_DIR / "fusion"

BRANCH1_LOGS_DIR = LOGS_DIR / "branch1"
BRANCH2_LOGS_DIR = LOGS_DIR / "branch2"
BRANCH3_LOGS_DIR = LOGS_DIR / "branch3"
FUSION_LOGS_DIR = LOGS_DIR / "fusion"

# Data folders
AI_DATA_DIR = DATA_DIR / "ai"
REAL_DATA_DIR = DATA_DIR / "real"

# Branch 1 files
BRANCH1_FEATURES_CSV = BRANCH1_FEATURES_DIR / "branch1_features.csv"
BRANCH1_FEATURES_CLEAN_CSV = BRANCH1_FEATURES_DIR / "branch1_features_clean.csv"
BRANCH1_TEST_CSV = BRANCH1_FEATURES_DIR / "branch1_test.csv"
BRANCH1_MODEL = BRANCH1_MODELS_DIR / "branch1_lr.joblib"

# Branch 2 files
BRANCH2_FEATURES_CSV = BRANCH2_FEATURES_DIR / "branch2_features.csv"
BRANCH2_TEST_CSV = BRANCH2_FEATURES_DIR / "branch2_test.csv"
BRANCH2_CNN_FEATURES_CSV = BRANCH2_FEATURES_DIR / "branch2_cnn_features.csv"
BRANCH2_CNN_FEATURES_PARQUET = BRANCH2_FEATURES_DIR / "branch2_cnn_features.parquet"
BRANCH2_CNN_TEST_CSV = BRANCH2_FEATURES_DIR / "branch2_cnn_test.csv"

BRANCH2_HAND_MODEL = BRANCH2_MODELS_DIR / "branch2_hand_lr.joblib"
BRANCH2_CNN_LR_MODEL = BRANCH2_MODELS_DIR / "branch2_cnn_lr.joblib"
BRANCH2_CNN_SGD_MODEL = BRANCH2_MODELS_DIR / "branch2_cnn_sgd.joblib"

# Branch 3 files
BRANCH3_FEATURES_PARQUET = BRANCH3_FEATURES_DIR / "branch3_features.parquet"
BRANCH3_FEATURES_V2_PARQUET = BRANCH3_FEATURES_DIR / "branch3_features_v2.parquet"
BRANCH3_RAW_FEATURES_PARQUET = BRANCH3_FEATURES_DIR / "branch3_raw_features.parquet"

BRANCH3_CLIP_META_CSV = BRANCH3_FEATURES_DIR / "branch3_clip_meta.csv"
BRANCH3_CLIP_TEST_META_CSV = BRANCH3_FEATURES_DIR / "branch3_clip_test_meta.csv"
BRANCH3_CLIP_V2_META_CSV = BRANCH3_FEATURES_DIR / "branch3_clip_v2_meta.csv"
BRANCH3_CLIP_V2_QUADS_NPY = BRANCH3_FEATURES_DIR / "branch3_clip_v2_quads.npy"

BRANCH3_CLIP_EMBEDDINGS_NPY = CLIP_EMBEDDINGS_DIR / "branch3_clip_embeddings.npy"
BRANCH3_CLIP_TEST_EMBEDDINGS_NPY = CLIP_EMBEDDINGS_DIR / "branch3_clip_test_embeddings.npy"

BRANCH3_SEMANTIC_MODEL = BRANCH3_MODELS_DIR / "branch3_semantic_lr.joblib"
BRANCH3_SEMANTIC_V2_MODEL = BRANCH3_MODELS_DIR / "branch3_semantic_v2_lr.joblib"
BRANCH3_RAW_CLIP_MODEL = BRANCH3_MODELS_DIR / "branch3_raw_clip_sgd.joblib"
BRANCH3_CLIP_V2_GLOBAL_NPY = BRANCH3_MODELS_DIR / "branch3_clip_v2_global.npy"

BRANCH3_RAW_TRAIN_LOG = BRANCH3_LOGS_DIR / "branch3_raw_train.log"
BRANCH3_RAW_TRAIN_FULL_LOG = BRANCH3_LOGS_DIR / "branch3_raw_train_full.log"

# Fusion files
FUSION_MODEL = FUSION_MODELS_DIR / "fusion_sgd.joblib"
FUSION_V2_MODEL = FUSION_MODELS_DIR / "fusion_v2_sgd.joblib"
FUSION_TRAIN_V2_LOG = FUSION_LOGS_DIR / "fusion_train_v2.log"

# Temp files
TEMP_DATASET_DICT_JSON = TEMP_DIR / "dataset_dict.json"
TEMP_TRAIN_DIR = TEMP_DIR / "train"


def ensure_dirs():
    dirs = [
        DATA_DIR,
        FEATURES_DIR,
        EMBEDDINGS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        TEMP_DIR,
        FUSION_INPUTS_DIR,
        BRANCH1_FEATURES_DIR,
        BRANCH2_FEATURES_DIR,
        BRANCH3_FEATURES_DIR,
        CLIP_EMBEDDINGS_DIR,
        BRANCH1_MODELS_DIR,
        BRANCH2_MODELS_DIR,
        BRANCH3_MODELS_DIR,
        FUSION_MODELS_DIR,
        BRANCH1_LOGS_DIR,
        BRANCH2_LOGS_DIR,
        BRANCH3_LOGS_DIR,
        FUSION_LOGS_DIR,
        AI_DATA_DIR,
        REAL_DATA_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")