from pathlib import Path
import os


class Config:
    ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    print(ROOT_DIR)
    POS_THRESHOLD = None
    NEG_THRESHOLD = None
    NUM_CLASSES = None
    DATA_REPRESENTATION = None
    WORD_2_VEC = {}
    POS = None
    RATING_TYPE = None
    w2v_embeddings = 'w2v.model'
