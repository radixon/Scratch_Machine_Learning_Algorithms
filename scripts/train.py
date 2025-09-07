import os
import sys
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.linearregression import LinearRegression

logger = logging.getLogger(__name__)

def main():
    """
    Training process function
    """
    print("="*5,"Linear Model Training Script","="*5)



if __name__ == "__main__":
    try:
        import tqdm
    except ImportError:
        print("tqdm not found. Install requirements")
        sys.exit(1)
    main()