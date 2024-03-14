import shutil
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_huggingface_cache(delete: bool):
    cache_dir = os.path.expanduser("~/.cache/huggingface/")
    if os.path.exists(cache_dir) and delete:
        try:
            shutil.rmtree(cache_dir)
            logging.info("Hugging Face cache cleared successfully.")
        except Exception as e:
            logging.info(f"Error clearing Hugging Face cache: {e}")
    else:
        logging.info("Hugging Face cache directory not found.")
