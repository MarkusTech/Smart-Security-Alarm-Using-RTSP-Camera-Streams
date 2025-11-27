import logging
import os

def setup_logger(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging
