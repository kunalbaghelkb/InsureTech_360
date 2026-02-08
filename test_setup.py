from src.logger import logging
from src.exception import CustomException
import sys

if __name__ == "__main__":
    logging.info("Starting the test script...")
    
    try:
        a = 1 / 0 # Jaan bujh ke error (Zero Division)
    except Exception as e:
        logging.info("Divide by Zero error aa gaya bhai!")
        raise CustomException(e, sys)