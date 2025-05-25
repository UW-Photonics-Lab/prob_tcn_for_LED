import logging

def setup_logger(log_file="demodula.txt"):
    logger = logging.getLogger("LabVIEW_Debug")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger