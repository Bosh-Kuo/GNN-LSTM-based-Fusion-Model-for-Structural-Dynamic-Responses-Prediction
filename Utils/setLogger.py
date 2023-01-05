import logging
import os


def setLogger(fileDir):
    if os.path.isfile(fileDir):
        os.remove(fileDir)

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(fileDir)
    fh.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S "
    )

    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
