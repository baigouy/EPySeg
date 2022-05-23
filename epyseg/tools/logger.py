import logging

class TA_logger(object):

    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    default_format = '%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s\n'
    master_logger_name = 'master'
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    DEFAULT = INFO # DEBUG # INFO

    loggers = {}

    def __new__(cls, name=master_logger_name, logging_level=DEFAULT, format=default_format, handler=None):
        if name is not None:
            if name in cls.loggers:
                return cls.loggers.get(name)

        logger = logging.getLogger(name)

        if handler is None:
            # create a formatter
            formatter = logging.Formatter(format)
            # create handler
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)

        logger.addHandler(handler)

        # set level to logging_level
        cls.DEFAULT = logging_level
        logger.setLevel(logging_level)
        cls.loggers[name] = logger

        return logger

    @staticmethod
    def setHandler(handler, name=master_logger_name):
        # easy way to redirect all logs to the same logger
        logger = logging.getLogger(name)
        try:
            for hndlr in logger.handlers:
                logger.removeHandler(hndlr)
        except:
            pass
        logger.addHandler(handler)
        logger.setLevel(TA_logger.DEFAULT)

if __name__ == '__main__':
    logger = TA_logger()
    logger.debug("test")
    logger.info("test")
    logger.warning("test")
    logger.error("test")
    logger.critical("test")

    formatter = logging.Formatter(TA_logger.default_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    TA_logger.setHandler(handler)
    logger.debug("test")
    logger.info("test")
    logger.warning("test")
    logger.error("test")
    logger.critical("test")

