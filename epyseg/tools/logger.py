import logging

class TA_logger(object):
    """
    A logger class for TA (Tracking and Analysis) module.

    Attributes:
        default_format (str): Default log format.
        master_logger_name (str): Master logger name.
        DEBUG (int): Debug logging level.
        INFO (int): Info logging level.
        WARNING (int): Warning logging level.
        ERROR (int): Error logging level.
        CRITICAL (int): Critical logging level.
        DEFAULT (int): Default logging level.

    Examples:
        >>> logger = TA_logger()
        >>> logger.debug("test")
        >>> logger.info("test")
        >>> logger.warning("test")
        >>> logger.error("test")
        >>> logger.critical("test")
        >>> formatter = logging.Formatter(TA_logger.default_format)
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)
        >>> TA_logger.setHandler(handler)
        >>> logger.debug("test")
        >>> logger.info("test")
        >>> logger.warning("test")
        >>> logger.error("test")
        >>> logger.critical("test")
    """

    default_format = '%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - line %(lineno)d - %(message)s\n'
    master_logger_name = 'master'
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    DEFAULT = INFO

    loggers = {}

    def __new__(cls, name=master_logger_name, logging_level=DEFAULT, format=default_format, handler=None):
        if name is not None:
            if name in cls.loggers:
                return cls.loggers.get(name)

        logger = logging.getLogger(name)

        if handler is None:
            formatter = logging.Formatter(format)
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)

        logger.addHandler(handler)
        cls.DEFAULT = logging_level
        logger.setLevel(logging_level)
        cls.loggers[name] = logger

        return logger

    @staticmethod
    def setHandler(handler, name=master_logger_name):
        """
        Set the handler for the logger.

        Args:
            handler (logging.Handler): Handler object.
            name (str): Logger name (default: master_logger_name).
        """
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
