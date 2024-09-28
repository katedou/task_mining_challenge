import logging
import platform

from .config import LOGGING_LEVEL


# Create customized logger
class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


handler = logging.StreamHandler()
handler.addFilter(HostnameFilter())
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s; Process ID: p%(process)s; Host: %(hostname)s; %(levelname)s] "
        "%(message)s (%(pathname)s:%(lineno)d)"
    )
)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(LOGGING_LEVEL)
