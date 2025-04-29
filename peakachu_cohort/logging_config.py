import logging
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging(verbosity: int = 0, log_file: str | None = None, config_level: str | None = None):
    """Configures the root logger.

    Args:
        verbosity: Verbosity level from CLI flags (0=WARN, 1=INFO, 2=DEBUG).
        log_file: Path to a file for logging output (from config).
        config_level: Logging level specified in the config file (e.g., 'DEBUG').
    """
    # Determine effective log level, prioritizing CLI verbosity
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Override with config level if it's more verbose
    if config_level:
        try:
            config_level_int = getattr(logging, config_level.upper(), level)
            if config_level_int < level: # Lower level means more verbose
                level = config_level_int
        except AttributeError:
            logging.warning(f"Invalid logging level in config: {config_level}. Using default.")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Setup formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except IOError as e:
            logging.error(f"Could not open log file {log_file}: {e}", exc_info=False)

    logging.debug(f"Logging setup complete. Level: {logging.getLevelName(level)}")

# TODO: Consider adding colorized output for console (e.g., using 'rich' library)
# TODO: Consider log rotation for file handler 