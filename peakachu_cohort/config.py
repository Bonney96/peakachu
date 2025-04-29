import yaml
import logging
import os
from schema import Schema, SchemaError, Optional, Or

log = logging.getLogger(__name__)

# Basic schema - will be expanded as modules are added
CONFIG_SCHEMA = Schema({
    Optional('output_dir'): str,
    Optional('resolutions'): [int],
    Optional('samples'): {
        str: { # Sample Name
            'path': str,
            Optional('group'): str,
        }
    },
    Optional('groups'): {
        str: [str] # Group Name: [Sample Names]
    },
    Optional('ctcf_bed_path'): str,
    Optional('logging'): {
        Optional('level'): Or('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        Optional('file'): str
    }
})

def load_config(config_path):
    """Loads and validates the YAML configuration file."""
    if not config_path or not os.path.exists(config_path):
        log.warning("Configuration file not found or not specified. Using defaults.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log.error(f"Error parsing configuration file {config_path}: {e}")
        raise click.ClickException(f"Invalid YAML format in {config_path}")
    except IOError as e:
        log.error(f"Error reading configuration file {config_path}: {e}")
        raise click.ClickException(f"Could not read configuration file {config_path}")

    if not config_data:
        log.warning(f"Configuration file {config_path} is empty.")
        return {}

    try:
        # Basic validation for now
        # TODO: Add more complex validation logic (e.g., path existence)
        CONFIG_SCHEMA.validate(config_data)
        log.info(f"Configuration loaded successfully from {config_path}")
        return config_data
    except SchemaError as e:
        log.error(f"Configuration validation failed: {e}")
        raise click.ClickException(f"Invalid configuration format in {config_path}: {e}")

# TODO: Implement config merging (CLI args override config file)
# TODO: Implement environment variable substitution 