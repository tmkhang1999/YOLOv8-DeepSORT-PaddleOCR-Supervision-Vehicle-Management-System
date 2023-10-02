import logging
from os import path
from pyaml_env import parse_config

log = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

    def load_config(self):
        if not path.isfile(self.config_file_path):
            log.error(f"Unable to load {self.config_file_path}, exiting...")
            raise SystemExit

        return parse_config(self.config_file_path)
