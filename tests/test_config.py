import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.config import load_bot_token


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.env_file = Path(temporary_directory.name) / ".env"
        self.enterContext(patch("bot.config.ENV_FILE", self.env_file))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_loads_token_from_env_file(self) -> None:
        self.env_file.write_text("BOT_TOKEN=test-token\n", encoding="utf-8")
        self.assertEqual(load_bot_token(), "test-token")

    def test_environment_has_priority(self) -> None:
        self.env_file.write_text("BOT_TOKEN=file-token\n", encoding="utf-8")
        os.environ["BOT_TOKEN"] = "environment-token"
        self.assertEqual(load_bot_token(), "environment-token")

    def test_missing_token_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Укажите BOT_TOKEN"):
            load_bot_token()

    def test_blank_token_has_clear_error(self) -> None:
        self.env_file.write_text('BOT_TOKEN="   "\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Укажите BOT_TOKEN"):
            load_bot_token()
