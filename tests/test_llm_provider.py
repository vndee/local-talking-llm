"""Unit tests for the LLM provider factory in app.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import create_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCreateLlm(unittest.TestCase):
    """Tests for the create_llm() factory function."""

    def _import_create_llm(self):
        """Import create_llm with heavy dependencies mocked out."""
        # Mock heavy imports that aren't needed for testing the factory logic
        mock_nltk = MagicMock()
        mock_nltk.__spec__ = MagicMock()  # transformers checks __spec__ to detect nltk

        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
            "numpy": MagicMock(),
            "torch": MagicMock(),
            "torchaudio": MagicMock(),
            "nltk": mock_nltk,
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(),
            "tts": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            # Prevent argparse from parsing test runner args
            with patch("sys.argv", ["app.py"]):
                # Re-import to get the create_llm function
                if "app" in sys.modules:
                    del sys.modules["app"]
                from app import create_llm
        return create_llm

    @patch("langchain_ollama.OllamaLLM")
    def test_ollama_provider_default_model(self, mock_ollama):
        """Test that Ollama provider uses default model 'gemma3'."""
        create_llm = self._import_create_llm()
        mock_ollama.reset_mock()  # Clear calls from module-level initialization
        create_llm("ollama")
        mock_ollama.assert_called_once_with(model="gemma3", base_url="http://localhost:11434")

    @patch("langchain_ollama.OllamaLLM")
    def test_ollama_provider_custom_model(self, mock_ollama):
        """Test that Ollama provider accepts a custom model name."""
        create_llm = self._import_create_llm()
        mock_ollama.reset_mock()  # Clear calls from module-level initialization
        create_llm("ollama", model="llama3")
        mock_ollama.assert_called_once_with(model="llama3", base_url="http://localhost:11434")

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_provider_default_model(self, mock_chat):
        """Test that MiniMax provider uses default model 'MiniMax-M3'."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key")
        mock_chat.assert_called_once_with(
            model="MiniMax-M3",
            base_url="https://api.minimax.io/v1",
            api_key="test-key",
            temperature=0.7,
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_provider_custom_model(self, mock_chat):
        """Test that MiniMax provider accepts a custom model name."""
        create_llm = self._import_create_llm()
        create_llm("minimax", model="MiniMax-M2.7-highspeed", api_key="test-key")
        mock_chat.assert_called_once_with(
            model="MiniMax-M2.7-highspeed",
            base_url="https://api.minimax.io/v1",
            api_key="test-key",
            temperature=0.7,
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_api_key_from_env(self, mock_chat):
        """Test that MiniMax provider reads API key from MINIMAX_API_KEY env var."""
        create_llm = self._import_create_llm()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            create_llm("minimax")
        mock_chat.assert_called_once_with(
            model="MiniMax-M3",
            base_url="https://api.minimax.io/v1",
            api_key="env-key",
            temperature=0.7,
        )

    def test_minimax_no_api_key_raises(self):
        """Test that MiniMax provider raises ValueError when no API key is provided."""
        create_llm = self._import_create_llm()
        with patch.dict(os.environ, {}, clear=True):
            # Ensure MINIMAX_API_KEY is not set
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                create_llm("minimax")
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_clamping_zero(self, mock_chat):
        """Test that temperature 0.0 is clamped to 0.01 for MiniMax."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=0.0)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.01)

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_clamping_negative(self, mock_chat):
        """Test that negative temperature is clamped to 0.01 for MiniMax."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=-0.5)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.01)

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_clamping_high(self, mock_chat):
        """Test that temperature > 1.0 is clamped to 1.0 for MiniMax."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=2.0)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 1.0)

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_normal(self, mock_chat):
        """Test that a valid temperature is passed through unchanged."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=0.5)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.5)

    def test_unknown_provider_raises(self):
        """Test that an unknown provider raises ValueError."""
        create_llm = self._import_create_llm()
        with self.assertRaises(ValueError) as ctx:
            create_llm("unknown_provider")
        self.assertIn("Unknown provider", str(ctx.exception))

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_api_key_arg_overrides_env(self, mock_chat):
        """Test that explicit api_key argument takes precedence over env var."""
        create_llm = self._import_create_llm()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            create_llm("minimax", api_key="arg-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "arg-key")

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_base_url(self, mock_chat):
        """Test that MiniMax provider uses the correct base URL."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")


class TestAnalyzeEmotion(unittest.TestCase):
    """Tests for the analyze_emotion() helper function."""

    def _import_analyze_emotion(self):
        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
            "numpy": MagicMock(),
            "torch": MagicMock(),
            "torchaudio": MagicMock(),
            "nltk": MagicMock(),
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(),
            "tts": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            with patch("sys.argv", ["app.py"]):
                if "app" in sys.modules:
                    del sys.modules["app"]
                from app import analyze_emotion
        return analyze_emotion

    def test_neutral_text(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("The weather is nice today.")
        self.assertAlmostEqual(score, 0.5)

    def test_emotional_text(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("This is amazing! I love it!")
        self.assertGreater(score, 0.5)

    def test_score_capped_at_max(self):
        analyze_emotion = self._import_analyze_emotion()
        # Text with many emotional keywords
        score = analyze_emotion("amazing terrible love hate excited sad happy angry wonderful awful !")
        self.assertLessEqual(score, 0.9)

    def test_score_has_minimum(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("ok")
        self.assertGreaterEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
