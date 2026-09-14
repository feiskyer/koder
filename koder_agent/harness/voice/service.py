"""Provider-backed voice dictation services."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import wave
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI

from koder_agent.auth.client_integration import get_provider_auth_info
from koder_agent.config import get_config
from koder_agent.harness.voice.keyterms import get_all_keyterms
from koder_agent.utils.client import get_provider_api_env_var

SUPPORTED_VOICE_PROVIDERS = {"openai", "chatgpt", "google", "gemini", "azure"}
DOUBLE_SPACE_WINDOW_SECONDS = 0.35
DEFAULT_OPENAI_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_GEMINI_TRANSCRIBE_MODEL = "gemini-2.5-flash"
TRANSCRIPTION_PROMPT = (
    "Generate a transcript of the spoken audio. Return only the spoken words as plain text."
)
logger = logging.getLogger(__name__)


class VoiceDictationError(RuntimeError):
    """Voice dictation runtime error."""


def resolve_voice_provider(runtime_config, model_provider: str) -> Optional[str]:
    configured = getattr(runtime_config.voice, "provider", None)
    if configured:
        return configured.strip().lower()
    provider = (model_provider or "").strip().lower()
    return provider or None


def resolve_voice_model(runtime_config, provider: str) -> str:
    configured = getattr(runtime_config.voice, "model", None)
    if configured:
        return configured.strip()
    provider_lower = provider.lower()
    if provider_lower in {"openai", "chatgpt", "azure"}:
        return DEFAULT_OPENAI_TRANSCRIBE_MODEL
    return DEFAULT_GEMINI_TRANSCRIBE_MODEL


def should_start_voice_shortcut(
    *,
    buffer_text: str,
    cursor_position: int,
    last_space_at: Optional[float],
    now: float,
    enabled: bool,
    busy: bool,
) -> bool:
    if not enabled or busy or last_space_at is None:
        return False
    if now - last_space_at > DOUBLE_SPACE_WINDOW_SECONDS:
        return False
    return buffer_text == " " and cursor_position == 1


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return str(value)


def _encode_wav(audio_bytes: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


def _extract_gemini_text(payload: dict) -> str:
    texts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return "".join(texts)


def resolve_voice_credentials(provider: str) -> tuple[str, dict[str, str], Optional[str]]:
    api_key, extra_headers, _ = get_provider_auth_info(provider)
    headers = dict(extra_headers or {})
    config = get_config()

    if config.voice.api_key:
        return config.voice.api_key, headers, _resolve_provider_base_url(provider, config)

    if api_key:
        return api_key, headers, _resolve_provider_base_url(provider, config)

    provider_lower = provider.lower()
    family_env_vars = [get_provider_api_env_var(provider_lower)]
    if provider_lower in {"chatgpt", "openai"}:
        family_env_vars.append("OPENAI_API_KEY")
    if provider_lower == "azure":
        family_env_vars.extend(["AZURE_API_KEY", "OPENAI_API_KEY"])
    if provider_lower in {"google", "gemini"}:
        family_env_vars.extend(["GOOGLE_API_KEY", "GEMINI_API_KEY"])

    for env_var in family_env_vars:
        value = os.environ.get(env_var)
        if value:
            return value, headers, _resolve_provider_base_url(provider, config)

    config_provider = (config.model.provider or "").strip().lower()
    if config.model.api_key and config_provider in {
        provider_lower,
        "openai",
        "chatgpt",
        "google",
        "gemini",
        "azure",
    }:
        same_openai_family = provider_lower in {"openai", "chatgpt"} and config_provider in {
            "openai",
            "chatgpt",
        }
        same_azure_family = provider_lower == "azure" and config_provider == "azure"
        same_gemini_family = provider_lower in {"google", "gemini"} and config_provider in {
            "google",
            "gemini",
        }
        if (
            config_provider == provider_lower
            or same_openai_family
            or same_azure_family
            or same_gemini_family
        ):
            return config.model.api_key, headers, _resolve_provider_base_url(provider, config)

    raise VoiceDictationError(f"Voice mode requires credentials for provider '{provider_lower}'.")


def _resolve_provider_base_url(provider: str, config) -> Optional[str]:
    if config.voice.base_url:
        return config.voice.base_url
    provider_lower = provider.lower()
    config_provider = (config.model.provider or "").strip().lower()
    if config.model.base_url and (
        config_provider == provider_lower
        or (provider_lower in {"openai", "chatgpt"} and config_provider in {"openai", "chatgpt"})
    ):
        return config.model.base_url
    if provider_lower == "azure":
        return os.environ.get("AZURE_API_BASE") or os.environ.get("OPENAI_API_BASE")
    if provider_lower in {"openai", "chatgpt"}:
        return os.environ.get("OPENAI_BASE_URL")
    return None


def _resolve_azure_endpoint_and_api_version(config) -> tuple[str, str]:
    base_url = _resolve_provider_base_url("azure", config)
    if not base_url:
        raise VoiceDictationError(
            "Azure voice transcription requires `voice.base_url` or `AZURE_API_BASE`."
        )
    parsed = urlsplit(base_url)
    path_parts = parsed.path.split("/")
    if "openai" in path_parts:
        path_parts = path_parts[: path_parts.index("openai")]
    azure_endpoint = urlunsplit(
        (parsed.scheme, parsed.netloc, "/".join(path_parts).rstrip("/"), "", "")
    )
    model_api_version = (
        getattr(config.model, "azure_api_version", None)
        if (config.model.provider or "").strip().lower() == "azure"
        else None
    )
    api_version = (
        config.voice.api_version
        or model_api_version
        or os.environ.get("AZURE_API_VERSION")
        or "2025-04-01-preview"
    )
    return azure_endpoint, api_version


class SoundDeviceRecorder:
    """Cross-platform recorder backed by sounddevice/PortAudio."""

    def __init__(self, *, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._chunks: list[bytes] = []

    def start(self) -> None:
        if self._stream is not None:
            raise VoiceDictationError("Voice recording is already active.")
        self._chunks = []
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise VoiceDictationError("Voice capture requires the `sounddevice` package.") from exc

        def callback(indata, _frames, _time, status):
            if status:
                return
            self._chunks.append(bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            try:
                self.cancel()
            except Exception:
                logger.warning("Failed to clean up voice capture after startup failure")
            raise VoiceDictationError(f"Voice recording failed to start: {exc}") from exc

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def stop(self) -> bytes:
        if self._stream is None:
            raise VoiceDictationError("Voice recording is not active.")
        try:
            self._close_stream()
            if not self._chunks:
                raise VoiceDictationError("No audio was captured.")
            audio = b"".join(self._chunks)
        finally:
            self._chunks = []
        return _encode_wav(audio, sample_rate=self.sample_rate, channels=self.channels)

    def cancel(self) -> None:
        try:
            self._close_stream()
        finally:
            self._chunks = []


class ProviderVoiceTranscriber:
    """Provider-specific speech-to-text dispatcher."""

    async def transcribe(
        self,
        *,
        audio_bytes: bytes,
        provider: str,
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> str:
        provider_lower = provider.lower()
        if provider_lower in {"openai", "chatgpt", "azure"}:
            return await self._transcribe_openai_family(audio_bytes, provider_lower)
        if provider_lower in {"google", "gemini"}:
            return await self._transcribe_gemini_family(
                audio_bytes,
                provider_lower,
                on_partial=on_partial,
            )
        raise VoiceDictationError(f"Voice mode is not available for provider: {provider_lower}.")

    async def _transcribe_openai_family(self, audio_bytes: bytes, provider: str) -> str:
        api_key, headers, base_url = resolve_voice_credentials(provider)
        config = get_config()
        if provider == "azure":
            azure_endpoint, api_version = _resolve_azure_endpoint_and_api_version(config)
            client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                default_headers=headers or None,
            )
        else:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=headers or None,
            )
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "dictation.wav"
        model = resolve_voice_model(config, provider)

        # Get keyterms for better technical vocabulary recognition
        try:
            cwd = os.getcwd()
            keyterms = get_all_keyterms(cwd)
            prompt_hint = ", ".join(keyterms[:50])  # Limit to first 50 terms
        except Exception:
            prompt_hint = None

        try:
            kwargs = {
                "model": model,
                "file": audio_file,
                "response_format": "text",
            }
            if prompt_hint:
                kwargs["prompt"] = prompt_hint
            result = await client.audio.transcriptions.create(**kwargs)
            transcript = _coerce_text(result).strip()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                maybe = close()
                if hasattr(maybe, "__await__"):
                    await maybe
        if not transcript:
            raise VoiceDictationError("Voice transcription returned no text.")
        return transcript

    async def _transcribe_gemini_family(
        self,
        audio_bytes: bytes,
        provider: str,
        *,
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> str:
        api_key, headers, base_url = resolve_voice_credentials(provider)
        model = resolve_voice_model(get_config(), provider)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": TRANSCRIPTION_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0},
        }
        api_base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url_base = f"{api_base}/models/{model}"
        request_headers = {"Content-Type": "application/json", **headers}
        params = {}
        if "Authorization" not in request_headers:
            params["key"] = api_key

        if on_partial is not None:
            text = await self._stream_gemini_transcript(
                url=f"{url_base}:streamGenerateContent",
                payload=payload,
                headers=request_headers,
                params=params,
                on_partial=on_partial,
            )
            if text:
                return text

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{url_base}:generateContent",
                headers=request_headers,
                params=params,
                json=payload,
            )
            response.raise_for_status()
            text = _extract_gemini_text(response.json()).strip()
            if not text:
                raise VoiceDictationError("Voice transcription returned no text.")
            return text

    async def _stream_gemini_transcript(
        self,
        *,
        url: str,
        payload: dict,
        headers: dict[str, str],
        params: dict[str, str],
        on_partial: Callable[[str], None],
    ) -> str:
        accumulated = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                params={**params, "alt": "sse"},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped.startswith("data:"):
                        continue
                    data = stripped[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    text = _extract_gemini_text(chunk)
                    if not text:
                        continue
                    accumulated += text
                    on_partial(accumulated)
        return accumulated.strip()


@dataclass
class VoiceDictationController:
    """Coordinates recording and provider-backed transcription."""

    config_getter: Callable[[], object]
    model_provider_getter: Callable[[], str]
    recorder_factory: Callable[[], object] = SoundDeviceRecorder
    transcriber: ProviderVoiceTranscriber = ProviderVoiceTranscriber()

    def __post_init__(self) -> None:
        self._state = "idle"
        self._recorder = None
        self._provider: Optional[str] = None
        self._generation = 0
        self._transcription_task: Optional[asyncio.Task] = None

    @property
    def is_busy(self) -> bool:
        return self._state in {"recording", "transcribing"}

    @property
    def is_recording(self) -> bool:
        return self._state == "recording"

    def is_enabled(self) -> bool:
        config = self.config_getter()
        return bool(getattr(config.voice, "enabled", False))

    async def start_recording(self, *, on_status: Callable[[Optional[str]], None]) -> None:
        if self.is_busy:
            raise VoiceDictationError("Voice dictation is already active.")
        config = self.config_getter()
        provider = resolve_voice_provider(config, self.model_provider_getter())
        if provider not in SUPPORTED_VOICE_PROVIDERS:
            raise VoiceDictationError(
                f"Voice mode is not available for provider: {provider or 'unknown'}."
            )
        recorder = self.recorder_factory()
        self._generation += 1
        self._recorder = recorder
        self._provider = provider
        try:
            recorder.start()
            self._state = "recording"
            on_status("recording")
        except BaseException:
            self.cancel()
            raise

    async def stop_recording(
        self,
        *,
        on_status: Callable[[Optional[str]], None],
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> str:
        if self._state != "recording" or self._recorder is None or self._provider is None:
            raise VoiceDictationError("Voice dictation is not recording.")
        generation = self._generation
        recorder = self._recorder
        provider = self._provider
        self._transcription_task = asyncio.current_task()

        def report_partial(text: str) -> None:
            if generation == self._generation and on_partial is not None:
                on_partial(text)

        try:
            try:
                audio_bytes = recorder.stop()
            finally:
                # Capture is finished before any provider await. Release it now
                # so a later cancellation/finalizer cannot touch a new capture.
                self._cancel_recorder(recorder)
                self._recorder = None
            self._state = "transcribing"
            on_status("transcribing")
            if generation != self._generation:
                raise asyncio.CancelledError
            transcript = await self.transcriber.transcribe(
                audio_bytes=audio_bytes,
                provider=provider,
                on_partial=report_partial if on_partial is not None else None,
            )
            # Cancellation can be suppressed by a provider. Never return that
            # stale transcript to a caller that will submit it as user input.
            if generation != self._generation:
                raise asyncio.CancelledError
            return transcript.strip()
        finally:
            if generation == self._generation:
                self._recorder = None
                self._provider = None
                self._transcription_task = None
                self._state = "idle"
                on_status(None)

    @staticmethod
    def _cancel_recorder(recorder) -> None:
        if recorder is not None and hasattr(recorder, "cancel"):
            try:
                recorder.cancel()
            except Exception:
                logger.warning("Failed to clean up voice capture")

    def cancel(self) -> None:
        # Revoke ownership before signalling cancellation: a new recording can
        # start before the old transcription's finally block runs.
        self._generation += 1
        task, self._transcription_task = self._transcription_task, None
        recorder = self._recorder
        self._recorder = None
        self._provider = None
        self._state = "idle"
        if task is not None and not task.done():
            task.cancel()
        self._cancel_recorder(recorder)
