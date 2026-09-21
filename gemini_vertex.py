"""Run the course code on Google Gemini via Vertex AI, with no API keys.

Authentication uses Application Default Credentials (ADC) from
`gcloud auth application-default login` (or the VM's service account).
Project / region / model come from the .env file in the repo root:

    GOOGLE_GENAI_USE_VERTEXAI=true
    GOOGLE_CLOUD_PROJECT=<project id>
    GOOGLE_CLOUD_LOCATION=global
    GEMINI_MODEL=gemini-2.5-flash

Usage in a notebook (in place of `OpenAI()` / `AsyncOpenAI()`):

    from gemini_vertex import openai_client, async_openai_client, MODEL_ID
    client = openai_client()
    client.chat.completions.create(model=MODEL_ID, messages=[...])

    # OpenAI Agents SDK
    from gemini_vertex import agents_model
    agent = Agent(name="x", instructions="...", model=agents_model())
"""

import os

import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Name to pass as `model=` to the OpenAI-compatible endpoint (needs the `google/` publisher prefix)
MODEL_ID = MODEL if "/" in MODEL else f"google/{MODEL}"

_HOST = "aiplatform.googleapis.com" if LOCATION == "global" else f"{LOCATION}-aiplatform.googleapis.com"
BASE_URL = f"https://{_HOST}/v1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi"

_credentials = None


def _token() -> str:
    """Return a fresh OAuth access token from ADC (refreshed when expired)."""
    global _credentials
    if _credentials is None:
        _credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not _credentials.valid:
        _credentials.refresh(google.auth.transport.requests.Request())
    return _credentials.token


async def _atoken() -> str:
    return _token()


def openai_client():
    """OpenAI SDK client that talks to Gemini on Vertex AI."""
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=_token)


def async_openai_client():
    """Async OpenAI SDK client that talks to Gemini on Vertex AI."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=BASE_URL, api_key=_atoken)


def agents_model(model: str = MODEL):
    """Model object for the OpenAI Agents SDK (`Agent(model=...)`).

    Vertex's OpenAI-compatible endpoint expects the `google/` publisher prefix.
    Tracing is disabled because it would otherwise need an OpenAI API key.
    """
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled

    set_tracing_disabled(True)
    return OpenAIChatCompletionsModel(model=model if "/" in model else f"google/{model}", openai_client=async_openai_client())
