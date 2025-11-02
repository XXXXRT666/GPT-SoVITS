from typing import Any


SHARED_DOCS_API = """
# Text To Speech TTFB-optimized API Documentation
"""

SHARED_DOCS_API_BATCH = SHARED_DOCS_API.replace("TTFB", "Batch Inference TTFB")


SHARED_DOCS_TTS = """
# Text To Speech Endpoint

- If the **Prompt Text** is missing, the API will run in prompt free mode for SoVITS v1/v2 models.
- For V3/V4 models, the **Prompt Text** is required.

## Language Options

|   Code   |                    Meaning                    |
| :------: | :-------------------------------------------: |
|  all_zh  |                  All Chinese                  |
| all_yue  |                 All Cantonese                 |
|    en    |                    English                    |
|  all_ja  |                 All Japanese                  |
|  all_ko  |                  All Korean                   |
|    zh    |                    Chinese                    |
|   yue    |            Cantonese-English Mixed            |
|    ja    |            Japanese-English Mixed             |
|    ko    |             Korean-English Mixed              |
|   auto   |        Auto Detecting, Default Chinese        |
| auto_yue | Auto Detecting, Default Cantonese not Chinese |

## Response
- **Success:** Returns the generated speech in stream
- **Error:** Provides error messages or traceback
"""

SHARED_DOCS_SET_PROMPT = """
# Set Prompt Audio Endpoint

The API requires the **Prompt Audio Path** to be specified in the request.

Prompt Text is optional for SoVITS V1/V2/V2 Pro/V2 Pro Plus models, but required for V3/V4 models.
"""

SHARED_TTS_RESPONSE_DICT: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Streaming audio content",
        "content": {
            "audio/wav": {"example": "WAV data"},
            "audio/aac": {"example": "AAC data"},
            "audio/ogg": {"example": "OGG data"},
            "audio/raw": {"example": "RAW PCM data"},
        },
    },
    400: {
        "description": "Plain text error response",
        "content": {
            "text/plain": {"example": "Error message or tracebacks"},
        },
    },
    422: {
        "description": "Validation error (missing or invalid query parameters)",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["query", "text"],
                            "msg": "Field required",
                            "type": "missing",
                        }
                    ]
                }
            }
        },
    },
    500: {
        "description": "Plain text error response during inference",
        "content": {
            "text/plain": {"example": "Runtime Error message or tracebacks"},
        },
    },
}

SHARED_OTHER_RESPONSE_DICT: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Success message",
        "content": {
            "application/json": {"example": "success"},
        },
    },
    400: {
        "description": "Plain text error response",
        "content": {"text/plain": {"example": "Error message or tracebacks"}},
    },
    422: {
        "description": "Validation error (missing or invalid query parameters)",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["query", "text"],
                            "msg": "Field required",
                            "type": "missing",
                        }
                    ]
                }
            }
        },
    },
}
