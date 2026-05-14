# DS2API

Unofficial Python client for the DeepSeek web chat backend at
[`chat.deepseek.com`](https://chat.deepseek.com/).

DS2API wraps the browser-session flows used by the DeepSeek web app: one-shot
chat, managed multi-turn sessions, streaming, DeepThink, web search, file
attachment prompts, and Proof-of-Work solving.

> [!IMPORTANT]
> DS2API is not an official `api.deepseek.com` SDK. It uses undocumented web
> endpoints, so DeepSeek can change authentication, payloads, model flags, or
> Proof-of-Work behavior without notice.

## Contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Authentication](#authentication)
- [Quick Start](#quick-start)
- [One-Shot Chat](#one-shot-chat)
- [Session Chat](#session-chat)
- [DeepThink And Web Search](#deepthink-and-web-search)
- [File Uploads](#file-uploads)
- [Async Client](#async-client)
- [Error Handling](#error-handling)
- [Client Options](#client-options)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [License](#license)

## Highlights

- `instant` and `expert` chat modes.
- One-shot prompts with automatic temporary session cleanup.
- Managed multi-turn sessions with parent-message chaining.
- Streaming events for answer text, thinking text, citations, token usage,
  message IDs, and completion markers.
- Optional DeepThink and web search flags.
- File upload and attachment support for web-chat prompts.
- Bounded timeouts, strict input validation, typed exceptions, and safe error
  previews.
- `AsyncDeepSeekClient` wrapper for asyncio applications.

## Requirements

- Python 3.11 or newer.
- A valid DeepSeek web `userToken` from a logged-in browser session.
- Runtime dependencies:
  - `requests`
  - `wasmtime`

The package includes `src/sha3_wasm_bg.wasm`, which is required for DeepSeek
web Proof-of-Work challenges. Keep that file when copying, packaging, or
publishing this project.

## Installation

### Install From GitHub

```bash
python -m pip install "ds2api @ git+https://github.com/HeIT2703/DS2API-PRO.git"
```

The distribution name is `ds2api`, while the Python import package is `DS2API`:

```python
from DS2API import DeepSeekClient
```

### Install Locally For Development

From the folder that contains this README:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The `.venv` folder is created locally by `python -m venv .venv`. It is
intentionally ignored by Git and should not be committed.

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Verify the installation:

```bash
python -c "from DS2API import DeepSeekClient; print(DeepSeekClient.__name__)"
```

### Manual Copy Usage

If you copy the `DS2API` package folder into another project instead of
installing it with pip, install the runtime dependencies yourself:

```bash
python -m pip install "requests>=2.31,<3" "wasmtime>=25"
```

Your project should then contain a folder named `DS2API` next to the script
that imports it, or that folder should be available on `PYTHONPATH`.

## Authentication

You need a `userToken` from a logged-in DeepSeek web session.

1. Open [`chat.deepseek.com`](https://chat.deepseek.com/) and log in.
2. Open browser developer tools.
3. Go to Application or Storage, then Local Storage.
4. Select `https://chat.deepseek.com`.
5. Copy the `userToken` value.

Treat this token like a password. Do not commit it, print it in logs, paste it
into public issues, or hard-code it in scripts.

PowerShell:

```powershell
$env:DEEPSEEK_USER_TOKEN = "your-userToken-here"
python your_script.py
Remove-Item Env:\DEEPSEEK_USER_TOKEN
```

macOS or Linux:

```bash
export DEEPSEEK_USER_TOKEN="your-userToken-here"
python your_script.py
unset DEEPSEEK_USER_TOKEN
```

## Quick Start

```python
import os

from DS2API import DeepSeekClient

token = os.environ["DEEPSEEK_USER_TOKEN"]

with DeepSeekClient(token=token) as client:
    response = client.ask("Reply with one short sentence.", model="instant")
    print(response.text)
```

## One-Shot Chat

`ask()` creates a temporary web chat session, sends one message, collects the
full response, and deletes the session.

```python
response = client.ask(
    "What is 12 * 7?",
    model="expert",
    thinking=True,
)

print(response.text)
print(response.thinking)
print(response.thinking_elapsed)
```

Use `ask_stream()` when you want events as they arrive:

```python
for event in client.ask_stream("Count from 1 to 3.", model="instant"):
    if event.event_type == "RESPONSE_TEXT":
        print(event.content, end="", flush=True)
```

Common event types:

- `MESSAGE_ID`: generated DeepSeek message ID.
- `RESPONSE_TEXT`: answer text chunk.
- `THINK_TEXT`: DeepThink text chunk.
- `THINKING_DONE`: thinking elapsed time.
- `SEARCH_RESULTS`: list of citation objects.
- `TOKEN_USAGE`: token usage count.
- `FINISHED`: stream completion marker.

## Session Chat

Use `new_chat()` for multi-turn context. DS2API tracks the latest message ID
for sessions it creates or sessions you explicitly adopt.

```python
session_id = client.new_chat(model="instant")

try:
    client.send(session_id, "Remember this code: BLUE-7319.")
    response = client.send(session_id, "What code did I ask you to remember?")
    print(response.text)
finally:
    client.delete_chat(session_id)
```

If a session was created outside this client instance, register it first:

```python
client.adopt_chat("existing-session-id", model="instant", last_message_id=123)
response = client.send("existing-session-id", "Continue from here.")
```

Unknown session IDs are rejected by high-level helpers so the client does not
silently guess model or context state. For lower-level calls, use `client.chat`.

## DeepThink And Web Search

DeepThink and web search are exposed through boolean flags:

```python
response = client.ask(
    "Solve 12 * 7.",
    model="expert",
    thinking=True,
)

print(response.thinking)
print(response.text)
```

```python
response = client.ask(
    "Use web search and answer briefly: what is DeepSeek's official domain?",
    search=True,
)

print(response.text)
for citation in response.citations:
    print(citation.title, citation.url)
```

## File Uploads

Upload a file, extract its ID, then attach that ID to a prompt.

```python
upload = client.file.upload_file("notes.txt", wait_ready=True)
file_info = client.file.extract_file_info(upload)

response = client.ask(
    "Summarize the attached file.",
    ref_file_ids=[file_info.id],
)

print(response.text)
```

Some model and file combinations may be rejected by DeepSeek. DS2API surfaces
those failures as explicit `APIRequestError` exceptions. Uploaded files may
remain visible in the DeepSeek account unless the web backend removes them.

## Async Client

`AsyncDeepSeekClient` is an asyncio-friendly wrapper around the synchronous
client. It uses worker threads through `asyncio.to_thread`; it is not a native
async HTTP transport.

```python
import asyncio
import os

from DS2API import AsyncDeepSeekClient


async def main():
    token = os.environ["DEEPSEEK_USER_TOKEN"]

    async with AsyncDeepSeekClient(token=token) as client:
        response = await client.ask("Hello from asyncio.", model="instant")
        print(response.text)

        async for event in client.ask_stream("Count from 1 to 3."):
            if event.event_type == "RESPONSE_TEXT":
                print(event.content, end="", flush=True)


asyncio.run(main())
```

## Error Handling

Import project exceptions from `DS2API`:

```python
from DS2API import APIRequestError, ValidationError

try:
    response = client.ask("")
except ValidationError as exc:
    print(f"Bad input: {exc}")
except APIRequestError as exc:
    print(f"DeepSeek request failed: {exc}")
```

`str(exc)` redacts common secrets and PII from response previews. The raw
response body is still available on `APIRequestError.response_body` for callers
that deliberately need it.

## Client Options

Most users only need `token`. Advanced callers can pass network, retry, upload,
and Proof-of-Work options:

```python
client = DeepSeekClient(
    token=token,
    timeout=(10, 60),
    stream_timeout=(10, 300),
    max_retries=2,
    user_agent=None,
    trust_env=False,
)
```

Useful options:

- `authorization`: pass a full authorization header instead of `token`.
- `cookies`: add browser cookies when the web backend requires them.
- `proxies`, `verify`, `cert`, `trust_env`: forward network settings to
  `requests`.
- `max_upload_size_bytes`: block oversized file uploads before sending them.
- `pow_max_difficulty`: refuse unexpectedly expensive Proof-of-Work challenges.

## Troubleshooting

### `AuthenticationError: A valid userToken must be provided.`

The token is missing or blank. Re-copy `userToken` from Local Storage and pass
it through an environment variable.

### `PoWSolverError: wasmtime library is missing.`

Install the missing dependency:

```bash
python -m pip install "wasmtime>=25"
```

### `ModuleNotFoundError: No module named 'DS2API'`

Install the package with pip, run Python from the project that contains the
`DS2API` folder, or make sure that folder is available on `PYTHONPATH`.

### Missing `sha3_wasm_bg.wasm`

Keep `src/sha3_wasm_bg.wasm` in the published repository and installed package.
It is required for web Proof-of-Work challenges.

### DeepSeek Returns A Web API Error

This project uses undocumented web endpoints. If DeepSeek changes the browser
API, auth requirements, model flags, or Proof-of-Work behavior, requests may
start failing until the client is updated.

## Security Notes

- Do not commit real `userToken` values, cookies, authorization headers, or
  `.env` files.
- Prefer environment variables over hard-coded credentials.
- Review local changes before publishing:

```bash
git status --short
git diff --check
```

## License

No license file is included yet. Add a license before publishing this project
for broader public use.
