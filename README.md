              __
           .-"  "-.
          /  .--.  \
         /  /    \  \
         |  | () |  |   _  _
         |  |____|  |  ( `   )  pudú
        /\__\____/__/\  `._.'   (the tiny deer) <- Chatgpt Idea of a Pudú
       /_/  /_/\_\  \_\
         /_/  \\_\
        (__)  (___)
```

## kortar
An AI-powered FFmpeg assistant that plans and generates video editing commands from natural language. It can analyze videos, plan editing tasks, and build runnable FFmpeg commands—optionally using AI services for content analysis and transcription.

### Key features
- Plan-first agent that breaks requests into goal-oriented tasks
- Generates validated FFmpeg commands for overlays, text, transitions, sync fixes, and more
- Technical analysis via ffprobe
- Optional content analysis with Gemini (gated by `GEMINI_API_KEY`)
- Optional transcription + SRT generation with Deepgram (gated by `DEEPGRAM_API_KEY`)
- Interactive CLI with progress and confirmations

### Requirements
- Python 3.11+
- FFmpeg and ffprobe installed and available on PATH
- API keys (set via environment variables or `.env`):
  - Required for core model usage: `ANTHROPIC_API_KEY`
  - Optional for content analysis: `GEMINI_API_KEY`
  - Optional for transcription: `DEEPGRAM_API_KEY`

### Install
Using pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

Or using `uv`:

```bash
uv venv && source .venv/bin/activate
uv sync
```

### Configure environment
Create a `.env` file in the project root (or export vars in your shell):

```bash
# Required for agent models
export ANTHROPIC_API_KEY="sk-ant-..."

# Enables AI content analysis tools
export GEMINI_API_KEY="sk-gem-..."

# Enables Deepgram transcription tools
export DEEPGRAM_API_KEY="dg_..."
```

When importing `tools`, the app logs which features are enabled based on these keys.

### Run (interactive)
Start the guided, multi-line interactive mode:

```bash
python start.py interactive
```

Tips shown in the UI: type your first line, continue on next lines, press Enter on an empty line to submit. Use `help`, `clear`, or `quit` anytime.

### Analyze a video
- Technical (ffprobe):
```bash
python start.py analyze path/to/video.mp4 --technical
```

- Content (Gemini, requires `GEMINI_API_KEY`):
```bash
python start.py analyze path/to/video.mp4 --content --query "Find empty moments and key highlights"
```

### Ask for edits (command generation)
Generate an FFmpeg command from a natural language request:

```bash
python start.py edit "Add a logo bottom-right and fade out audio in last 2s" --video path/to/video.mp4 --output out.mp4
```

Use `--dry-run` to skip execution and only print the command.

### Available tools (used by the agent)
- `initial_video_analysis` (ffprobe)
- `apply_overlay_filter` (overlays, watermarks, picture-in-picture)
- `apply_text_filter` (text overlays, timed text)
- `apply_transition_filter` (concat, crossfade, split-join, stacks)
- `apply_sync_filter` (audio/video sync fixes)
- `apply_compression` (size reduction)
- `analyze_video` (Gemini content analysis, only if `GEMINI_API_KEY` is set)
- `transcript_video` (Deepgram SRT, only if `DEEPGRAM_API_KEY` is set)

### Logging
Structured logs via `structlog` go to stdout. On startup, you’ll see messages indicating whether Gemini and Deepgram features are enabled.

### Notes
- Ensure FFmpeg/ffprobe are installed: `ffmpeg -version`, `ffprobe -version`
- Some operations may download/process large files—mind timeouts and disk space
- The agent emits copy-ready commands and an explanation; confirm before execution

### Development
- Code style: Python 3.11+, type hints, clear function/variable naming
- Main entry points: `start.py` (CLI), `main.py` (agent), `planner.py` (planner)
- Tools live under `tools/` and are registered via `tools/__init__.py`

### License
MIT License
