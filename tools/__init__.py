# Tools package for FFmpeg Agent
# Import all tools to register them with main_agent

import os
import logging

logger = logging.getLogger(__name__)

# Analysis tools
from .analysis import initial_video_analysis

# Conditional imports based on API keys
if os.getenv('GOOGLE_API_KEY'):
    from .content_analysis import analyze_video
    __gemini_tools__ = ["analyze_video"]
    logger.info("Gemini API key found - video analysis features enabled")
else:
    __gemini_tools__ = []
    logger.info("Gemini API key not found - video analysis features disabled")

if os.getenv('DEEPGRAM_API_KEY'):
    from .transcript import transcript_video
    __deepgram_tools__ = ["transcript_video"]
    logger.info("Deepgram API key found - video transcription features enabled")
else:
    __deepgram_tools__ = []
    logger.info("Deepgram API key not found - video transcription features disabled")

# Filter tools
from .overlay import apply_video_edit
from .text import apply_text_filter

# Utility tools
from .user_input import ask_user_for_clarification
from .compress import apply_compression

__all__ = [
    "initial_video_analysis",
    "apply_video_edit",
    "apply_text_filter",
    "ask_user_for_clarification",
    "apply_compression",
] + __gemini_tools__ + __deepgram_tools__
