# Tools package for FFmpeg Agent
# Import all tools to register them with main_agent

# Analysis tools
from .analysis import initial_video_analysis
from .content_analysis import analyze_video

# Filter tools
from .overlay import apply_overlay_filter
from .audio import apply_audio_filter
from .transition import apply_transition_filter
from .text import apply_text_filter
from .sync import apply_sync_filter

# Utility tools
from .doctor import doctor_command
from .user_input import ask_user_for_clarification

__all__ = [
    'initial_video_analysis',
    'analyze_video',
    'apply_overlay_filter',
    'apply_audio_filter',
    'apply_transition_filter',
    'apply_text_filter',
    'apply_sync_filter',
    'detect_object_bounds',
    'doctor_command',
    'ask_user_for_clarification'
]