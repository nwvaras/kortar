from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, RunContext, ModelRetry
import httpx
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import subprocess
from typing import List, Optional
import json
import click
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import tempfile
import os
from PIL import Image
import moonsit
# Load environment variables from .env file
load_dotenv()
print("[LOG] Environment variables loaded")



class FFmpegCommand(BaseModel):
    command: str
    explanation: str
    filters_used: List[str]

# Main FFmpeg agent with OpenAI 4o
main_agent = Agent(
    'openai:gpt-4.1',
    output_type=FFmpegCommand,
    system_prompt="""
    You are an expert FFmpeg video editing orchestrator. Your job is to plan and execute complex video editing workflows by coordinating specialized filter agents.

    ## Your Process:
    1. Analyze the user request and create a logical plan of filters to apply
    2. Consider filter dependencies and order (e.g., video analysis first, then overlays, then audio, then transitions)
    3. Start with a base FFmpeg command and incrementally build it using your tools
    4. Each tool will modify the current command and return the updated version
    5. Apply filters in the correct logical order

    ## Available Tools:
    - initial_video_analysis: Run ffprobe to get technical video characteristics (duration, resolution, fps, codecs, audio info)
    - analyze_video: Analyze video content based on specific queries using Gemini vision. Don't ask for pixels or positions
    - get_width_height: Get exact video dimensions for precise positioning, cropping, overlay placement, and aspect ratio calculations
    - apply_overlay_filter: Add overlays (watermarks, picture-in-picture, logos)
    - apply_audio_filter: Modify audio (mixing, fading, filtering)
    - apply_transition_filter: Add transitions and concatenations
    - apply_text_filter: Add text overlays and timed text
    - apply_sync_filter: Synchronize audio/video
    - detect_object_bounds: Extract frame at timestamp and detect object bounding coordinates using AI vision for precise video cropping
    - doctor_command: Fix a command that is not working.

    Every tool needs a request, and the request should have all the information to apply the filter.
    Like the time interval and other information needed.
    
    If a command fails or seems problematic, use doctor_command to fix technical issues.

    ## Filter Order Guidelines:
    1. Initial video analysis with ffprobe (to understand technical characteristics)
    2. Content analysis with Gemini (if needed for interval-based editing)
    3. Object detection with detect_object_bounds (if cropping specific objects)
    4. Base command setup
    5. Video filters (overlays, transitions, cropping)
    6. Audio filters
    7. Text overlays
    8. Sync adjustments

    Start with a base command like: "ffmpeg -y -i input_video" and build incrementally.
    Each tool will receive the current command state and your specific request for that filter.

    ## Explanation guidelines:
    - You need to explain the command in a detailed way, like the filters used, the time interval, the reason for the filter, and the suggestion for the filter.
     Example:
     
     **Interval 1**
        *   **Time:** 00:08 - 00:10
        *   **Action:** **trim**
        *   **Command:** -filter_complex "[0:v]trim=0:8,setpts=PTS-STARTPTS[v];[0:a]atrim=0:8,asetpts=PTS-STARTPTS[a];[v][a]concat=n=2:v=1:a=1[out]"
        *   **Reason:** This command trims both video and audio streams to 8 seconds, then resets their timestamps to start from 0 using setpts/asetpts filters. The trimmed streams are labeled [v] and [a], then concatenated together using the concat filter which combines them into a single output stream [out]. This maintains perfect audio/video sync while trimming.

    ## Notes:
    - When cropping, try to always be conservative with the crop. It's better to have a little bit of unwanted content than to have a cropped video that is not what the user asked for.
    - Analyze_video will only show you an aproximate position of the object.
    
    ## CRITICAL: H.264 Codec Requirements
    - The libx264 codec requires BOTH width and height to be divisible by 2 (even numbers)
    - If crop results in odd dimensions (e.g., 2542x1323), the encoding will FAIL
    - ALWAYS ensure crop dimensions are even: crop=w:h:x:y where w and h are even
    - Examples:
      * WRONG: crop=1325:753:100:200 (odd width and height)
      * CORRECT: crop=1324:752:100:200 (even width and height)
    - If needed, add scale filter to force even dimensions: scale=trunc(iw/2)*2:trunc(ih/2)*2
    - Alternative: use -vf "crop=w:h:x:y,scale=trunc(iw/2)*2:trunc(ih/2)*2" to auto-fix odd dimensions
    
    ## Overlay Expression Guidelines
    - Prefer simple, reliable patterns: overlay=10:10, overlay=W-w-10:H-h-10
    - For timed effects, use: overlay=10:10:enable='between(t,5,15)'
    - For animations, use simple linear: overlay=10+t*20:10
    - FORBIDDEN: sin(), cos(), tan() functions (cause parse errors)
    - FORBIDDEN: undefined variables like main_dur, main_d, video_length
    - AVOID complex nested expressions that often cause syntax errors
    - Test with simple static overlays first, then add time-based effects if needed
    """,
    retries=3
)

# Video analysis tool using Gemini
@main_agent.tool
async def initial_video_analysis(ctx: RunContext, video_path: str) -> str:
    """Run ffprobe to analyze video technical characteristics"""
    print(f"[LOG] FFPROBE - Analyzing: {video_path}")
    
    try:
        # Run ffprobe to get detailed video information
        ffprobe_cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(
            ffprobe_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            error_msg = f"ffprobe failed: {result.stderr}"
            print(f"[LOG] FFPROBE - Error: {error_msg}")
            return f"Error analyzing video: {error_msg}"
        
        # Parse JSON output
        probe_data = json.loads(result.stdout)
        
        # Extract relevant information
        format_info = probe_data.get('format', {})
        streams = probe_data.get('streams', [])
        
        # Find video and audio streams
        video_stream = None
        audio_streams = []
        
        for stream in streams:
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_streams.append(stream)
        
        # Build analysis report
        analysis = []
        analysis.append(f"**File:** {video_path}")
        analysis.append(f"**Duration:** {format_info.get('duration', 'unknown')} seconds")
        analysis.append(f"**Size:** {format_info.get('size', 'unknown')} bytes")
        analysis.append(f"**Format:** {format_info.get('format_name', 'unknown')}")
        
        if video_stream:
            width = video_stream.get('width', 'unknown')
            height = video_stream.get('height', 'unknown')
            fps_data = video_stream.get('r_frame_rate', '0/1').split('/')
            fps = round(int(fps_data[0]) / int(fps_data[1]), 2) if len(fps_data) == 2 and fps_data[1] != '0' else 'unknown'
            
            analysis.append(f"**Video Resolution:** {width}x{height}")
            analysis.append(f"**Video FPS:** {fps}")
            analysis.append(f"**Video Codec:** {video_stream.get('codec_name', 'unknown')}")
            analysis.append(f"**Video Bitrate:** {video_stream.get('bit_rate', 'unknown')} bps")
        else:
            analysis.append("**Video:** No video stream found")
        
        if audio_streams:
            analysis.append(f"**Audio Streams:** {len(audio_streams)}")
            for i, audio in enumerate(audio_streams):
                analysis.append(f"  - Stream {i}: {audio.get('codec_name', 'unknown')}, {audio.get('channels', 'unknown')} channels, {audio.get('sample_rate', 'unknown')} Hz")
        else:
            analysis.append("**Audio:** No audio streams found")
        
        final_analysis = "\n".join(analysis)
        print(f"[LOG] FFPROBE - Result: {final_analysis}")
        return final_analysis
        
    except subprocess.TimeoutExpired:
        error_msg = "ffprobe command timed out"
        print(f"[LOG] FFPROBE - Error: {error_msg}")
        return f"Error: {error_msg}"
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse ffprobe output: {str(e)}"
        print(f"[LOG] FFPROBE - Error: {error_msg}")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error during video analysis: {str(e)}"
        print(f"[LOG] FFPROBE - Error: {error_msg}")
        return f"Error: {error_msg}"

@main_agent.tool
async def analyze_video(ctx: RunContext, video_path: str, query: str) -> str:
    """Analyze video based on a specific query, identifying relevant intervals and actionable insights.
    
    Common types of analysis you can ask:
    - Empty spaces or moments where nothing happens
    - Parts that can be cut
    - Segments that need improvements
    - Key moments/highlights
    - Abrupt transitions
    - Audio/video issues
    - Repetitive moments
    - Long pauses
    - Silent segments
    - Low activity periods
    - Detect elements in the video
    """
    print(f"[LOG] Analyzing video with query: {query}")
    
    from pydantic_ai import Agent
    
    gemini_agent = Agent(
        'google-gla:gemini-2.5-flash',
        system_prompt="""
        You are a professional video editor analyzing content based on specific queries. Your task is to identify time intervals in the video that are relevant to the user's query and provide actionable insights.

        For each relevant interval you identify, provide:
        - Exact start and end times (use MM:SS format for videos under 60 minutes, HH:MM:SS for longer)
        - Clear description of what happens in that segment and why it's relevant to the query
        - If the user ask for detect an object, provide an approximate position of the object in the frame.
        - Specific action suggestion: "trim", "enhance", "keep", "add_transition", "adjust_audio", "add_effect", or "observation"
        
        <examples>
        <example>
        User query: "Empty spaces or moments where nothing happens?"
        
        Answer:
        **Interval 1**
        *   **Time:** 00:00 - 00:05
        *   **Description:** Static title screen with no movement or action.
        *   **Action:** **observation**
        *   **Suggestion:** None.

        **Interval 2**
        *   **Time:** 00:15 - 00:25
        *   **Description:** Extended sequence of routine file operations with no significant events.
        *   **Action:** **observation**
        *   **Suggestion:** None.

        **Interval 3**
        *   **Time:** 00:35 - 00:45
        *   **Description:** Loading screen with spinning progress indicator.
        *   **Action:** **observation**
        *   **Suggestion:** None.

        **Interval 4**
        *   **Time:** 01:05 - 01:15
        *   **Description:** Repeated navigation through same menu items without selection.
        *   **Action:** **observation**
        *   **Suggestion:** None.
        </examples>
        <examples>

        User query: "Where is the ball?"
        
        Answer:
        **Interval 1**
        *   **Time:** 00:05 - 00:08
        *   **Description:** A red ball appears bouncing from the left edge and moves across the middle third of the frame, taking up about 1/3 of the screen width as it bounces.
        *   **Action:** **observation**
        *   **Suggestion:** This is the first appearance of the ball in the video.

        **Interval 2**
        *   **Time:** 00:08 - 00:12 
        *   **Description:** The ball continues bouncing but starts to slow down and rolls towards the right edge of the frame, eventually exiting completely.
        *   **Action:** **observation**
        *   **Suggestion:** The ball's first appearance ends here as it exits the scene.

        **Interval 3**
        *   **Time:** 00:15 - 00:18
        *   **Description:** The ball reappears from the top of the frame in the center area, dropping straight down while staying in the middle third of the screen width.
        *   **Action:** **observation**
        *   **Suggestion:** Second appearance of the ball with a different motion pattern.

        **Interval 4**
        *   **Time:** 00:18 - 00:20
        *   **Description:** The ball bounces once near the bottom of the frame and rolls diagonally towards the center-left area, disappearing behind some objects in the scene.
        *   **Action:** **observation**
        *   **Suggestion:** Final appearance of the ball before it's completely out of view.
        </example>
        </examples>
        Be precise with timing and practical with suggestions. Only include intervals that directly answer the user's query.
        """
    )
    
    video_content = await load_video_as_binary(video_path)
    
    result = await gemini_agent.run([
        video_content,
        f'Query: {query}'
    ])
    print(result.output)
    return result.output

async def load_video_as_binary(video_path: str) -> BinaryContent:
    """Load video file as binary content"""
    print(f"[LOG] Loading video from: {video_path}")
    
    if video_path.startswith('http'):
        print("[LOG] Downloading video from URL...")
        async with httpx.AsyncClient() as client:
            response = await client.get(video_path)
            response.raise_for_status()
            content_type = response.headers.get('content-type', 'video/mp4')
            return BinaryContent(data=response.content, media_type=content_type)
    else:
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with open(video_path_obj, 'rb') as f:
            video_data = f.read()

        ext = video_path_obj.suffix.lower()
        media_type_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/avi',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.flv': 'video/x-flv'
        }
        media_type = media_type_map.get(ext, 'video/mp4')
        
        return BinaryContent(data=video_data, media_type=media_type)

# Specialized filter agents
overlay_agent = Agent(
    'openai:gpt-4.1-mini',
    output_type=str,
    system_prompt="""
You are an FFmpeg overlay expert. Your task is to modify a given FFmpeg command to add overlay effects (e.g., watermark, picture-in-picture, logo) via -filter_complex.

You will receive:
1. The original FFmpeg command
2. A specific overlay request

Your job is to modify the command accordingly, while preserving existing filters and chaining everything properly.

=== OVERLAY DESIGN RULES ===

✅ SAFE PATTERNS (Use these):
- Static position: overlay=10:10 or overlay=W-w-10:H-h-10
- Timed overlays: overlay=10:10:enable='between(t,5,15)'
- Basic movement: overlay=10+t*20:10
- Fades: overlay=10:10:eval=frame:alpha='min(1,t/2)'
- Proper chains: [logo]scale=100:100[s];[base][s]overlay=10:10[out]

❌ AVOID (These cause errors):
- Trigonometric functions: sin(), cos(), etc.
- Nested conditions or complex math: e.g., 'W-w-10 + sin(t*4)*20'
- Undefined vars: main_dur, video_length, etc.
- Invalid scale math: scale=-1:H/8
- Multiple functions inside expressions

=== CRITICAL SYNTAX RULES ===
- Always quote time expressions: enable='gte(t,10)'
- Use valid inputs: scale=iw/4:ih/4, not scale=-1:H/4
- Chain filters explicitly: [a]hflip[b];[video][b]overlay=...
- hflip takes one input only: [img]hflip[flipped]
- Use simple constants if dynamic timing is unknown

=== COMMON OVERLAYS ===
- Watermark: [0:v][1:v]overlay=W-w-10:H-h-10[out]
- PiP: [1:v]scale=iw/4:ih/4[pip];[0:v][pip]overlay=10:10[out]
- Logo (flipped): [1:v]hflip[flip];[0:v][flip]overlay=10:10[out]
- Timed display: overlay=10:10:enable='between(t,5,15)'[out]
- Centered: overlay=(W-w)/2:(H-h)/2

⚠️ Focus on correctness and syntax safety. Never break existing chains. Output only the full modified FFmpeg command.
""")


@main_agent.tool
async def apply_overlay_filter(ctx: RunContext, current_command: str, request: str) -> str:
    """Apply overlay effects to the current FFmpeg command"""
    print(f"[LOG] OVERLAY - Request: {request}")
    print(f"[LOG] OVERLAY - Current command: {current_command}")
    
    result = await overlay_agent.run([
        f"Current command: {current_command}",
        f"Overlay request: {request}"
    ])
    
    print(f"[LOG] OVERLAY - Result: {result.output}")
    return result.output

audio_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
    system_prompt="""
    You are an audio processing specialist. You modify FFmpeg commands to add audio effects like mixing, fading, and filtering.
    
    You will receive:
    1. Current FFmpeg command
    2. Specific audio request
    
    Your job is to modify the command to add the requested audio processing using -filter_complex.
    
    Common audio patterns:
    - Mix audio: [0:a][1:a]amix=inputs=2:duration=longest:weights='1 0.5'[mix]
    - Audio fade: [0:a]afade=t=in:st=0:d=2,afade=t=out:st=58:d=2[out]
    - Bandpass filter: [0:a]highpass=f=200,lowpass=f=3000[out]
    - Volume adjustment: [0:a]volume=0.5[out]
    
    Always preserve existing filters and properly chain them. Return only the modified FFmpeg command.
    """
)

@main_agent.tool
async def apply_audio_filter(ctx: RunContext, current_command: str, request: str) -> str:
    """Apply audio effects to the current FFmpeg command"""
    print(f"[LOG] AUDIO - Request: {request}")
    print(f"[LOG] AUDIO - Current command: {current_command}")
    
    result = await audio_agent.run([
        f"Current command: {current_command}",
        f"Audio request: {request}"
    ])
    
    print(f"[LOG] AUDIO - Result: {result.output}")
    return result.output

transition_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
    system_prompt="""
    You are a video transition specialist. You modify FFmpeg commands to add transitions, concatenations, and video arrangements.
    
    You will receive:
    1. Current FFmpeg command
    2. Specific transition request
    
    Your job is to modify the command to add the requested transitions using -filter_complex.
    
    CRITICAL: When trimming segments for concatenation, ALWAYS use setpts=PTS-STARTPTS to reset timestamps:
    - CORRECT: [0:v]trim=start=1:end=11,setpts=PTS-STARTPTS[v0]
    - WRONG: [0:v]trim=start=1:end=11[v0]
    
    Common transition patterns:
    - Trim+Concat: [0:v]trim=start=1:end=11,setpts=PTS-STARTPTS[v0];[0:v]trim=start=23:end=30,setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[out]
    - Crossfade: [0:v][1:v]xfade=transition=fade:duration=1:offset=4[v];[0:a][1:a]acrossfade=d=1:c1=tri:c2=tri[a]
    - Multi-file Concat: [0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]
    - Side-by-side: [0:v][1:v]hstack=inputs=2[out]
    - Split comparison: [0:v]split=2[orig][dup];[dup]hue=s=0[gray];[orig][gray]hstack=2[out]
    
    Timestamp Management Rules:
    1. Always use setpts=PTS-STARTPTS after trim operations
    2. For audio: use asetpts=PTS-STARTPTS after atrim
    3. This ensures proper segment joining without time gaps or extensions
    4. Final duration = sum of trimmed segment durations (no more, no less)
    
    Always preserve existing filters and properly chain them. Return only the modified FFmpeg command.
    If you have different width/height inputs, keep the same aspect ratio. Even if this means adding more black bars.
    """
)

@main_agent.tool
async def apply_transition_filter(ctx: RunContext, current_command: str, request: str) -> str:
    """Apply transition effects to the current FFmpeg command. The request should say if the video has or doesn't have audio
    Send the aspect ratio of the parts of the video, if you want to concatenate them."""
    print(f"[LOG] TRANSITION - Request: {request}")
    print(f"[LOG] TRANSITION - Current command: {current_command}")
    
    result = await transition_agent.run([
        f"Current command: {current_command}",
        f"Transition request: {request}"
    ])
    
    print(f"[LOG] TRANSITION - Result: {result.output}")
    return result.output

text_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
    system_prompt="""
    You are a text overlay specialist. You modify FFmpeg commands to add text overlays, captions, and timed text.
    
    You will receive:
    1. Current FFmpeg command
    2. Specific text request
    
    Your job is to modify the command to add the requested text using -filter_complex.
    
    Common text patterns:
    - Static overlay: [0:v]drawtext=text='Sample Text':fontsize=36:fontcolor=white:x=(w-text_w)-10:y=(h-text_h)-10:shadowx=2:shadowy=2[out]
    - Timed text: [0:v]drawtext=text='Timed Text':x=100:y=100:enable='gte(t,5)'[out]
    - Centered text: [0:v]drawtext=text='Center':x=(w-text_w)/2:y=(h-text_h)/2[out]
    - Subtitle style: [0:v]drawtext=text='Subtitle':x=(w-text_w)/2:y=h-100:fontsize=24:box=1:boxcolor=black@0.5[out]
    
    Always preserve existing filters and properly chain them. Return only the modified FFmpeg command.
    """
)

@main_agent.tool
async def apply_text_filter(ctx: RunContext, current_command: str, request: str) -> str:
    """Apply text overlays to the current FFmpeg command"""
    print(f"[LOG] TEXT - Request: {request}")
    print(f"[LOG] TEXT - Current command: {current_command}")
    
    result = await text_agent.run([
        f"Current command: {current_command}",
        f"Text request: {request}"
    ])
    
    print(f"[LOG] TEXT - Result: {result.output}")
    return result.output

sync_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
    system_prompt="""
    You are an audio/video synchronization specialist. You modify FFmpeg commands to fix sync issues and timing problems.
    
    You will receive:
    1. Current FFmpeg command
    2. Specific sync request
    
    Your job is to modify the command to add the requested sync adjustments using -filter_complex.
    
    Common sync patterns:
    - Audio delay: [1:a]adelay=5000:all=1[aud]
    - Video delay: [0:v]setpts=PTS+5/TB[out]
    - Audio speed: [0:a]atempo=1.2[out]
    - Video speed: [0:v]setpts=PTS/1.2[out]
    
    Always preserve existing filters and properly chain them. Return only the modified FFmpeg command.
    """
)

# Command doctor agent for fixing problematic commands
doctor_agent = Agent(
    'openai:gpt-4.1-mini',
    output_type=str,
    system_prompt="""
You are an FFmpeg command doctor. Your role is to analyze invalid FFmpeg commands, identify syntax or filter issues, and return a corrected version that preserves the user’s original intent.

You will receive:
1. A brief intent (what the user was trying to achieve)
2. The failing FFmpeg command
3. An optional error message

Return only the corrected FFmpeg command. Do not include explanation or comments.

---

=== 🔧 CORE FIXING STRATEGY ===
- Fix only what’s broken
- Preserve visual effects (e.g., trim, movement, overlays)
- Remove or simplify unsafe expressions
- Ensure syntax is valid and filter chains are properly labeled

---

=== ❌ UNSAFE / ERROR-PRONE EXPRESSIONS TO REMOVE ===
- Trigonometric functions: `sin()`, `cos()`, `tan()` → ❌
- Complex conditionals or nesting: `if(mod(...))` → ❌
- Floating-point math inside coordinates (e.g., `/3.0`, `*0.5`) → ❌ safer as integers
- Unsupported math functions: `mod()`, `hypot()` → ❌
- Nonexistent variables: `main_dur`, `video_length`, etc. → ❌

✅ Use:
- Simple integer math: `(t-3)*speed`
- Safe horizontal motion: `x=10+(t-3)*30`
- Static y positioning: `y=H-h-10`
- Constant timing: `enable='gte(t,3)'`, `between(t,5,10)`

---

=== 🛠 STRUCTURAL RULES ===
- `scale` must use `iw`, `ih`, or fixed values (never `H`)
- `overlay` requires 2 inputs, `hflip` 1 input
- `rotate` requires fixed `ow`, `oh`, and `:c=none`
- Always quote expressions: `enable='gte(t,5)'`
- No decimals inside `enable=` or coordinate math
- Chain filters explicitly: `[in1]filter[label];[label][in2]filter[out]`

---

=== ✅ SAFE PATTERN EXAMPLES ===
- Horizontal slide-in: `overlay=x=10+(t-3)*30:y=H-h-10`
- Timed overlay: `overlay=10:10:enable='between(t,5,8)'`
- Flipped logo overlay: `[img]hflip[flipped];[base][flipped]overlay=...`
- End-of-video reveal: `enable='gte(t,10)'`
- Basic animation: use linear motion only, no `mod()`, `sin()`, etc.

---

=== 🧠 TYPICAL ISSUES TO FIX ===
- Invalid or complex math expressions
- Filter chain structure errors
- Missing inputs or labels
- Broken audio/video mappings
- Codec compatibility (e.g., VP8 in MP4)
- Missing `-y` overwrite
- Incorrect stream maps or filter labels
- Use of decimals where integers are required

---

Your job is to fix the command and return it in full.
❗ Return only the corrected FFmpeg command — nothing else.
""")

@main_agent.tool
async def apply_sync_filter(ctx: RunContext, current_command: str, request: str) -> str:
    """Apply synchronization adjustments to the current FFmpeg command"""
    print(f"[LOG] SYNC - Request: {request}")
    print(f"[LOG] SYNC - Current command: {current_command}")
    
    result = await sync_agent.run([
        f"Current command: {current_command}",
        f"Sync request: {request}"
    ])
    
    print(f"[LOG] SYNC - Result: {result.output}")
    return result.output

@main_agent.tool
async def doctor_command(ctx: RunContext, intent: str, failing_command: str, error_message: str = "") -> str:
    """Fix problematic FFmpeg commands by analyzing the intent and fixing technical issues"""
    print(f"[LOG] DOCTOR - Intent: {intent}")
    print(f"[LOG] DOCTOR - Failing command: {failing_command}")
    print(f"[LOG] DOCTOR - Error: {error_message}")
    
    inputs = [
        f"Original intent: {intent}",
        f"Failing command: {failing_command}"
    ]
    
    if error_message:
        inputs.append(f"Error message: {error_message}")
    
    result = await doctor_agent.run(inputs)
    
    print(f"[LOG] DOCTOR - Fixed command: {result.output}")
    return result.output

async def detect_object_bounds(ctx: RunContext, video_path: str, timestamp: str, object_name:str) -> str:
    """Extract a frame at the given timestamp and detect bounding box coordinates for the specified object using moondream vision model
    Use just the object name, not a position."""
    print(f"[LOG] OBJECT_DETECTION - Video: {video_path}, Timestamp: {timestamp}, Object: {object_name}")
    
    try:
        # Create temporary directory for frame extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = os.path.join(temp_dir, "frame.jpg")
            
            # Extract frame at timestamp using ffmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', timestamp,
                '-frames:v', '1',
                '-q:v', '2',  # High quality
                frame_path
            ]
            
            print(f"[LOG] OBJECT_DETECTION - Extracting frame: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                error_msg = f"Frame extraction failed: {result.stderr}"
                print(f"[LOG] OBJECT_DETECTION - Error: {error_msg}")
                return f"Error extracting frame: {error_msg}"
            
            # Check if frame was created
            if not os.path.exists(frame_path):
                error_msg = "Frame file not created"
                print(f"[LOG] OBJECT_DETECTION - Error: {error_msg}")
                return f"Error: {error_msg}"
            
            # Load frame with PIL
            try:
                image = Image.open(frame_path)
                print(f"[LOG] OBJECT_DETECTION - Frame loaded: {image.size}")
            except Exception as e:
                error_msg = f"Failed to load frame image: {str(e)}"
                print(f"[LOG] OBJECT_DETECTION - Error: {error_msg}")
                return f"Error: {error_msg}"
            
            # Initialize moondream model
            try:
                # Use the moonsit module's get_points function
                point_output = moonsit.detect_object_bounds(image, object_name)
                print(f"[LOG] OBJECT_DETECTION - Raw points: {point_output}")
                
                # Parse the points to get bounding box coordinates
                # moondream.point typically returns coordinates in some format
                # We need to convert this to FFmpeg crop parameters
                
                # Extract image dimensions for reference
                img_width, img_height = image.size
                
                result_text = f"""**Object Detection Results for {object_name} at {timestamp}:**

**Image Size:** {img_width}x{img_height}
**Detection Points:** {point_output['pixels']}

**For FFmpeg Cropping:**
Based on the detected points, you can use a crop filter like:
`-filter_complex "[0:v]crop=w:h:x:y[cropped]"`

Where:
- w = width of the crop area
- h = height of the crop area  
- x = x coordinate of top-left corner
- y = y coordinate of top-left corner

The detection points above indicate where the {object_name} is located in the frame."""
                
                print(f"[LOG] OBJECT_DETECTION - Success: {result_text}")
                return result_text
                
            except Exception as e:
                error_msg = f"Moondream detection failed: {str(e)}"
                print(f"[LOG] OBJECT_DETECTION - Error: {error_msg}")
                return f"Error during object detection: {error_msg}"
                
    except subprocess.TimeoutExpired:
        error_msg = "Frame extraction timed out"
        print(f"[LOG] OBJECT_DETECTION - Error: {error_msg}")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error during object detection: {str(e)}"
        print(f"[LOG] OBJECT_DETECTION - Error: {error_msg}")
        return f"Error: {error_msg}"


@main_agent.output_validator
async def validate_ffmpeg_command(ctx: RunContext, output: FFmpegCommand) -> FFmpegCommand:
    """Validate the final FFmpeg command"""
    print(f"[LOG] Validating FFmpeg command: {output.command}")
    
    if not output.command.strip().lower().startswith('ffmpeg'):
        raise ModelRetry('The command must start with "ffmpeg". Please generate a valid FFmpeg command.')
    
    # Add -y flag if not present
    if ' -y ' not in output.command and not output.command.startswith('ffmpeg -y'):
        output.command = output.command.replace('ffmpeg ', 'ffmpeg -y ', 1)
    
    try:
        # Test command syntax without actual execution
        test_command = output.command.replace(' -c:v libx264', ' -f null').replace(' output.', ' -f null /dev/null 2>&1; echo $?')
        
        # Basic syntax validation
        if '-filter_complex' not in output.command and len(output.filters_used) > 0:
            raise ModelRetry('Command should use -filter_complex for the specified filters.')
        
        print("[LOG] Command validation successful")
        return output
        
    except Exception as e:
        print(f"[ERROR] Command validation failed: {str(e)}")
        raise ModelRetry(f'Command validation error: {str(e)}') from e

# Initialize rich console for better CLI experience
console = Console()
app = typer.Typer(
    name="ffmpeg-agent",
    help="🎬 FFmpeg Agent v3 - AI-powered video editing assistant",
    add_completion=False,
    rich_markup_mode="rich"
)

@app.command("interactive")
def interactive_mode():
    """🚀 Start interactive mode for conversational video editing"""
    console.print(Panel.fit(
        "[bold blue]🎬 FFmpeg Agent v3 - Interactive Mode[/bold blue]\n\n"
        "[yellow]Available commands:[/yellow]\n"
        "• Analyze video technical details\n"
        "• Apply filters and effects\n"
        "• Find editing opportunities\n"
        "• Fix problematic commands\n\n"
        "[green]Multiline Input Support:[/green]\n"
        "• After typing first line, continue on next lines\n"
        "• Press Enter on empty line to submit\n"
        "• Use '\\' at end of line for forced continuation\n\n"
        "[dim]Commands: 'help' for help, 'clear' to reset, 'quit' to exit[/dim]",
        title="Welcome",
        border_style="blue"
    ))
    
    asyncio.run(_interactive_session())

@app.command("analyze")
def analyze_video_file(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    technical: bool = typer.Option(False, "--technical", "-t", help="Run technical analysis with ffprobe"),
    content: bool = typer.Option(False, "--content", "-c", help="Run content analysis with AI"),
    query: str = typer.Option("", "--query", "-q", help="Specific query for content analysis")
):
    """📊 Analyze video file (technical specs and/or content)"""
    
    if not technical and not content:
        technical = True  # Default to technical analysis
    
    asyncio.run(_analyze_video(video_path, technical, content, query))

@app.command("edit")
def edit_video(
    request: str = typer.Argument(..., help="Video editing request"),
    video_path: str = typer.Option("", "--video", "-v", help="Video file path (if not in request)"),
    output: str = typer.Option("", "--output", "-o", help="Output file name"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show command without executing")
):
    """✨ Apply video editing effects based on natural language request"""
    
    full_request = f"{request} {video_path}".strip() if video_path else request
    if output:
        full_request += f" save as {output}"
    
    asyncio.run(_process_edit_request(full_request, dry_run))

@app.command("doctor")
def doctor_command_cli(
    intent: str = typer.Argument(..., help="Original intent of the command"),
    failing_command: str = typer.Argument(..., help="The FFmpeg command that failed"),
    error: str = typer.Option("", "--error", "-e", help="Error message (optional)")
):
    """🩺 Fix a problematic FFmpeg command"""
    
    asyncio.run(_doctor_fix(intent, failing_command, error))

async def _interactive_session():
    """Internal interactive session handler"""
    console.print("[LOG] Starting FFmpeg Agent v3...", style="dim")
    console.print("[dim]💡 Multiline support: Continue typing on next lines, press Enter on empty line to submit[/dim]")
    console.print("[dim]💡 Commands: 'help' for help, 'clear' to reset, 'quit' to exit[/dim]")
    
    history = []
    while True:
        try:
            # Support multiline input
            console.print()
            console.print("[dim]Enter your request:[/dim]")
            
            user_input = ""
            try:
                # Get first line
                first_line = input("❯ ").strip()
                
                # Check for special commands
                if first_line.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    return
                elif first_line.lower() == 'clear':
                    history = []
                    console.print("[green]✨ Chat history cleared![/green]")
                    continue
                elif first_line.lower() in ['help', '?']:
                    console.print(Panel.fit(
                        "[bold yellow]🎬 FFmpeg Agent v3 - Help[/bold yellow]\n\n"
                        "[green]Available Commands:[/green]\n"
                        "• quit/exit/q - Exit the program\n"
                        "• clear - Clear chat history\n"
                        "• help/? - Show this help\n\n"
                        "[green]Multiline Input:[/green]\n"
                        "• After first line, continue typing on next lines\n"
                        "• Press Enter on empty line to submit\n"
                        "• Use '\\' at end of line for forced continuation\n"
                        "• Example:\n"
                        "  [dim]❯ Analyze video.mp4 and\n"
                        "  ... find all the moments where\n"
                        "  ... nothing is happening\n"
                        "  ... [press Enter on empty line][/dim]\n\n"
                        "[green]Common Requests:[/green]\n"
                        "• Analyze video for editing opportunities\n"
                        "• Crop/trim specific sections\n"
                        "• Add overlays, text, transitions\n"
                        "• Fix problematic FFmpeg commands",
                        title="Help",
                        border_style="yellow"
                    ))
                    continue
                
                user_input = first_line
                
                # Simple multiline support - always check for more input
                console.print("[dim](Press Enter on empty line to submit, or continue typing)[/dim]")
                while True:
                    try:
                        if user_input.endswith('\\'):
                            # Remove backslash and continue
                            user_input = user_input[:-1] + " "
                            next_line = input("... ").strip()
                            user_input += next_line
                        else:
                            # Check for additional lines
                            next_line = input("... ")
                            if not next_line.strip():  # Empty line = done
                                break
                            user_input += " " + next_line.strip()
                    except EOFError:
                        break
                    
            except EOFError:
                # Handle Ctrl+D
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Show processing indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing request...", total=None)
                
                result = await main_agent.run(
                    user_input, message_history=history
                )
                progress.update(task, description="Complete!")
            
            history = result.all_messages()
            
            # Display results in a nice format
            _display_result(result.output)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")

async def _analyze_video(video_path: str, technical: bool, content: bool, query: str):
    """Internal video analysis handler"""
    console.print(f"[LOG] Analyzing video: {video_path}", style="dim")
    
    try:
        if technical:
            console.print("[blue]🔍 Running technical analysis...[/blue]")
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                task = progress.add_task("Analyzing with ffprobe...", total=None)
                tech_result = await initial_video_analysis(None, video_path)
                
            console.print(f"\n[bold blue]🔍 Technical Analysis:[/bold blue]")
            console.print(f"[dim]{'─' * 60}[/dim]")
            # Print analysis with no formatting for easy copying
            print(tech_result)
            console.print(f"[dim]{'─' * 60}[/dim]")
        
        if content:
            content_query = query if query else "Analyze this video for editing opportunities"
            console.print(f"[green]🎯 Running content analysis: {content_query}[/green]")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                task = progress.add_task("Analyzing with AI...", total=None)
                content_result = await analyze_video(None, video_path, content_query)
                
            console.print(f"\n[bold green]🎯 Content Analysis:[/bold green]")
            console.print(f"[dim]{'─' * 60}[/dim]")
            # Print analysis with no formatting for easy copying
            print(content_result)
            console.print(f"[dim]{'─' * 60}[/dim]")
            
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")

async def _process_edit_request(request: str, dry_run: bool):
    """Internal edit request handler"""
    console.print(f"[LOG] Processing edit request: {request}", style="dim")
    
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("Generating FFmpeg command...", total=None)
            result = await main_agent.run(request)
        
        _display_result(result.output)
        
        if dry_run:
            console.print("[yellow]📋 Dry run mode - command not executed[/yellow]")
        else:
            if Confirm.ask("Execute this command?"):
                console.print("[green]🚀 Executing command...[/green]")
                # Here you could add actual command execution
                console.print("[yellow]⚠️ Command execution not implemented yet[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Edit request failed: {str(e)}[/red]")

async def _doctor_fix(intent: str, failing_command: str, error: str):
    """Internal doctor fix handler"""
    console.print("[blue]🩺 Analyzing and fixing command...[/blue]")
    
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("Diagnosing issues...", total=None)
            fixed_command = await doctor_command(None, intent, failing_command, error)
        
        console.print(f"\n[bold red]❌ Original Command:[/bold red]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        # Print commands with no formatting for easy copying
        print(failing_command)
        console.print(f"[dim]{'─' * 60}[/dim]")
        
        console.print(f"\n[bold green]✅ Fixed Command (copy-ready):[/bold green]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        print(fixed_command)
        console.print(f"[dim]{'─' * 60}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Doctor fix failed: {str(e)}[/red]")

def _display_result(output):
    """Display command result in a copy-friendly format"""
    
    # Display the command in a copy-friendly format first
    console.print("\n[bold cyan]📋 FFmpeg Command (copy-ready):[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    # Print command with no formatting at all for easy copying
    print(output.command)
    console.print(f"[dim]{'─' * 60}[/dim]")
    
    # Then show additional details in a formatted way
    console.print(f"\n[bold yellow]📝 Explanation:[/bold yellow]")
    # Print explanation with no formatting for easy copying if needed
    print(output.explanation)
    
    if output.filters_used:
        console.print(f"\n[bold magenta]🔧 Filters Used:[/bold magenta]")
        print(", ".join(output.filters_used))

# Legacy main function for backwards compatibility
async def main():
    console.print("[LOG] Starting legacy interactive mode...", style="dim")
    await _interactive_session()

if __name__ == "__main__":
    # Check if any arguments were passed, if not, start interactive mode
    import sys
    if len(sys.argv) == 1:
        console.print("[yellow]No command specified, starting interactive mode...[/yellow]")
        asyncio.run(_interactive_session())
    else:
        app() 