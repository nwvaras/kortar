from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from main import main_agent
from common.logger import get_logger
from common.validators import validate_ffmpeg_filter_complex

logger = get_logger("kortar.tools.overlay")

overlay_agent = Agent(
    "openai:gpt-4.1",
    output_type=str,
    result_retries=3,
    system_prompt="""
You are an FFmpeg command specialist. Your task is to accept as input:

- The original FFmpeg command (may include -filter_complex or multiple input files)
- An overlay or zoom request

Your goal is to produce a single, syntactically valid FFmpeg command where -filter_complex is constructed or modified according to the overlay/zoom request and all FFmpeg constraints. Crucially, *detect and avoid* any use of undefined variables or unsupported expressions (such as 'duration' in overlay expressions) and replace them with well-supported alternatives. If an error-prone variable or expression is requested, substitute with simple, FFmpeg-compatible values or supported per-frame variables (`t`, `W`, `w`, etc.) to achieve the closest possible intent.

# Required Behaviors

- Compose the full -filter_complex graph as one compact, properly quoted shell string (no line breaks, backslashes, or explanations).
- Any branch re-use must use explicit `[label]split=n[...]` chain construction with labels for each filter stage.
- Chain and label all filter stages; always label intermediates and outputs.
- **Do not use undefined variables** (e.g. 'duration' in overlay; only built-in FFmpeg variables for the chosen filter).
- For moving overlays: Replace unsupported expressions (such as `10+t*(W-w-20)/duration`) with simple, frame-time-based math using only built-in variables (e.g., `10+t*((W-w-20)/F)`, if `F` is known/available, or else a fixed value or series for demonstration).
- Avoid complex/unsupported alpha expressions; if unsupported, use `fade=t=in:...` on the overlay image before overlay.
- Never output notes, comments, or explanations—only the final command.

# Steps

1. Parse and inspect the original FFmpeg command and overlay/zoom request.
2. Identify any unsupported filter expressions (e.g., 'duration' with overlay filter, math functions only available in later FFmpeg versions, etc.).
3. Adapt the overlay/zoom filter expression using only variables and functions natively supported by the relevant FFmpeg filter—rewrite or simplify any unsupported expressions.
4. Strictly structure the filter_complex as a labeled multi-stream graph according to requirements.
5. Quote and insert the resulting -filter_complex string in the command, in a single shell argument or argv list entry.
6. If the desired effect (e.g., smooth motion or alpha fade) cannot be achieved due to limitations, use the closest FFmpeg-supported method.
7. Output only the final, single-line FFmpeg command, with no comments or extra text.
8. Persist and refine the filter graph as needed to ensure the command is accepted and valid; continue until a valid command is produced.

# Output Format

Provide the output as a single, fully formed FFmpeg command, in one line, quoted for shell usage. Do not use line breaks, code block notation, explanations, or notes. Output only the FFmpeg command itself.

# Examples

**Example 1**
Input:
Original command: ffmpeg -i test.mp4 -i pinwi.png -filter_complex "[0:v][1:v]overlay=x=10:y=10:alpha='min(1,t/2)'[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4  
Overlay request: Add animated alpha fade-in over first 2 seconds

Output:
ffmpeg -i test.mp4 -i pinwi.png -filter_complex "[1:v]fade=t=in:st=0:d=2:alpha=1[fadepng];[0:v][fadepng]overlay=x=10:y=10[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4

**Example 2**
Input:
Original command: ffmpeg -i bg.mp4 -i fg.png -filter_complex "" -map "[out]" -c:v libx264 result.mp4  
Overlay request: Place overlay at top-right, statically

Output:
ffmpeg -i bg.mp4 -i fg.png -filter_complex "[0:v][1:v]overlay=x=W-w-10:y=10[out]" -map "[out]" -c:v libx264 result.mp4

**Example 3**
Input:
Original command: ffmpeg -i test.mp4 -i pinwi.png -filter_complex "[0:v][1:v]overlay=x='10+t*(W-w-20)/duration':y=10[out]" -map "[out]" -c:v libx264 -crf 23 output.mp4  
Overlay request: Add overlay that moves horizontally from left to right, pinwi.png

Output:
ffmpeg -i test.mp4 -i pinwi.png -filter_complex "[0:v][1:v]overlay=x='10+t*({{(W-w-20)/EXPECTED_VIDEO_DURATION}})':y=10[out]" -map "[out]" -c:v libx264 -crf 23 output.mp4  
(Or: substitute `EXPECTED_VIDEO_DURATION` with an explicit duration if available; if not, use a constant factor per time, e.g. `x='10+t*X'`, where X is computed for a desired effect. Avoid 'duration'.)

# Notes

- Never use 'duration' or undefined variables in any overlay/zoom/filter expressions.
- If a moving overlay must move left-to-right over the whole video, calculate the needed speed as `(W-w-20)/VIDEO_DURATION` and use that as the multiplier for `t`. If the duration is unknown, use a fixed 'speed' value (e.g., 50 px/sec).
- If variables like duration are unavailable in the FFmpeg context, requesters should substitute with a constant value matching their needs.
- Output only the command, no further description.

# Instructions Reminder

- Output only a valid, full FFmpeg command as a single line.
- Any undefined/unsupported variables must be eliminated or replaced with supported, constant, or calculated values.
- If stepwise refinement is needed for compatibility, persist until a valid command is achieved with no errors or unsupported syntax."""
)


# Example of chromatic abberation:

# ffmpeg -y -i input.mp4
# -filter_complex
# "[0:v]split=3;lutrgb=g=0:b=0;lutrgb=r=0:b=0;lutrgb=r=0:
# g=0;crop=iw-4:ih:4:0, pad=iw+4:ih:0:0;crop=iw-4:ih:0:0,
# pad=iw+4:ih:4:0;crop=iw:ih-4:0:4,
# pad=iw:ih+4:0:0;blend=all_mode='addition';blend=all_mod
# e='addition'" -map "" -c:v libx264 -crf 18 -preset
# veryfast output.mp4



@main_agent.tool
async def apply_overlay_filter(
    ctx: RunContext, current_command: str, request: str, video_path: str
) -> str:
    """Apply effects to the current FFmpeg command"""
    logger.info("Processing overlay effect request", 
                request=request, 
                current_command=current_command, 
                video_path=video_path)

    result = await overlay_agent.run(
        [
            f"Video path: {video_path}",
            f"Current command: {current_command}",
            f"Effect request: {request}",
        ]
    )

    logger.info("Overlay effect result generated", result=result.output)
    return result.output


@overlay_agent.output_validator
async def validate_ffmpeg_command(ctx: RunContext, output: str) -> str:
    """Validate the final FFmpeg command using the common validator"""
    is_valid, error_message, cleaned_command = validate_ffmpeg_filter_complex(output, timeout=10)
    
    if not is_valid:
        raise ModelRetry(error_message)
    
    return cleaned_command
