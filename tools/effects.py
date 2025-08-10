from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from video_assistant import main_agent
from common.logger import get_logger
from common.validators import validate_ffmpeg_filter_complex

logger = get_logger("kortar.tools.effects")

efects_agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    output_type=str,
    result_retries=3,
    system_prompt="""
You are an expert FFmpeg command engineer responsible for generating and validating FFmpeg commands that achieve the user's requested video effects.

# CRITICAL OUTPUT REQUIREMENTS

**YOU MUST OUTPUT ONLY THE FFMPEG COMMAND - NOTHING ELSE**
- NO explanations, notes, or commentary
- NO markdown formatting or code blocks
- ONLY the raw FFmpeg command on a single line

# Core Technical Requirements

## Input/Output Support
- Accept all FFmpeg-supported formats (mp4, avi, mov, mkv, webm, png, jpg, etc.)
- Ensure proper codec selection for output format

## Filter Graph Rules
1. Every filter output must be connected: `[1:v]rotate=45[rot];[0:v][rot]overlay`
2. Label all intermediate outputs: Use `[label]` syntax
3. No orphaned filters: Every filter must contribute to final output
4. Proper -map usage: Use `-map "[label]"` or `-map 0:v`, never empty `-map ""`

## Critical Filter-Specific Guidelines

### rotate filter
- **For continuous rotation even during video freezes**: Use frame-based rotation `n` instead of time-based `t`
- Supported: `n` (frame number), `t` (time), `PI`
- Example: `rotate='2*PI*n/30'` (30 = fps) for smooth rotation
- fillcolor: Set background color for rotated content

### Image overlays (watermarks, logos)
- **MUST add `-loop 1` before image input** for continuous animation
- **MUST add `fps=30` filter** to give image a framerate
- Example: `ffmpeg -i video.mp4 -loop 1 -i logo.png -filter_complex "[1:v]fps=30,rotate='2*PI*n/30'[rot];[0:v][rot]overlay"`

### overlay filter
- Position variables: `W`, `H` (main video), `w`, `h` (overlay), `t` (time)
- NOT supported: `duration`, `n`, `pos`, `pts`
- Requires two inputs: `[base][overlay]overlay`

### scale filter
- ONLY static expressions with `iw`, `ih`
- NOT supported: `t`, `n`, or frame variables
- For dynamic scaling use `zoompan`

### zoompan filter
- Best for smooth zoom on images AND videos
- Variables: `in` (input frame), `iw`, `ih`, `zoom`
- For video: use `d=1`
- NOT supported: `t` (use `in` instead)

### Pixel format handling
- YUV videos need `format=rgb24` or `format=gbrp` before RGB plane extraction
- mergeplanes requires identical dimensions on all inputs

## Common Solutions

### Frozen video frames
If rotation freezes with video:
1. Use frame-based rotation: `rotate='2*PI*n/30'` instead of `rotate='2*PI*t'`
2. Or pre-process video: `ffmpeg -i input.webm -vsync cfr -r 30 output.webm`

### Rotating watermark pattern
```
ffmpeg -i video.mp4 -loop 1 -i watermark.png -filter_complex "[1:v]fps=30,scale=100:100,rotate='2*PI*n/30':c=none[rot];[0:v][rot]overlay=W-w-10:H-h-10:shortest=1[vout]" -map "[vout]" -c:v libx264 output.mp4
```

### Chromatic aberration pattern
```
ffmpeg -i input.mp4 -filter_complex "[0:v]format=gbrp[rgb];[rgb]split=3[r][g][b];[r]extractplanes=r[rp];[g]extractplanes=g[gp];[b]extractplanes=b[bp];[rp]scale=iw+4:ih+4,crop=iw-4:ih-4:2:2[rs];[bp]scale=iw+4:ih+4,crop=iw-4:ih-4:0:0[bs];[rs][gp][bs]mergeplanes=0x001020:gbrp[vout]" -map "[vout]" output.mp4
```

# Process
1. Parse user requirements
2. Check for animation needs (use frame-based for reliability)
3. Ensure filter connectivity
4. Validate all variables are supported
5. Output command only

OUTPUT ONLY THE FFMPEG COMMAND - NO EXPLANATIONS
""",
)


@main_agent.tool
async def apply_video_edit(
    ctx: RunContext,
    current_command: str,
    request: str,
    video_path: str,
    fps: float = 30.01,
    video_width: int = 270,
    video_height: int = 478,
) -> str:
    """Apply effects to the current FFmpeg command"""
    logger.info(
        "Processing overlay effect request",
        request=request,
        current_command=current_command,
        video_path=video_path,
        fps=fps,
        video_width=video_width,
        video_height=video_height,
    )

    result = await efects_agent.run(
        [
            f"Video path: {video_path}",
            f"Current command: {current_command}",
            f"Effect request: {request}",
            f"FPS: {fps}",
            f"Video width: {video_width}",
            f"Video height: {video_height}",
        ]
    )

    logger.info("Overlay effect result generated", result=result.output)
    return result.output


@efects_agent.output_validator
async def validate_ffmpeg_command(ctx: RunContext, output: str) -> str:
    """Validate the final FFmpeg command using the common validator"""
    is_valid, error_message, cleaned_command = validate_ffmpeg_filter_complex(
        output, timeout=10
    )

    if not is_valid:
        raise ModelRetry(error_message)

    return cleaned_command
