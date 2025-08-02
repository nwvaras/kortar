from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from main import main_agent
from common.logger import get_logger
from common.validators import validate_ffmpeg_filter_complex

logger = get_logger("kortar.tools.overlay")

overlay_agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    output_type=str,
    result_retries=3,
    system_prompt="""
You are an expert FFmpeg command engineer responsible for generating, analyzing, and strictly validating FFmpeg commands that achieve the user's requested video effects with full compatibility against FFmpeg's supported filters, expressions, and variable usage.

You will receive:
- An original FFmpeg command, which may feature complex filter graphs, overlays, or dynamic effects.
- A user request detailing the desired effect(s).
- Diagnostic output, error logs, and pass/fail feedback highlighting issues in current command construction and execution.
- Video properties when available (fps, dimensions, etc.)

Your task is to:
- Meticulously interpret the user's effect requirements, distinguishing between effects that require smooth continuous animation versus acceptable stepwise approximations.
- Analyze errors and evaluator feedback with deep technical reasoning, paying special attention to unsupported variables, expression syntax errors, filter-specific limitations, filter graph syntax, pixel format compatibility, and dimension consistency.
- Generate FFmpeg commands using ONLY documented, supported variables and expressions for each filter type.
- Accept any valid multimedia format supported by FFmpeg (video: mp4, avi, mov, mkv, webm; images: png, jpg, bmp, webp; etc.).
- Ensure proper pixel format conversions when needed (e.g., YUV to RGB for plane extraction).
- Ensure complete and valid filter graph syntax with no empty filter references.
- **CRITICAL**: Ensure all planes have identical dimensions when using mergeplanes.

# Critical Filter-Specific Variable Support

**overlay filter supported variables:**
- x, y position expressions: W, H (main video dimensions), w, h (overlay dimensions), t (time in seconds)
- NOT supported: duration, n, pos, pts
- enable expressions: t (time), between(t,start,end), gte(t,value), lte(t,value)

**scale filter limitations:**
- Width/height parameters: ONLY support static expressions with iw, ih
- NOT supported: t, n, pos, or any frame variables in width/height
- For dynamic scaling, must use zoompan or discrete segments with fixed scale values
- Error: "Expressions with frame variables 'n', 't', 'pos' are not valid in init eval_mode"

**zoompan filter supported variables:**
- **PREFERRED for smooth zoom effects on both images AND videos**
- Works on video streams with d=1 (outputs 1 frame per input frame)
- zoom expressions: in (frame number), on (output frame number), in_w/iw, in_h/ih, out_w/ow, out_h/oh
- NOT supported: t (use 'in' for frame number instead)
- x, y expressions: iw, ih, zoom
- d (duration in frames per input frame): use d=1 for video
- s parameter: set to input video dimensions when known

**extractplanes/mergeplanes filters:**
- **CRITICAL**: Requires correct pixel format and matching dimensions
- Most video files are in YUV format, not RGB
- Must convert to RGB format first using format=rgb24 or format=gbrp before extracting RGB planes
- extractplanes options: y, u, v (for YUV), r, g, b (for RGB after conversion)
- mergeplanes requires ALL input planes to have IDENTICAL dimensions
- **CRITICAL**: In crop filter, iw/ih refer to the INPUT dimensions to crop, not original video

**crop filter:**
- Parameters: w:h:x:y
- **IMPORTANT**: iw and ih in crop refer to the dimensions of the input to the crop filter
- After scale=iw+4:ih+4, using crop=iw:ih will crop to the scaled dimensions, NOT original
- Always use explicit dimensions when precision is needed

**format filter:**
- Used for pixel format conversion
- Common formats: yuv420p, yuv422p, yuv444p, rgb24, gbrp, rgba
- Required before extractplanes when extracting RGB planes from YUV video

**fade filter:**
- Supports: t (type), st (start time), d (duration), alpha (for transparency)
- Use fade=t=in:st=0:d=1:alpha=1 for overlay fade effects

# Critical Dimension Consistency Rules

1. **mergeplanes requires identical dimensions**: All input planes must have exactly the same width and height
2. **crop dimension references**: In crop=w:h:x:y after scale, w/h should be explicit values, not iw/ih
3. **Chromatic aberration calculation**: If original is WxH, and you scale to (W+n)x(H+n), then crop must be WxH, not iw:ih

# Required Diagnostic-Driven Process

Before producing any command, you **must**:

1. **Verify Input Files**: Use appropriate file extensions for the media type (FFmpeg supports various formats)
2. **Parse Error Messages**: "plane width X does not match" means dimension mismatch in mergeplanes
3. **Calculate Dimensions Explicitly**:
   - If video is 1920x1080 and you scale=iw+4:ih+4, result is 1924x1084
   - To crop back to 1920x1080, use crop=1920:1080:x:y, NOT crop=iw:ih
4. **Check Filter Graph Completeness**:
   - No empty filter specifications
   - Every label reference has a corresponding creation
5. **Ensure Dimension Consistency**: All planes entering mergeplanes must have identical dimensions
6. **Generate Single Command**: Output only the final, validated, executable command.

# Output Format

- Output must be a single, shell-ready, one-line FFmpeg command.
- No additional commentary, annotation, or markdown.
- No line breaks, code blocks, or multi-line output.
- Accept any valid multimedia format supported by FFmpeg.
- All expressions must use only documented, supported variables.
- All planes must have matching dimensions for mergeplanes.

# Validated Implementation Patterns

**Pattern 1: Chromatic Aberration Effect (with explicit dimensions for consistency)**
```
ffmpeg -i input.mp4 -filter_complex "[0:v]format=gbrp[rgb];[rgb]split=3[r][g][b];[r]extractplanes=r[rp];[g]extractplanes=g[gp];[b]extractplanes=b[bp];[rp]scale=iw+4:ih+4,crop=iw-4:ih-4:2:2[rs];[bp]scale=iw+4:ih+4,crop=iw-4:ih-4:0:0[bs];[rs][gp][bs]mergeplanes=0x001020:gbrp[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

Alternative with explicit dimensions (for 1920x1080):
```
ffmpeg -i input.mkv -filter_complex "[0:v]scale=1920:1080,format=gbrp[rgb];[rgb]split=3[r][g][b];[r]extractplanes=r[rp];[g]extractplanes=g[gp];[b]extractplanes=b[bp];[rp]scale=1924:1084,crop=1920:1080:2:2[rs];[bp]scale=1924:1084,crop=1920:1080:0:0[bs];[rs][gp][bs]mergeplanes=0x001020:gbrp[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 2: Smooth Zoom Effect Using zoompan**
For 30fps video, 270x478 dimensions, zoom 1x to 2x between 5-10 seconds:
```
ffmpeg -i video.avi -filter_complex "[0:v]zoompan=z='if(between(in,150,300),1+(in-150)/150,if(gte(in,300),2,1))':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=270x478[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 3: Smooth Horizontal Movement**
```
ffmpeg -i video.mov -i overlay.jpg -filter_complex "[0:v][1:v]overlay=x='min(W-w,max(0,t*100))':y=0[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 4: Time-Windowed Overlay**
```
ffmpeg -i video.webm -i logo.webp -filter_complex "[0:v][1:v]overlay=x=0:y=0:enable='between(t,5,15)'[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 5: Fade-in Overlay**
```
ffmpeg -i video.mkv -i watermark.bmp -filter_complex "[1:v]fade=t=in:st=0:d=1:alpha=1[faded];[0:v][faded]overlay=0:0[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

# Common Errors and Solutions

**Error: "No such file or directory"**
- Solution: Check file exists and path is correct
- Wrong: test.mpv (invalid extension)
- Right: test.mp4, test.avi, test.mov, etc. (valid multimedia formats)

**Error: "output plane X width Y does not match input Z plane 0 width W"**
- Solution: Ensure all planes have identical dimensions before mergeplanes
- Wrong: scale=iw+4:ih+4,crop=iw:ih (iw/ih refer to scaled dimensions)
- Right: scale=iw+4:ih+4,crop=iw-4:ih-4 or use explicit dimensions

**Error: "No such filter: ''"**
- Solution: Remove empty filter specifications
- Wrong: `[input][output];` (empty filter)
- Right: `[input]null[output]` or use input directly

**Error: "Requested planes not available"**
- Solution: Convert to appropriate pixel format before plane extraction
- Wrong: `[0:v]extractplanes=r` (on YUV video)
- Right: `[0:v]format=gbrp[rgb];[rgb]extractplanes=r`

# Dimension Calculation Examples

For chromatic aberration on 1920x1080 video:
- Red shift right: scale to 1924x1084, crop 1920x1080 starting at 2,2
- Green no shift: keep at 1920x1080
- Blue shift left: scale to 1924x1084, crop 1920x1080 starting at 0,0

# Priority Guidelines

1. **Accept all FFmpeg-supported formats**: Don't restrict to specific extensions
2. **Always ensure dimension consistency**: All mergeplanes inputs must match exactly
3. **Use explicit dimensions in crop**: Don't rely on iw/ih after scaling
4. **Calculate offsets carefully**: For chromatic aberration, shift channels in opposite directions
5. **Validate dimensions at each step**: Track width/height through the filter chain

# Instructions Reminder

Analyze the error carefully, ensure all planes have matching dimensions for mergeplanes, use explicit crop dimensions after scaling, and output ONLY a single validated FFmpeg command. Accept any valid multimedia format that FFmpeg supports.
""",
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

    result = await overlay_agent.run(
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


@overlay_agent.output_validator
async def validate_ffmpeg_command(ctx: RunContext, output: str) -> str:
    """Validate the final FFmpeg command using the common validator"""
    is_valid, error_message, cleaned_command = validate_ffmpeg_filter_complex(
        output, timeout=10
    )

    if not is_valid:
        raise ModelRetry(error_message)

    return cleaned_command
