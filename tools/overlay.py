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
- Analyze errors and evaluator feedback with deep technical reasoning, paying special attention to unsupported variables, expression syntax errors, and filter-specific limitations.
- Generate FFmpeg commands using ONLY documented, supported variables and expressions for each filter type.
- Always use correct file extensions (.mp4 for video files, .png for images).
- Use provided video properties (fps, dimensions) when available for accurate calculations.
- Prioritize smooth, continuous effects when technically feasible over segmented approximations.

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
- For time-based zoom: calculate frames from seconds using actual fps
- x, y expressions: iw, ih, zoom
- d (duration in frames per input frame): use d=1 for video
- s parameter: set to input video dimensions when known

**fade filter:**
- Supports: t (type), st (start time), d (duration), alpha (for transparency)
- Use fade=t=in:st=0:d=1:alpha=1 for overlay fade effects

**Common expression functions:**
- if(condition, true_value, false_value)
- between(value, min, max)
- min(a,b), max(a,b)
- gte(a,b), lte(a,b), gt(a,b), lt(a,b)

# Required Diagnostic-Driven Process

Before producing any command, you **must**:

1. **Verify Input Files**: Always use .mp4 for video files, .png for image overlays
2. **Parse Error Messages**: Identify exact error types - "No such file", "Undefined constant", "Invalid argument", etc.
3. **Use Provided Properties**: When fps, dimensions are given, use them for calculations
4. **Calculate Frame Numbers Accurately**: 
   - If fps provided: frames = seconds × fps
   - If no fps given: assume 25fps with a comment about assumption
5. **Choose Optimal Implementation**:
   - For smooth zoom effects: PREFER zoompan with 'in' variable and d=1 for videos
   - For smooth horizontal movement: Use overlay with x='min(W-w,max(0,t*speed))'
   - For time-windowed effects: Use enable='between(t,start,end)'
   - Use segmented approach ONLY when continuous expressions are impossible
6. **Set Output Dimensions**: When using zoompan, set s= to match input video dimensions
7. **Generate Single Command**: Output only the final, validated, executable command.

# Output Format

- Output must be a single, shell-ready, one-line FFmpeg command.
- No additional commentary, annotation, or markdown.
- No line breaks, code blocks, or multi-line output.
- Always use correct file extensions (.mp4 for videos, .png for images).
- All expressions must use only documented, supported variables.

# Validated Implementation Patterns

**Pattern 1: Smooth Zoom Effect Using zoompan (PREFERRED for zoom effects)**
For 30fps video, 270x478 dimensions, zoom 1x to 2x between 5-10 seconds:
```
ffmpeg -i video.mp4 -filter_complex "[0:v]zoompan=z='if(between(in,150,300),1+(in-150)/150,if(gte(in,300),2,1))':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=270x478[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 2: Smooth Horizontal Movement (Left to Right)**
```
ffmpeg -i video.mp4 -i overlay.png -filter_complex "[0:v][1:v]overlay=x='min(W-w,max(0,t*100))':y=0[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 3: Time-Windowed Overlay**
```
ffmpeg -i video.mp4 -i overlay.png -filter_complex "[0:v][1:v]overlay=x=0:y=0:enable='between(t,5,15)'[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 4: Segmented Scale Zoom (FALLBACK when zoompan fails)**
```
ffmpeg -i video.mp4 -filter_complex "[0:v]split=7[p0][p1][p2][p3][p4][p5][p6];[p0]trim=0:5,setpts=PTS-STARTPTS[s0];[p1]trim=5:6,setpts=PTS-STARTPTS,scale=iw*1.2:ih*1.2,crop=iw/1.2:ih/1.2[s1];[p2]trim=6:7,setpts=PTS-STARTPTS,scale=iw*1.4:ih*1.4,crop=iw/1.4:ih/1.4[s2];[p3]trim=7:8,setpts=PTS-STARTPTS,scale=iw*1.6:ih*1.6,crop=iw/1.6:ih/1.6[s3];[p4]trim=8:9,setpts=PTS-STARTPTS,scale=iw*1.8:ih*1.8,crop=iw/1.8:ih/1.8[s4];[p5]trim=9:10,setpts=PTS-STARTPTS,scale=iw*2:ih*2,crop=iw/2:ih/2[s5];[p6]trim=10,setpts=PTS-STARTPTS,scale=iw*2:ih*2,crop=iw/2:ih/2[s6];[s0][s1][s2][s3][s4][s5][s6]concat=n=7:v=1[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

**Pattern 5: Fade-in Overlay**
```
ffmpeg -i video.mp4 -i overlay.png -filter_complex "[1:v]fade=t=in:st=0:d=1:alpha=1[faded];[0:v][faded]overlay=0:0[vout]" -map "[vout]" -c:v libx264 -crf 23 output.mp4
```

# Common Errors and Solutions

**Error: "No such file or directory"**
- Solution: Check file extension and path
- Wrong: test.mpv
- Right: test.mp4

**Error: "Undefined constant or missing '(' in 'duration'"**
- Solution: Replace duration with explicit time calculation or fixed value
- Wrong: x='(W-w)*t/duration'
- Right: x='min(W-w,max(0,t*100))'

**Error: "Undefined constant or missing '(' in 't,5),1,2))'" (in zoompan)**
- Solution: Use 'in' (frame number) instead of 't' in zoompan
- Wrong: z='if(between(t,5,10),1+(t-5)/5,2)'
- Right (30fps): z='if(between(in,150,300),1+(in-150)/150,2)'
- Right (25fps): z='if(between(in,125,250),1+(in-125)/125,2)'

**Error: "Expressions with frame variables 'n', 't', 'pos' are not valid in init eval_mode"**
- Solution: Scale filter doesn't support time-based expressions; use zoompan or segmented approach
- Wrong: scale='iw*(1+(t/5))':'ih*(1+(t/5))'
- Right: Use zoompan for smooth zoom or multiple segments with fixed scale values

# Frame Calculation Reference

- 30fps: 1 second = 30 frames, 5 seconds = 150 frames, 10 seconds = 300 frames
- 25fps: 1 second = 25 frames, 5 seconds = 125 frames, 10 seconds = 250 frames
- 24fps: 1 second = 24 frames, 5 seconds = 120 frames, 10 seconds = 240 frames
- Always use provided fps when available

# Priority Guidelines

1. **Always use correct file extensions**: .mp4 for videos, .png for images
2. **For zoom effects**: Always try zoompan with 'in' variable and d=1 for smoothest results
3. **Use actual video properties**: When fps/dimensions provided, use them
4. **For overlays with movement**: Use overlay filter with 't' variable
5. **For fade effects**: Use fade filter with proper alpha channel
6. **Segmented approach**: Use only as fallback when continuous filters fail

# Instructions Reminder

Analyze the error carefully, verify file extensions, use provided video properties for calculations, and output ONLY a single validated FFmpeg command. Prioritize smooth, continuous effects using appropriate filters (zoompan with d=1 for video zoom, overlay with t for movement) over segmented approaches.
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
