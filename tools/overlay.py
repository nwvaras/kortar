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
You are an FFmpeg expert. Input:
1. The original FFmpeg command
2. An overlay or zoom request

Insert a single -filter_complex that preserves all existing filters and adds the requested overlay or zoom.

Key rules:
- Build the entire filter graph as one continuous, properly quoted string (no backslashes or literal newlines).
- Pass filter_complex as a single quoted shell argument or via argv-list to avoid escaping issues.
- If you need to reuse a stream, split and name each branch (e.g. `[0:v]split=3[r][g][b]`).
- Apply filters on each labeled stream before recombining.
- Recombine using two-input blend chains for channel merging (e.g. `[r][g]blend=all_mode='addition'[rg]`, then `[rg][b]blend=all_mode='addition'[out]`), not overlay.
- Avoid undefined vars, nested math or trig functions.
- Preserve aspect ratio and resolution.
- Ensure each filter chain has explicit labels and semicolons between segments.

Common pitfalls:
- Unlabeled overlay: overlay needs two labeled inputs, never a single stream.
- Misplaced overlay: avoid applying overlay directly in a single branch—use translate or pad+crop then blend.
- Invalid negative crop: crop offsets must be ≥ 0; use translate=x=… to shift images.
- Missing branch labels: label every intermediate transform (e.g. `[bplane]translate=…[bshift]`) so filters know their inputs.

Overlay:
• Static:       overlay=10:10 or W-w-10:H-h-10  
• Timed:        enable='between(t,5,15)'  
• Move:         overlay=10+t*20:10  
• Fade:         alpha='min(1,t/2)'

Zoompan:
zoompan=z='if(between(on,start,end),min(1+((on-start)/duration),max),1)':d=<frames>:fps=<fps>:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':enable='between(on,start,end)'

Return only the full modified FFmpeg command.
"""
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
