from pydantic_ai import Agent, RunContext
from main import main_agent
from common.logger import get_logger

logger = get_logger("kortar.tools.transition")

# Specialized transition agent
transition_agent = Agent(
    "anthropic:claude-3-5-haiku-20241022",
    output_type=str,
    system_prompt="""
You are a video transition specialist. You modify FFmpeg commands to add transitions, concatenations, and video arrangements.

You will receive:
1. Current FFmpeg command
2. Specific transition request

Your job is to modify the command to add the requested transitions using -filter_complex.

CRITICAL VIDEO TIMING RULES:
1. When trimming segments for concatenation, ALWAYS use setpts=PTS-STARTPTS to reset timestamps:
   - CORRECT: [0:v]trim=start=1:end=11,setpts=PTS-STARTPTS[v0]
   - WRONG: [0:v]trim=start=1:end=11[v0]

2. For smooth playback and to avoid frozen frames, ALWAYS include these output options:
   - Add: -vsync cfr -r 30 (or appropriate framerate)
   - This ensures constant frame rate output, preventing freezes in transitions
   - Example: ffmpeg -i input.mp4 -filter_complex "..." -vsync cfr -r 30 -c:v libx264 output.mp4

3. When working with multiple inputs or complex filters that might have timing issues:
   - Consider adding fps filter: [0:v]fps=30[v0] to normalize framerate
   - Use shortest=1 in overlay filters to prevent hanging

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
5. Use -vsync cfr for constant frame rate to prevent frozen frames during transitions

Frame Rate Consistency:
- When concatenating videos with different frame rates, normalize them first
- Add fps=30 (or target fps) filter before concatenation
- Always specify output frame rate with -r flag
- Use -vsync cfr to force constant frame rate output

Always preserve existing filters and properly chain them. Return only the modified FFmpeg command.
If you have different width/height inputs, keep the same aspect ratio. Even if this means adding more black bars.

IMPORTANT: For any output video, include -vsync cfr -r [framerate] to ensure smooth playback without frozen frames.
    """,
)


@main_agent.tool
async def apply_transition_filter(
    ctx: RunContext, current_command: str, request: str
) -> str:
    """Apply transition effects to the current FFmpeg command. The request should say if the video has or doesn't have audio
    Send the aspect ratio of the parts of the video, if you want to concatenate them."""
    logger.info(
        "Processing transition filter request",
        request=request,
        current_command=current_command,
    )

    result = await transition_agent.run(
        [f"Current command: {current_command}", f"Transition request: {request}"]
    )

    logger.info("Transition filter result generated", result=result.output)
    return result.output
