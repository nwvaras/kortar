from pydantic_ai import Agent, RunContext
from main import main_agent
from common.logger import get_logger

logger = get_logger("kortar.tools.transition")

# Specialized transition agent
transition_agent = Agent(
    "openai:gpt-4o-mini",
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
