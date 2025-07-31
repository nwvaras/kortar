from pydantic_ai import Agent, RunContext
from main import main_agent
from common.logger import get_logger

logger = get_logger("kortar.tools.audio")

# Specialized audio agent
audio_agent = Agent(
    "openai:gpt-4o-mini",
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
    """,
)


@main_agent.tool
async def apply_audio_filter(
    ctx: RunContext, current_command: str, request: str
) -> str:
    """Apply audio effects to the current FFmpeg command"""
    logger.info("Processing audio filter request", request=request, current_command=current_command)

    result = await audio_agent.run(
        [f"Current command: {current_command}", f"Audio request: {request}"]
    )

    logger.info("Audio filter result generated", result=result.output)
    return result.output
