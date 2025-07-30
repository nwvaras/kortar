from pydantic_ai import Agent, RunContext
from main import main_agent


# Specialized text agent
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