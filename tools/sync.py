from pydantic_ai import Agent, ModelRetry, RunContext
from main import FFmpegCommand, main_agent


# Specialized sync agent
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



@sync_agent.output_validator
async def validate_ffmpeg_command(ctx: RunContext, output: FFmpegCommand) -> FFmpegCommand:
    """Validate the final FFmpeg command"""
    import subprocess
    import re
    
    print(f"[LOG][Sync] Validating FFmpeg command: {output.command}")
    
    if not output.command.strip().lower().startswith('ffmpeg'):
        raise ModelRetry('The command must start with "ffmpeg". Please generate a valid FFmpeg command.')
    
    # Check for missing input file after -i flag
    if re.search(r'-i\s+(-f\s+null|$|\s+-)', output.command):
        raise ModelRetry('Missing input file after -i flag. Please specify a valid input file path.')
    
    # Add -y flag if not present
    if ' -y ' not in output.command and not output.command.startswith('ffmpeg -y'):
        output.command = output.command.replace('ffmpeg ', 'ffmpeg -y ', 1)
    
    try:
        # Create test command with null output to validate syntax and execution
        test_command = output.command
        
        # Replace output file with null output for testing
        test_command = re.sub(r'\s+output\.[a-zA-Z0-9]+(?:\s|$)', ' -f null /dev/null ', test_command)
        test_command = re.sub(r'\s+[a-zA-Z0-9_.-]+\.(mp4|avi|mov|mkv)(?:\s|$)', ' -f null /dev/null ', test_command)
        
        # Ensure we have null output if no output file was found
        if '-f null' not in test_command:
            test_command += ' -f null /dev/null'
        
        print(f"[LOG] Testing command with null flag: {test_command}")
        
        # Execute the test command to validate it works
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout for validation
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            print(f"[ERROR] FFmpeg command failed: {error_msg}")
            raise ModelRetry(f'FFmpeg command validation failed with error: {error_msg}')
        
        # Basic syntax validation
        if '-filter_complex' not in output.command and len(output.filters_used) > 0:
            raise ModelRetry('Command should use -filter_complex for the specified filters.')
        
        print("[LOG] Command validation successful - FFmpeg executed without errors")
        return output
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command validation timed out")
        raise ModelRetry('Command validation timed out - command may be too complex or have infinite loops')
    except Exception as e:
        print(f"[ERROR] Command validation failed: {str(e)}")
        raise ModelRetry(f'Command validation error: {str(e)}') from e