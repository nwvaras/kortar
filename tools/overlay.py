from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from main import main_agent


overlay_agent = Agent(
    'openai:gpt-4.1',
    output_type=str,
    result_retries=3,
    system_prompt="""
You are an FFmpeg expert. You will receive:
1. An FFmpeg command
2. An overlay or zoom request

Insert a single -filter_complex to apply overlays or zoompan, preserving all existing filters.

OVERLAY RULES
• Static:       overlay=10:10 or W-w-10:H-h-10  
• Timed:        enable='between(t,5,15)'  
• Move:         overlay=10+t*20:10  
• Fade:         alpha='min(1,t/2)'  
• Chain:        [logo]scale=…;[base][logo]overlay=…

ZOOMPAN RULES
• Syntax: zoompan=z='if(between(on,start,end),min(1+((on-start)/duration),max),1)':d=<frames>:fps=<fps>:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':enable='between(on,start,end)'  
• Must include fps=<input_fps>  
• Define start,end,duration,max explicitly  

AVOID trigonometry, nested math, undefined vars, invalid scale.

EXAMPLES
• Watermark:  [0:v][1:v]overlay=W-w-10:H-h-10[out]  
• PiP:        [1:v]scale=iw/4:ih/4[pip];[0:v][pip]overlay=10:10[out]  
• Zoom:       [v]zoompan=z='…':d=30:fps=30:enable='between(on,30,60)'[z]

Return only the full modified FFmpeg command.
"""
)




@main_agent.tool
async def apply_overlay_filter(ctx: RunContext, current_command: str, request: str, video_path: str) -> str:
    """Apply effects to the current FFmpeg command"""
    print(f"[LOG] Effect - Request: {request}")
    print(f"[LOG] Effect - Current command: {current_command}")
    
    result = await overlay_agent.run([
        f"Video path: {video_path}",
        f"Current command: {current_command}",
        f"Effect request: {request}"
    ])
    
    print(f"[LOG] Effect - Result: {result.output}")
    return result.output

@overlay_agent.output_validator
async def validate_ffmpeg_command(ctx: RunContext, output: str) -> str:
    """Validate the final FFmpeg command"""
    import subprocess
    import re
    
    print(f"[LOG][Overlay] Validating FFmpeg command: {output}")
    
    if not output.strip().lower().startswith('ffmpeg'):
        raise ModelRetry('The command must start with "ffmpeg". Please generate a valid FFmpeg command.')
    
    # Check for missing input file after -i flag
    if re.search(r'-i\s+(-f\s+null|$|\s+-)', output):
        raise ModelRetry('Missing input file after -i flag. Please specify a valid input file path.')
    
    # Add -y flag if not present
    if ' -y ' not in output and not output.startswith('ffmpeg -y'):
        output = output.replace('ffmpeg ', 'ffmpeg -y ', 1)
    
    try:
        # Create test command with null output to validate syntax and execution
        test_command = output
        
        # Replace output file with null output for testing
        # Parse FFmpeg command to find the actual output file (typically the last argument)
        import shlex
        try:
            # Split command into tokens while preserving quoted arguments
            tokens = shlex.split(test_command)
            
            # Find the last token that looks like a filename (not a flag, not /dev/null)
            output_file_index = None
            for i in range(len(tokens) - 1, -1, -1):
                token = tokens[i]
                # Skip flags and /dev/null
                if token.startswith('-') or token == '/dev/null':
                    continue
                # Skip if it's likely a flag value (previous token is a flag)
                if i > 0 and tokens[i-1].startswith('-') and not tokens[i-1] in ['-map', '-i']:
                    continue
                # This looks like a filename - should be the output file
                output_file_index = i
                break
            
            if output_file_index is not None:
                # Replace the output file with null output
                tokens[output_file_index:] = ['-f', 'null', '/dev/null']
                test_command = ' '.join(shlex.quote(token) for token in tokens)
            else:
                # Fallback: just append null output
                test_command += ' -f null /dev/null'
                
        except (ValueError, IndexError):
            # Fallback for complex quoting - just append null output
            test_command += ' -f null /dev/null'
        
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
        if '-filter_complex' not in output and len(output.filters_used) > 0:
            raise ModelRetry('Command should use -filter_complex for the specified filters.')
        
        print("[LOG] Command validation successful - FFmpeg executed without errors")
        return output
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command validation timed out")
        raise ModelRetry('Command validation timed out - command may be too complex or have infinite loops')
    except Exception as e:
        print(f"[ERROR] Command validation failed: {str(e)}")
        raise ModelRetry(f'Command validation error: {str(e)}') from e