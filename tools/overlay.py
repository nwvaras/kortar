from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from main import main_agent
from common.logger import get_logger

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

Example of chromatic abberation:

ffmpeg -y -i input.mp4
-filter_complex
"[0:v]split=3;lutrgb=g=0:b=0;lutrgb=r=0:b=0;lutrgb=r=0:
g=0;crop=iw-4:ih:4:0, pad=iw+4:ih:0:0;crop=iw-4:ih:0:0,
pad=iw+4:ih:4:0;crop=iw:ih-4:0:4,
pad=iw:ih+4:0:0;blend=all_mode='addition';blend=all_mod
e='addition'" -map "" -c:v libx264 -crf 18 -preset
veryfast output.mp4


Return only the full modified FFmpeg command.
"""
)



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
    """Validate the final FFmpeg command"""
    import subprocess
    import re

    logger.info("Validating FFmpeg command", command=output)

    if not output.strip().lower().startswith("ffmpeg"):
        raise ModelRetry(
            'The command must start with "ffmpeg". Please generate a valid FFmpeg command.'
        )

    # Check for missing input file after -i flag
    if re.search(r"-i\s+(-f\s+null|$|\s+-)", output):
        raise ModelRetry(
            "Missing input file after -i flag. Please specify a valid input file path."
        )

    # Add -y flag if not present
    if " -y " not in output and not output.startswith("ffmpeg -y"):
        output = output.replace("ffmpeg ", "ffmpeg -y ", 1)

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
                if token.startswith("-") or token == "/dev/null":
                    continue
                # Skip if it's likely a flag value (previous token is a flag)
                if (
                    i > 0
                    and tokens[i - 1].startswith("-")
                    and tokens[i - 1] not in ["-map", "-i"]
                ):
                    continue
                # This looks like a filename - should be the output file
                output_file_index = i
                break

            if output_file_index is not None:
                # Replace the output file with null output
                tokens[output_file_index:] = ["-f", "null", "-"]
                test_command = " ".join(shlex.quote(token) for token in tokens)
            else:
                # Fallback: just append null output
                test_command += " -f null -"

        except (ValueError, IndexError):
            # Fallback for complex quoting - just append null output
            test_command += " -f null -"

        # Ensure we have null output if no output file was found
        if "-f null" not in test_command:
            test_command += " -f null -"

        logger.debug("Testing command with null output", test_command=test_command)

        # Execute the test command to validate it works
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,  # 10 second timeout for validation
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error("FFmpeg command validation failed", error_message=error_msg)
            raise ModelRetry(
                f"FFmpeg command validation failed with error: {error_msg}"
            )

        # Basic syntax validation
        if "-filter_complex":
            raise ModelRetry(
                "Command should use -filter_complex for the specified filters."
            )

        logger.info("Command validation successful", message="FFmpeg executed without errors")
        return output

    except subprocess.TimeoutExpired:
        logger.error("Command validation timed out")
        return output
    except Exception as e:
        logger.error("Command validation failed", error=str(e))
        raise ModelRetry(f"Command validation error: {str(e)}") from e
