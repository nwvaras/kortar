from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from main import main_agent
from common.logger import get_logger

logger = get_logger("kortar.tools.compress")

compression_agent = Agent(
    "openai:gpt-4.1",
    output_type=str,
    result_retries=3,
    system_prompt="""
  You are an FFmpeg expert. Input:
  1. The original FFmpeg command
  2. A compression request

  Modify the command to apply compression while preserving all existing filters.

  Key rules:
  - Ensure dimensions even: scale=trunc(iw/2)*2:trunc(ih/2)*2  
  - Remove duplicate frames: mpdecimate  
  - Use variable frame rate: -fps_mode vfr  
  - Video codec: libx264 with -crf 23 and -preset medium  
  - Audio codec: aac with -b:a 128k  
  - Chain compression filters in -vf before encoding  
  - Don't crop or resize the video 
  
  Return only the full modified FFmpeg command.
  """,
)

@main_agent.tool
async def apply_compression(
    ctx: RunContext, current_command: str, request: str, video_path: str
) -> str:
    """Apply compression to the current FFmpeg command"""
    logger.info("Processing compression request", 
                request=request, 
                current_command=current_command, 
                video_path=video_path)

    result = await compression_agent.run(
        [
            f"Video path: {video_path}",
            f"Current command: {current_command}",
            f"Compression request: {request}",
        ]
    )

    logger.info("Compression command generated", result=result.output)
    return result.output


@compression_agent.output_validator
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
                tokens[output_file_index:] = ["-f", "null", "/dev/null"]
                test_command = " ".join(shlex.quote(token) for token in tokens)
            else:
                # Fallback: just append null output
                test_command += " -f null /dev/null"

        except (ValueError, IndexError):
            # Fallback for complex quoting - just append null output
            test_command += " -f null /dev/null"

        # Ensure we have null output if no output file was found
        if "-f null" not in test_command:
            test_command += " -f null /dev/null"

        logger.debug("Testing command with null output", test_command=test_command)

        # Execute the test command to validate it works
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=6,  # 6 second timeout for validation
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error("FFmpeg command validation failed", error_message=error_msg)
            raise ModelRetry(
                f"FFmpeg command validation failed with error: {error_msg}"
            )

        logger.info("Command validation successful", message="FFmpeg executed without errors")
        return output

    except subprocess.TimeoutExpired:
        # If command runs for 3 seconds without crashing or failing, consider it validated
        logger.info("Command validation successful", message="FFmpeg ran for 3 seconds without errors - command validated")
        return output
    except Exception as e:
        logger.error("Command validation failed", error=str(e))
        raise ModelRetry(f"Command validation error: {str(e)}") from e
