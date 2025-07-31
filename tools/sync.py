from pydantic_ai import Agent, ModelRetry, RunContext
from main import main_agent
from common.logger import get_logger

logger = get_logger("kortar.tools.sync")

# Specialized sync agent
sync_agent = Agent(
    "openai:gpt-4o-mini",
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
    """,
)


@main_agent.tool
async def apply_sync_filter(ctx: RunContext, current_command: str, request: str) -> str:
    """Apply synchronization adjustments to the current FFmpeg command"""
    logger.info("Processing sync filter request", request=request, current_command=current_command)

    result = await sync_agent.run(
        [f"Current command: {current_command}", f"Sync request: {request}"]
    )

    logger.info("Sync filter result generated", result=result.output)
    return result.output


@sync_agent.output_validator
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
            timeout=10,  # 10 second timeout for validation
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error("FFmpeg command validation failed", error_message=error_msg)
            raise ModelRetry(
                f"FFmpeg command validation failed with error: {error_msg}"
            )

        # Basic syntax validation
        if "-filter_complex" not in output and len(output.filters_used) > 0:
            raise ModelRetry(
                "Command should use -filter_complex for the specified filters."
            )

        logger.info("Command validation successful", message="FFmpeg executed without errors")
        return output

    except subprocess.TimeoutExpired:
        logger.error("Command validation timed out")
        raise ModelRetry(
            "Command validation timed out - command may be too complex or have infinite loops"
        )
    except Exception as e:
        logger.error("Command validation failed", error=str(e))
        raise ModelRetry(f"Command validation error: {str(e)}") from e
