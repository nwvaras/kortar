"""Common FFmpeg validation utilities for reuse across the codebase"""

import os
import subprocess
import shlex
import re
from typing import Tuple, Optional
from common.logger import get_logger

logger = get_logger("kortar.common.validators")


def prepare_ffmpeg_test_command(command: str) -> str:
    """
    Prepare an FFmpeg command for testing by replacing output with null output.

    Args:
        command: The original FFmpeg command

    Returns:
        The modified command with null output for testing
    """
    test_command = command

    # Add -y flag if not present
    if " -y " not in test_command and not test_command.startswith("ffmpeg -y"):
        test_command = test_command.replace("ffmpeg ", "ffmpeg -y ", 1)

    # Add flags to clean up output: hide banner and only show errors
    if " -hide_banner" not in test_command:
        test_command = test_command.replace("ffmpeg ", "ffmpeg -hide_banner ", 1)
    if " -loglevel" not in test_command:
        test_command = test_command.replace("ffmpeg ", "ffmpeg -loglevel error ", 1)

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

    return test_command


def validate_ffmpeg_filter_complex(
    command: str, timeout: int = 10
) -> Tuple[bool, Optional[str], str]:
    """
    Validate an FFmpeg command by executing it with null output.

    Args:
        command: The FFmpeg command to validate
        timeout: Timeout in seconds for the validation

    Returns:
        Tuple of (is_valid, error_message, cleaned_command)
        - is_valid: True if command is valid
        - error_message: Error message if validation failed, None if successful
        - cleaned_command: The cleaned command (with -y flag added if needed)
    """
    logger.info("Validating FFmpeg command", command=command)

    # Basic validation first
    if not command.strip().lower().startswith("ffmpeg"):
        return False, 'The command must start with "ffmpeg"', command

    # Check for missing input file after -i flag
    if re.search(r"-i\s+(-f\s+null|$|\s+-)", command):
        return (
            False,
            "Missing input file after -i flag. Please specify a valid input file path.",
            command,
        )

    # Add -y flag if not present
    cleaned_command = command
    if " -y " not in cleaned_command and not cleaned_command.startswith("ffmpeg -y"):
        cleaned_command = cleaned_command.replace("ffmpeg ", "ffmpeg -y ", 1)

    try:
        # Prepare test command with null output
        test_command = prepare_ffmpeg_test_command(cleaned_command)

        logger.debug("Testing command with null output", test_command=test_command)

        # Execute the test command to validate it works
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error(
                "Validator: FFmpeg command validation failed",
                pwd=os.getcwd(),
                error_message=error_msg,
            )
            return (
                False,
                f"FFmpeg command validation failed with error: {error_msg}",
                cleaned_command,
            )

        # Basic syntax validation for filter_complex
        if "-filter_complex" not in cleaned_command and (
            "overlay=" in cleaned_command or "zoompan=" in cleaned_command
        ):
            return (
                False,
                "Command should use -filter_complex for the specified filters.",
                cleaned_command,
            )

        logger.info(
            "Command validation successful", message="FFmpeg executed without errors"
        )
        return True, None, cleaned_command

    except subprocess.TimeoutExpired:
        logger.error("Command validation timed out")
        return (
            True,
            f"Command validation timed out after {timeout} seconds",
            cleaned_command,
        )
    except Exception as e:
        logger.error("Command validation failed", error=str(e), pwd=os.getcwd())
        return False, f"Command validation error: {str(e)}", cleaned_command
