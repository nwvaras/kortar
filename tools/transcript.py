import os
import subprocess
import tempfile
from pathlib import Path
from pydantic_ai import Agent, RunContext, ModelRetry
from main import main_agent
from deepgram import DeepgramClient, PrerecordedOptions
from deepgram_captions import DeepgramConverter, srt as deepgram_srt
import srt
from dotenv import load_dotenv
from common.logger import get_logger
from common.progress import add_task, update_task, remove_task

logger = get_logger("kortar.tools.transcript")

# Load environment variables
load_dotenv()

translate_agent = Agent(
    "anthropic:claude-3-5-haiku-20241022",
    output_type=str,
    system_prompt="""You are a language translator specialized in SRT subtitle files. 

Your task:
1. Receive SRT content and a target language code
2. Translate ONLY the text content to the target language
3. Keep the SRT format EXACTLY the same (numbering, timestamps, structure)
4. Preserve speaker labels like [speaker 0], [speaker 1], etc.
5. Do not modify subtitle numbers, timestamps, or formatting

SRT format requirements:
- Each subtitle block: number, timestamp line, text content, blank line
- Timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm
- Sequential numbering starting from 1
- Text content can span multiple lines within a subtitle block""",
)


@translate_agent.output_validator
async def validate_srt_format(ctx: RunContext, output: str) -> str:
    """Validate SRT format using the robust srt library"""
    logger.info("Validating SRT format")

    if not output or not output.strip():
        logger.warning("Translation validation failed: output is empty")
        raise ModelRetry("Output cannot be empty. Please provide a valid SRT content.")

    try:
        # Parse the SRT content - this validates format automatically and handles many edge cases
        subtitles = list(srt.parse(output))

        if not subtitles:
            logger.warning("Translation validation failed: no subtitles found")
            raise ModelRetry("No valid subtitles found in the output.")

        # Re-compose to ensure clean formatting and catch any structural issues
        validated_srt = srt.compose(subtitles)

        logger.info("SRT validation successful", subtitle_blocks=len(subtitles))
        return validated_srt

    except Exception as e:
        logger.error("SRT validation failed", error=str(e))
        raise ModelRetry(
            f"Invalid SRT format: {str(e)}. Please ensure proper SRT structure with sequential numbering, valid timestamps (HH:MM:SS,mmm --> HH:MM:SS,mmm), and text content for each subtitle block."
        )


@main_agent.tool
async def transcript_video(
    ctx: RunContext,
    video_path: str,
    output_srt_path: str = None,
    translate: bool = False,
    language: str = "en",
    extra_context: str = None,
) -> str:
    """Create an SRT subtitle file from a video using Deepgram transcription.
    If translate is True, the SRT file will be translated to the target language.
    If extra_context is provided, it will be added to the transcription. Use the extra context to define who is speaker 0, 1, 2, etc.
    """
    logger.info(
        "Starting video transcription",
        video_path=video_path,
        language=language,
        extra_context=extra_context,
    )

    try:
        # Initialize Deepgram client
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            return "Error: DEEPGRAM_API_KEY environment variable not set"

        deepgram = DeepgramClient(api_key)

        # Verify video file exists
        if not os.path.exists(video_path):
            return f"Error: Video file not found at {video_path}"

        # Generate output SRT path if not provided
        if output_srt_path is None:
            video_path_obj = Path(video_path)
            output_srt_path = str(video_path_obj.with_suffix(".srt"))

        logger.info("Transcript output path set", output_srt_path=output_srt_path)

        # Configure transcription options
        options = PrerecordedOptions(
            model="nova-3",
            detect_language=True,
            smart_format=True,
            punctuate=True,
            diarize=True,
        )

        # Extract audio from video using ffmpeg
        logger.info("Extracting audio from video")
        extraction_task = add_task("Extracting audio from video...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        # Use ffmpeg to extract audio as WAV
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM 16-bit
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            temp_audio_path,
        ]

        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            logger.info("Audio extraction completed successfully")
            update_task(extraction_task, description="Audio extraction complete")
        except subprocess.CalledProcessError as e:
            remove_task(extraction_task)
            os.unlink(temp_audio_path)  # Clean up temp file
            return f"Error extracting audio: {e.stderr}"

        # Transcribe the extracted audio file
        logger.info("Starting Deepgram transcription")
        transcription_task = add_task("Transcribing audio with Deepgram...")
        try:
            with open(temp_audio_path, "rb") as audio_file:
                buffer_data = audio_file.read()
                payload = {"buffer": buffer_data}

                response = deepgram.listen.rest.v("1").transcribe_file(
                    source=payload, options=options
                )

            logger.info("Deepgram transcription completed")
            update_task(transcription_task, description="Converting to SRT format...")

        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                logger.debug("Cleaned up temporary audio file")

        # Convert to SRT format using deepgram-captions
        transcription = DeepgramConverter(response)
        srt_content = deepgram_srt(transcription)
        remove_task(transcription_task)

        # Translate if requested
        if translate and language:
            logger.info("Starting SRT translation", target_language=language)
            translation_task = add_task(f"Translating to {language}...")
            try:
                translation_result = await translate_agent.run(
                    [
                        f"SRT content to translate:\n{srt_content}",
                        f"Target language: {language}",
                        f"Extra context: {extra_context}",
                    ]
                )
                srt_content = translation_result.output
                logger.info("SRT translation completed successfully")
                remove_task(translation_task)
            except Exception as e:
                logger.error("SRT translation failed", error=str(e))
                remove_task(translation_task)
                return f"Transcription successful but translation failed: {str(e)}"

        # Save SRT file
        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        logger.info("SRT file saved", output_path=output_srt_path)

        translation_note = (
            f" (translated to {language})" if translate and language else ""
        )
        return f"Successfully created SRT subtitle file at: {output_srt_path}{translation_note}"

    except Exception as e:
        error_msg = f"Error creating transcript: {str(e)}"
        logger.error("Transcription process failed", error=error_msg)
        return error_msg
