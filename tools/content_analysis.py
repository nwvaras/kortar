import httpx
from pathlib import Path
from pydantic_ai import Agent, BinaryContent, RunContext
from main import main_agent
from common.logger import get_logger

logger = get_logger("kortar.tools.content_analysis")

@main_agent.tool
async def analyze_video(ctx: RunContext, video_path: str, query: str) -> str:
    """Analyze video based on a specific query, identifying relevant intervals and actionable insights.

    Common types of analysis you can ask:
    - Empty spaces or moments where nothing happens
    - Parts that can be cut
    - Segments that need improvements
    - Key moments/highlights
    - Abrupt transitions
    - Audio/video issues
    - Repetitive moments
    - Long pauses
    - Silent segments
    - Low activity periods
    - Detect elements in the video
    """
    logger.info("Starting video content analysis", video_path=video_path, query=query)


    gemini_agent = Agent(
        "google-gla:gemini-2.5-flash",
        system_prompt="""
        You are a professional video editor analyzing content based on specific queries. Your task is to identify time intervals in the video that are relevant to the user's query and provide actionable insights.

        For each relevant interval you identify, provide:
        - Exact start and end times (use MM:SS format for videos under 60 minutes, HH:MM:SS for longer)
        - Clear description of what happens in that segment and why it's relevant to the query
        - If the user ask for detect an object, provide an approximate position of the object in the frame.
        - Specific action suggestion: "trim", "enhance", "keep", "add_transition", "adjust_audio", "add_effect", or "observation"
        
        <examples>
        <example>
        User query: "Empty spaces or moments where nothing happens?"
        
        Answer:
        **Interval 1**
        *   **Time:** 00:00 - 00:05
        *   **Description:** Static title screen with no movement or action.
        *   **Action:** **observation**
        *   **Suggestion:** None.

        **Interval 2**
        *   **Time:** 00:15 - 00:25
        *   **Description:** Extended sequence of routine file operations with no significant events.
        *   **Action:** **observation**
        *   **Suggestion:** None.

        **Interval 3**
        *   **Time:** 00:35 - 00:45
        *   **Description:** Loading screen with spinning progress indicator.
        *   **Action:** **observation**
        *   **Suggestion:** None.

        **Interval 4**
        *   **Time:** 01:05 - 01:15
        *   **Description:** Repeated navigation through same menu items without selection.
        *   **Action:** **observation**
        *   **Suggestion:** None.
        </examples>
        <examples>

        User query: "Where is the ball?"
        
        Answer:
        **Interval 1**
        *   **Time:** 00:05 - 00:08
        *   **Description:** A red ball appears bouncing from the left edge and moves across the middle third of the frame, taking up about 1/3 of the screen width as it bounces.
        *   **Action:** **observation**
        *   **Suggestion:** This is the first appearance of the ball in the video.

        **Interval 2**
        *   **Time:** 00:08 - 00:12 
        *   **Description:** The ball continues bouncing but starts to slow down and rolls towards the right edge of the frame, eventually exiting completely.
        *   **Action:** **observation**
        *   **Suggestion:** The ball's first appearance ends here as it exits the scene.

        **Interval 3**
        *   **Time:** 00:15 - 00:18
        *   **Description:** The ball reappears from the top of the frame in the center area, dropping straight down while staying in the middle third of the screen width.
        *   **Action:** **observation**
        *   **Suggestion:** Second appearance of the ball with a different motion pattern.

        **Interval 4**
        *   **Time:** 00:18 - 00:20
        *   **Description:** The ball bounces once near the bottom of the frame and rolls diagonally towards the center-left area, disappearing behind some objects in the scene.
        *   **Action:** **observation**
        *   **Suggestion:** Final appearance of the ball before it's completely out of view.
        </example>
        </examples>
        Be precise with timing and practical with suggestions. Only include intervals that directly answer the user's query.
        """,
    )

    video_content = await load_video_as_binary(video_path)

    result = await gemini_agent.run([video_content, f"Query: {query}"])
    print(result.output)
    return result.output


async def load_video_as_binary(video_path: str) -> BinaryContent:
    """Load video file as binary content"""
    logger.info("Loading video for analysis", video_path=video_path)

    if video_path.startswith("http"):
        logger.info("Downloading video from URL")
        async with httpx.AsyncClient() as client:
            response = await client.get(video_path)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "video/mp4")
            return BinaryContent(data=response.content, media_type=content_type)
    else:
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with open(video_path_obj, "rb") as f:
            video_data = f.read()

        ext = video_path_obj.suffix.lower()
        media_type_map = {
            ".mp4": "video/mp4",
            ".avi": "video/avi",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".flv": "video/x-flv",
        }
        media_type = media_type_map.get(ext, "video/mp4")

        return BinaryContent(data=video_data, media_type=media_type)
