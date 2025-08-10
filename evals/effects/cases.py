from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()

# Test cases for overlay agent functionality

static_overlay_basic = Golden(
    input="Add a logo overlay at position 10,10, pinwi.png",
    additional_metadata={
        "name": "static_overlay_basic",
        "current_command": "ffmpeg -i test.mp4 -c:v libx264 -crf 23 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "easy",
        "effect_type": "static_overlay",
    },
)

timed_overlay = Golden(
    input="Add overlay that appears between 5 and 15 seconds, pinwi.png",
    additional_metadata={
        "name": "timed_overlay",
        "current_command": "ffmpeg -i test.mp4 -c:v libx264 -crf 23 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "medium",
        "effect_type": "timed_overlay",
    },
)

moving_overlay = Golden(
    input="Add overlay that moves horizontally from left to right, pinwi.png",
    additional_metadata={
        "name": "moving_overlay",
        "current_command": "ffmpeg -i test.mp4 -c:v libx264 -crf 23 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "medium",
        "effect_type": "moving_overlay",
    },
)

fade_overlay = Golden(
    input="Add overlay with fade-in effect, pinwi.png",
    additional_metadata={
        "name": "fade_overlay",
        "current_command": "ffmpeg -i test.mp4 -c:v libx264 -crf 23 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "medium",
        "effect_type": "fade_overlay",
    },
)

corner_positioned_overlay = Golden(
    input="Add overlay in bottom-right corner with 10px padding, use pinwi.png",
    additional_metadata={
        "name": "corner_positioned_overlay",
        "current_command": "ffmpeg -i test.mp4 -c:v libx264 -crf 23 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "easy",
        "effect_type": "positioned_overlay",
    },
)

zoom_effect = Golden(
    input="Add zoom effect that zooms in from 1x to 2x between 5 and 10 seconds",
    additional_metadata={
        "name": "zoom_effect",
        "current_command": "ffmpeg -i test.mp4 -c:v libx264 -crf 23 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "hard",
        "effect_type": "zoom",
    },
)

preserve_existing_filters = Golden(
    input="Add static overlay at top-left corner, pinwi.png",
    additional_metadata={
        "name": "preserve_existing_filters",
        "current_command": "ffmpeg -i test.mp4 -vf scale=1920:1080,fps=30 -c:v libx264 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "medium",
        "effect_type": "preserve_filters",
    },
)

chromatic_aberration = Golden(
    input="Add chromatic aberration effect",
    additional_metadata={
        "name": "chromatic_aberration",
        "current_command": "ffmpeg -i test.mp4 -vf scale=1920:1080,fps=30 -c:v libx264 output.mp4",
        "video_path": "test.mp4",
        "difficulty": "medium",
    },
)


# Create evaluation metric - single comprehensive LLM judge
llm_judge = GEval(
    name="FFmpeg Effects Evaluation",
    criteria="Evaluate if the FFmpeg command correctly implements the requested effect. Check for: 1) Correct overlay syntax, 2) Valid FFmpeg command structure, 3) Appropriate filter usage for the effect type",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# Create dataset with all test cases
dataset = EvaluationDataset(
    goldens=[
        static_overlay_basic,
        timed_overlay,
        moving_overlay,
        fade_overlay,
        corner_positioned_overlay,
        zoom_effect,
        preserve_existing_filters,
        chromatic_aberration,
    ]
)
