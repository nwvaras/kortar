from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()

# Test cases for planner agent functionality

simple_crop = Golden(
    input="Crop the video to focus on the speaker, test.mp4",
    additional_metadata={
        "name": "simple_crop",
        "video_path": "test_video.mp4",
        "difficulty": "easy",
        "expected_task_types": ["crop"],
        "expected_task_count": 1,
        "request_type": "single_task",
    },
)

simple_trim = Golden(
    input="Trim the video to keep only the first 30 seconds, test.mp4",
    additional_metadata={
        "name": "simple_trim",
        "video_path": "test_video.mp4",
        "difficulty": "easy",
        "expected_task_types": ["trim"],
        "expected_task_count": 1,
        "request_type": "single_task",
    },
)

crop_and_overlay = Golden(
    input="Crop the video to focus on the speaker and add a watermark in the corner, test.mp4",
    additional_metadata={
        "name": "crop_and_overlay",
        "video_path": "test_video.mp4",
        "difficulty": "medium",
        "expected_task_types": ["crop", "overlay"],
        "expected_task_count": 2,
        "request_type": "multi_task",
    },
)

complex_editing = Golden(
    input="Trim the video to keep only the main presentation (2:30 to 8:45), crop to remove background, and add subtitles, test.mp4",
    additional_metadata={
        "name": "complex_editing",
        "video_path": "presentation.mp4",
        "difficulty": "hard",
        "expected_task_types": ["trim", "crop", "text"],
        "expected_task_count": 3,
        "request_type": "multi_task",
    },
)

audio_processing = Golden(
    input="Remove background noise and add a fade-out effect to the audio, test.mp4",
    additional_metadata={
        "name": "audio_processing",
        "video_path": "noisy_video.mp4",
        "difficulty": "medium",
        "expected_task_types": ["audio_processing"],
        "expected_task_count": 1,
        "request_type": "audio_focused",
    },
)

time_based_effects = Golden(
    input="Add a zoom effect from 1x to 2x between 10 and 20 seconds, then add a transition to fade out, test.mp4",
    additional_metadata={
        "name": "time_based_effects",
        "video_path": "test_video.mp4",
        "difficulty": "hard",
        "expected_task_types": ["zoom", "transition"],
        "expected_task_count": 2,
        "request_type": "time_specific",
    },
)

comprehensive_workflow = Golden(
    input="Start the video when the red microphone appears, zoom to the cat portrait when it appears until the end, and compress the final output, test.mp4",
    additional_metadata={
        "name": "comprehensive_workflow",
        "video_path": "complex_video.webm",
        "difficulty": "very_hard",
        "expected_task_types": ["trim", "zoom", "compress"],
        "expected_task_count": 3,
        "request_type": "complex_analysis_required",
    },
)

ambiguous_request = Golden(
    input="Make the video look better and more professional, test.mp4",
    additional_metadata={
        "name": "ambiguous_request",
        "video_path": "raw_video.mp4",
        "difficulty": "hard",
        "expected_task_types": ["crop", "audio_processing", "composite"],  # flexible
        "expected_task_count_range": [1, 4],
        "request_type": "ambiguous",
    },
)

minimal_request = Golden(
    input="Compress the video to reduce file size, test.mp4",
    additional_metadata={
        "name": "minimal_request",
        "video_path": "large_video.mp4",
        "difficulty": "easy",
        "expected_task_types": ["compress"],
        "expected_task_count": 1,
        "request_type": "single_task",
    },
)

# Create evaluation metrics

# LLM judge for plan quality
plan_quality_judge = GEval(
    name="Plan Quality Evaluation",
    criteria="""Evaluate the execution plan quality based on:
    1) Task types match the user request appropriately
    2) Tasks are in logical order with proper dependencies
    3) File pipeline is coherent (proper input/output chain)
    4) Task descriptions are clear and goal-oriented
    5) No unnecessary tasks are added beyond user request
    6) All required functionality is covered""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# LLM judge for task descriptions
task_clarity_judge = GEval(
    name="Task Clarity Evaluation",
    criteria="""Evaluate task descriptions for:
    1) Clear, outcome-focused language
    2) Specific objectives rather than vague goals
    3) Proper time intervals when needed
    4) Appropriate file paths and dependencies
    5) Professional and actionable descriptions""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

# Create dataset with all test cases
dataset = EvaluationDataset(
    goldens=[
        simple_crop,
        simple_trim,
        crop_and_overlay,
        complex_editing,
        audio_processing,
        time_based_effects,
        comprehensive_workflow,
        ambiguous_request,
        minimal_request,
    ]
)
