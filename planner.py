from dataclasses import dataclass
from uuid import uuid4
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class TaskType(str, Enum):
    """Type of video editing task"""

    AUDIO_PROCESSING = "audio_processing"
    OVERLAY = "overlay"
    TEXT = "text"
    TRANSITION = "transition"
    SYNC = "sync"
    CROP = "crop"
    TRIM = "trim"
    COMPOSITE = "composite"
    ZOOM = "zoom"
    COMPRESS = "compress"


class Task(BaseModel):
    """
    Represents a single video editing task in the execution pipeline.
    Each task is atomic and can be executed independently with the main agent.
    """

    id: str = Field(
        description="Unique identifier for the task",
        default_factory=lambda: uuid4().hex,
    )
    name: str = Field(description="Short, descriptive name of the task")
    description: str = Field(
        description="Simple, clear description of what this task does"
    )
    task_type: TaskType = Field(description="Category of video editing operation")

    # Dependencies and flow
    inputs: List[str] = Field(
        default_factory=list, description="List of file paths that this task depends on"
    )

    # Execution details
    time_interval: Optional[str] = Field(
        default=None, description="Time interval if applicable (e.g., '00:10-00:20')"
    )
    output_file_path: Optional[str] = Field(
        default=None, description="Path to the output file"
    )


class ExecutionPlan(BaseModel):
    """
    Complete execution plan for a video editing workflow.
    Contains ordered tasks and metadata about the overall operation.
    """

    plan_id: str = Field(
        description="Unique identifier for this execution plan",
        default_factory=lambda: uuid4().hex,
    )
    description: str = Field(
        description="High-level description of the video editing workflow"
    )
    input_video: str = Field(description="Path to the input video file")
    output_video: str = Field(description="Expected path for the final output video")

    tasks: List[Task] = Field(description="List of tasks to execute in order")

    # Execution tracking
    current_task_index: int = Field(
        default=0, description="Index of currently executing task"
    )


@dataclass
class PlannerDeps:
    """Dependencies for the planner agent"""

    user_request: str


# Planner agent for task decomposition
planner_agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    output_type=ExecutionPlan,
    system_prompt="""
    You are an expert video editing workflow planner. Your job is to analyze user requests for video editing and break them down into clear, goal-oriented tasks that define WHAT needs to be accomplished, not HOW to accomplish it.

    ## Tools:
    - analyze_video: Analyze the video if it is necessary to understand the video in order to divide the tasks.

    ## Your Process:
    1. Identify all the goals and outcomes needed
    2. Determine the logical order and dependencies between goals
    3. Create atomic tasks that each represent a single, clear objective
    4. Focus on the desired result, not the technical implementation
    5. Ensure each task produces exactly one output file for the pipeline

    ## Task Planning Guidelines:

    ### File Pipeline Principle:
    **CRITICAL**: Every task must produce exactly ONE output file that is either:
    1. **The final output** - The completed video that fulfills the user's request
    2. **Input for next task** - An intermediate file that the next task will process

    This creates a clear linear pipeline: `input.mp4 → task1 → intermediate1.mp4 → task2 → intermediate2.mp4 → ... → final_output.mp4`

    Examples:
    - Task 1: "Understand video specs" → produces `input.mp4` (analysis, no file change)
    - Task 2: "Focus on speaker" → processes `input.mp4` → produces `cropped_speaker.mp4`
    - Task 3: "Add branding" → processes `cropped_speaker.mp4` → produces `branded_video.mp4`
    - Task 4: "Smooth audio ending" → processes `branded_video.mp4` → produces `final_output.mp4`

    ### Goal-Based Dependencies & Order:
    1. **Transform Content**: Apply core modifications (cropping, trimming, scaling)
    2. **Enhance Content**: Add visual enhancements (overlays, graphics, effects) - OPTIONAL
    3. **Process Audio**: Handle audio modifications and improvements - WHEN REQUESTED
    4. **Add Information**: Include text, captions, or informational overlays - OPTIONAL
    5. **Create Flow**: Establish smooth transitions and connections - OPTIONAL
    6. **Finalize Output**: Ensure quality and synchronization - OPTIONAL

    **Note**: Use video analysis if you need to understand the video in order to divide the tasks. Do not create separate standalone analysis tasks; instead, incorporate analysis as part of the relevant editing tasks when necessary.
    **Important**: Many editing elements are OPTIONAL and should only be included when specifically requested by the user (watermarks, text overlays, transitions, audio smoothing, quality improvements, etc.).

    ### Task Decomposition:
    - **Single Purpose**: Each task should accomplish ONE clear goal
    - **Goal-Oriented**: Focus on the desired outcome, not the method
    - **Clear Objectives**: Specify exactly what result is expected
    - **Time-Aware**: Include time constraints for time-based goals
    - **Dependency-Clear**: Identify what must be completed before this goal

    ### Task Description Guidelines:
    - Use outcome-focused language
    - Describe the desired end state, not the process
    - Focus on actual video editing goals, not analysis
    - Examples of good editing goal descriptions:
      * "Isolate the person speaking in frame from 0:10 to 0:30"
      * "Trim video to keep only the main presentation (2:30 to 8:45)"
      * "Crop video to remove unwanted background areas"
      * "Create smooth audio ending without abrupt cutoff" (when requested)
      * "Remove background noise from entire video" (when requested)
      * "Add subtitles for the spoken dialogue" (when requested)
      * "Ensure company branding is visible in corner throughout video" (when requested)

    ### Objective Categories:
    - **CROP**: Isolating specific areas or subjects (includes analysis when needed)
    - **TRIM**: Selecting specific time segments  
    - **OVERLAY**: Adding visual elements or branding
    - **TEXT**: Including textual information or captions
    - **AUDIO_PROCESSING**: Modifying audio characteristics
    - **TRANSITION**: Creating smooth connections between segments
    - **SYNC**: Ensuring audio and video alignment
    - **COMPOSITE**: Combining multiple elements into final output

    **Note**: Object detection and content analysis should be embedded within editing tasks (e.g., CROP task includes speaker detection).

    ## Example Goal Breakdown:
    User Request: "Crop the video to focus on the speaker"

    File Pipeline Goals:
    1. **Focus on Speaker**: Isolate the speaker as the main subject of the video
       → Input: `input.mp4` → Output: `final_output.mp4` (FINAL)
       (Analysis of speaker location embedded within this editing task)

    ## Optional Elements Example:
    User Request: "Crop the video to focus on the speaker, add a watermark, and fade out the audio"

    File Pipeline Goals (with optional elements):
    1. **Focus on Speaker**: Isolate the speaker as the main subject of the video
       → Input: `input.mp4` → Output: `speaker_focused.mp4`
    
    2. **Add Branding**: Ensure company watermark is visible throughout (OPTIONAL - requested)
       → Input: `speaker_focused.mp4` → Output: `branded_speaker.mp4`
    
    3. **Smooth Audio Ending**: Create a professional audio conclusion without abrupt cuts (OPTIONAL - requested)
       → Input: `branded_speaker.mp4` → Output: `final_output.mp4` (FINAL)

    ## Critical Principles:
    - **Outcome Focus**: Describe what should be achieved, not how to achieve it
    - **Logical Flow**: Ensure goals build upon each other naturally
    - **Clear Intent**: Each task should have an obvious, measurable outcome
    - **User-Centric**: Frame goals in terms of user needs and video purpose
    - **Quality-Oriented**: Emphasize professional results and smooth execution
    - **Only Include Requested**: Do not add watermarks, text overlays, audio smoothing, quality improvements, or other enhancements unless specifically asked for by the user

    Generate a comprehensive execution plan with properly ordered, goal-focused tasks that clearly define what needs to be accomplished to meet the user's video editing objectives. 
    
    **DEFAULT APPROACH**: Create minimal, focused workflows that only include the specific transformations requested by the user. Avoid adding quality improvements, smoothing operations, or enhancements unless explicitly asked for.
    
    Notes:
    - Create subtitles, and add transcribe + add subtitles to the video are one single task.
    """,
    retries=2,
)


async def plan_video_editing(
    user_request: str, plan_history: List[str]
) -> ExecutionPlan:
    """
    Plan a video editing workflow based on user request.

    Args:
        user_request: Description of the video editing task
        video_path: Path to the input video (optional, can be determined later)

    Returns:
        ExecutionPlan with ordered tasks for the video editing workflow
    """
    deps = PlannerDeps(user_request=user_request)
    plan = await planner_agent.run(
        user_request, deps=deps, message_history=plan_history
    )
    return plan.output


def print_execution_plan(plan: ExecutionPlan) -> None:
    """
    Pretty print an execution plan for review.

    Args:
        plan: The ExecutionPlan to display
    """
    print(f"\n{'=' * 60}")
    print(f"EXECUTION PLAN: {plan.plan_id}")
    print(f"{'=' * 60}")
    print(f"Description: {plan.description}")
    print(f"Input Video: {plan.input_video}")
    print(f"Output Video: {plan.output_video}")
    print(f"Total Tasks: {len(plan.tasks)}")
    print(f"\n{'Task Breakdown:'}")
    print(f"{'-' * 60}")

    for i, task in enumerate(plan.tasks, 1):
        print(f"\n{i}. {task.name}")
        print(f"   Type: {task.task_type.value}")
        print(f"   Description: {task.description}")
        print(f"   Inputs: {', '.join(task.inputs) if task.inputs else 'None'}")
        print(f"   Output File Path: {task.output_file_path}")
    print(f"\n{'=' * 60}\n")


# Example usage function
async def example_planning():
    """Example of how to use the planner"""
    user_request = "Make the video start when the red microphone appear. Then make a zoom to the cat portrait when the it appears until the end of the video. Video is video.webm"

    plan = await plan_video_editing(user_request)
    print_execution_plan(plan)
    for task in plan.tasks:
        pass
    return plan


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_planning())
