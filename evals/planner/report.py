from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
import asyncio
import json
from unittest.mock import patch

from evals.planner.cases import dataset, plan_quality_judge, task_clarity_judge
from evals.planner.evaluators import (
    PipelineIntegrityEvaluator,
    TaskTypeAccuracyEvaluator,
    TaskQualityEvaluator,
)
from planner import plan_video_editing

load_dotenv()


async def mock_analyze_video_plan(ctx, video_path: str, query: str) -> str:
    """Mock video analysis function that returns reasonable video analysis data without calling LLM"""
    # Return a generic but realistic video analysis response
    print(f"Mocking analyze_video_plan for {video_path} with query {query}")
    mock_response = """
**Interval 1**
*   **Time:** 00:00 - 00:15
*   **Description:** Initial content before red microphone appears.
*   **Action:** **trim**
*   **Suggestion:** This segment should be removed as it's before the red microphone.

**Interval 2**
*   **Time:** 00:15 - 00:30
*   **Description:** Red microphone appears in frame, marking the start point.
*   **Action:** **trim**
*   **Suggestion:** This is where the video should start.

**Interval 3**
*   **Time:** 00:30 - 00:45
*   **Description:** Cat portrait appears in upper right quadrant of frame.
*   **Action:** **zoom**
*   **Suggestion:** Begin zoom effect focused on cat portrait.

**Analysis Summary:**
Video analysis shows key elements for editing: red microphone appears at 00:15 marking start point, cat portrait appears at 00:30 requiring zoom until end. Content structure supports the requested trim, zoom, and final compression operations.
"""
    return mock_response.strip()


async def run_planner_agent(
    user_request: str, video_path: str = "test_video.mp4"
) -> str:
    """Run the planner agent and return the execution plan as JSON string"""
    try:
        # Mock the wrapped_analyze_video function to avoid calling external LLM
        with patch(
            "tools.content_analysis.wrapped_analyze_video",
            side_effect=mock_analyze_video_plan,
        ):
            # Run the planner with empty history
            plan_history = []

            plan = await plan_video_editing(user_request, plan_history)

        # Convert ExecutionPlan to JSON string for evaluation
        plan_dict = plan.model_dump()
        return json.dumps(plan_dict, indent=2)

    except Exception as e:
        print(f"Error running planner for request '{user_request}': {str(e)}")
        # Return a minimal error plan for evaluation
        error_plan = {
            "plan_id": "error",
            "description": f"Failed to generate plan: {str(e)}",
            "input_video": video_path,
            "output_video": "error_output.mp4",
            "tasks": [],
            "current_task_index": 0,
        }
        return json.dumps(error_plan, indent=2)


def print_plan_summary(plan_json: str, test_name: str) -> None:
    """Print a summary of the generated plan for debugging"""
    try:
        plan_dict = json.loads(plan_json)
        tasks = plan_dict.get("tasks", [])
        print(f"\n--- Plan Summary for {test_name} ---")
        print(f"Description: {plan_dict.get('description', 'N/A')}")
        print(f"Tasks ({len(tasks)}):")
        for i, task in enumerate(tasks, 1):
            task_type = task.get("task_type", "unknown")
            task_name = task.get("name", "unnamed")
            print(f"  {i}. [{task_type}] {task_name}")
        print("---")
    except Exception as e:
        print(f"Error parsing plan for {test_name}: {e}")


async def main():
    """Main evaluation function using deepeval"""
    print(f"Starting planner evaluation with {len(dataset.goldens)} test cases...")

    # Convert goldens to test cases by running the planner agent
    test_cases = []
    for golden in dataset.goldens:
        test_name = golden.additional_metadata["name"]
        print(f"\nProcessing: {test_name}")
        print(f"Request: {golden.input}")

        # Get video path from metadata
        video_path = golden.additional_metadata.get("video_path", "test_video.mp4")

        # Run the planner agent
        actual_output = await run_planner_agent(golden.input, video_path)

        # Print plan summary for debugging
        print_plan_summary(actual_output, test_name)

        # Create test case
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            additional_metadata=golden.additional_metadata,
        )
        test_cases.append(test_case)

    # Setup evaluation metrics: LLM judges + custom evaluators
    pipeline_evaluator = PipelineIntegrityEvaluator(threshold=0.8)
    task_type_evaluator = TaskTypeAccuracyEvaluator(threshold=0.7)
    task_quality_evaluator = TaskQualityEvaluator(threshold=0.6)

    metrics = [
        plan_quality_judge,
        task_clarity_judge,
        pipeline_evaluator,
        task_type_evaluator,
        task_quality_evaluator,
    ]

    # Run evaluation
    print(f"\n{'=' * 60}")
    print(f"Running evaluation with {len(metrics)} metrics...")
    print(f"Metrics: {[metric.__name__ for metric in metrics]}")
    print(f"{'=' * 60}")

    try:
        evaluate(test_cases=test_cases, metrics=metrics)
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        raise


def run_single_test(test_name: str):
    """Run evaluation for a single test case by name"""

    async def single_test():
        golden = None
        for g in dataset.goldens:
            if g.additional_metadata["name"] == test_name:
                golden = g
                break

        if not golden:
            print(f"Test case '{test_name}' not found")
            return

        print(f"Running single test: {test_name}")
        print(f"Request: {golden.input}")

        video_path = golden.additional_metadata.get("video_path", "test_video.mp4")
        actual_output = await run_planner_agent(golden.input, video_path)

        print_plan_summary(actual_output, test_name)

        # Create and evaluate single test case
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            additional_metadata=golden.additional_metadata,
        )

        # Quick evaluation with just one metric
        pipeline_evaluator = PipelineIntegrityEvaluator(threshold=0.8)
        evaluate(test_cases=[test_case], metrics=[pipeline_evaluator])

    asyncio.run(single_test())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run single test if test name provided
        test_name = sys.argv[1]
        run_single_test(test_name)
    else:
        # Run full evaluation suite
        asyncio.run(main())
