from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evals.overlay.cases import dataset, llm_judge
from evals.overlay.evaluators import FFmpegExecutionEvaluator
from tools.overlay import overlay_agent

load_dotenv()


async def run_overlay_agent(
    query: str,
    current_command: str = "ffmpeg -i test.mp4 output.mp4",
    video_path: str = "test.mp4",
):
    """Run the overlay agent and return the generated FFmpeg command"""
    overlay_agent._output_validators = []
    # TODO: get fps, video width, video height from the actuall video
    fps = 30.01
    video_width = 270
    video_height = 478
    result = await overlay_agent.run(
        [
            f"Video path: {video_path}",
            f"Current command: {current_command}",
            f"Effect request: {query}",
            f"FPS: {fps}",
            f"Video width: {video_width}",
            f"Video height: {video_height}",
        ]
    )
    return result.output


import asyncio


async def main():
    """Main evaluation function using deepeval"""
    print(f"Starting evaluation with {len(dataset.goldens)} test cases...")

    # Convert goldens to test cases by running the overlay agent
    test_cases = []
    for golden in dataset.goldens:
        print(f"Processing: {golden.additional_metadata['name']}")

        # Get parameters from golden metadata
        current_command = golden.additional_metadata.get(
            "current_command", "ffmpeg -i test.mp4 output.mp4"
        )
        video_path = golden.additional_metadata.get("video_path", "test.mp4")

        # Run the overlay agent
        actual_output = await run_overlay_agent(
            golden.input, current_command, video_path
        )

        # Create test case
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            additional_metadata=golden.additional_metadata,
        )
        test_cases.append(test_case)

    # Setup evaluation metrics: LLM judge + execution evaluator
    execution_evaluator = FFmpegExecutionEvaluator(threshold=0.8)
    metrics = [llm_judge, execution_evaluator]

    # Run evaluation
    print(f"\nRunning evaluation with {len(metrics)} metrics...")
    evaluate(test_cases=test_cases, metrics=metrics)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
