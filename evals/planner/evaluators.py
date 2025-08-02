from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from typing import List
import json
from planner import ExecutionPlan, TaskType


class PipelineIntegrityEvaluator(BaseMetric):
    """Evaluator that checks if the execution plan has a coherent file pipeline"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            # Parse the execution plan from actual_output
            plan_dict = json.loads(test_case.actual_output)
            plan = ExecutionPlan(**plan_dict)
            score = 0.0
            failed_checks = []

            # Check 1: Plan has input and output video specified
            if plan.input_video and plan.output_video:
                score += 0.25
            else:
                failed_checks.append("Missing input/output video paths")

            # Check 2: Tasks form a coherent pipeline
            if self._check_pipeline_coherence(plan):
                score += 0.25
            else:
                failed_checks.append(
                    "Tasks do not form a coherent pipeline - outputs not properly connected to inputs"
                )

            # Check 3: All tasks have output file paths
            if all(task.output_file_path for task in plan.tasks):
                score += 0.25
            else:
                failed_checks.append("Some tasks missing output file paths")

            # Check 4: No circular dependencies or orphaned tasks
            if self._check_no_circular_deps(plan):
                score += 0.25
            else:
                failed_checks.append("Circular dependencies or orphaned tasks detected")

            self.score = score
            self.success = self.score >= self.threshold

            if not self.success:
                self.reason = f"Pipeline integrity issues detected (score: {score:.2f}). Issues found: {'; '.join(failed_checks)}"

            return self.score

        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            return 0.0

    def _check_pipeline_coherence(self, plan: ExecutionPlan) -> bool:
        """Check that tasks form a coherent input->output pipeline"""
        if not plan.tasks:
            return False

        # First task should use input video
        first_task = plan.tasks[0]
        if plan.input_video not in first_task.inputs:
            return False

        # Last task should produce final output
        last_task = plan.tasks[-1]
        if last_task.output_file_path != plan.output_video:
            return False

        # Each task's output should be input to next task (if not final)
        for i in range(len(plan.tasks) - 1):
            current_output = plan.tasks[i].output_file_path
            next_inputs = plan.tasks[i + 1].inputs
            if current_output not in next_inputs:
                return False

        return True

    def _check_no_circular_deps(self, plan: ExecutionPlan) -> bool:
        """Check for circular dependencies in task pipeline"""
        # For each task, check if it depends on outputs from tasks that come after it
        for i, task in enumerate(plan.tasks):
            # Get all output file paths from tasks that come after this one
            future_outputs = {
                plan.tasks[j].output_file_path for j in range(i + 1, len(plan.tasks))
            }

            # Check if current task depends on any future outputs (circular dependency)
            for input_file in task.inputs:
                if input_file in future_outputs:
                    return False

        return True

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if hasattr(self, "error") and self.error is not None:
            return False
        return self.success

    @property
    def __name__(self):
        return "Pipeline Integrity Evaluator"


class TaskTypeAccuracyEvaluator(BaseMetric):
    """Evaluator that checks if task types match the user request"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            # Parse the execution plan
            plan_dict = json.loads(test_case.actual_output)
            plan = ExecutionPlan(**plan_dict)

            expected_types = test_case.additional_metadata.get(
                "expected_task_types", []
            )
            expected_count = test_case.additional_metadata.get("expected_task_count")
            expected_count_range = test_case.additional_metadata.get(
                "expected_task_count_range"
            )

            score = 0.0

            # Check 1: Expected task types are present
            plan_task_types = [task.task_type.value for task in plan.tasks]
            if self._check_expected_types_present(plan_task_types, expected_types):
                score += 0.4

            # Check 2: Task count is appropriate
            if expected_count and len(plan.tasks) == expected_count:
                score += 0.3
            elif (
                expected_count_range
                and expected_count_range[0]
                <= len(plan.tasks)
                <= expected_count_range[1]
            ):
                score += 0.3
            elif not expected_count and not expected_count_range:
                score += 0.3  # No specific expectation

            # Check 3: No obviously wrong task types for the request
            if self._check_no_wrong_types(test_case.input, plan_task_types):
                score += 0.3

            self.score = score
            self.success = self.score >= self.threshold

            if not self.success:
                self.reason = f"Task type mismatch. Expected: {expected_types}, Got: {plan_task_types}"

            return self.score

        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            return 0.0

    def _check_expected_types_present(
        self, plan_types: List[str], expected_types: List[str]
    ) -> bool:
        """Check if all expected task types are present in the plan"""
        if not expected_types:
            return True
        plan_types_set = set(plan_types)
        expected_types_set = set(expected_types)
        return expected_types_set.issubset(plan_types_set)

    def _check_no_wrong_types(self, user_input: str, plan_types: List[str]) -> bool:
        """Basic heuristic check for obviously wrong task types"""
        input_lower = user_input.lower()

        # Simple keyword-based validation
        wrong_combinations = [
            (
                "audio" not in input_lower and "noise" not in input_lower,
                "audio_processing",
            ),
            (
                "crop" not in input_lower
                and "speaker" not in input_lower
                and "focus" not in input_lower,
                "crop",
            ),
            (
                "trim" not in input_lower
                and "seconds" not in input_lower
                and "time" not in input_lower,
                "trim",
            ),
        ]

        for condition, wrong_type in wrong_combinations:
            if condition and wrong_type in plan_types:
                return False

        return True

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if hasattr(self, "error") and self.error is not None:
            return False
        return self.success

    @property
    def __name__(self):
        return "Task Type Accuracy Evaluator"


class TaskQualityEvaluator(BaseMetric):
    """Evaluator that checks the quality of individual task definitions"""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            # Parse the execution plan
            plan_dict = json.loads(test_case.actual_output)
            plan = ExecutionPlan(**plan_dict)

            if not plan.tasks:
                self.score = 0.0
                self.success = False
                self.reason = "No tasks found in plan"
                return 0.0

            total_score = 0.0

            for task in plan.tasks:
                task_score = self._evaluate_task_quality(task)
                total_score += task_score

            average_score = total_score / len(plan.tasks)

            self.score = average_score
            self.success = self.score >= self.threshold

            if not self.success:
                self.reason = (
                    f"Task quality below threshold. Average score: {average_score:.2f}"
                )

            return self.score

        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            return 0.0

    def _evaluate_task_quality(self, task) -> float:
        """Evaluate the quality of a single task"""
        score = 0.0
        checks = 0

        # Check 1: Task has a clear, non-empty name
        if task.name and len(task.name.strip()) > 3:
            score += 0.2
        checks += 1

        # Check 2: Task has a descriptive description
        if task.description and len(task.description.strip()) > 10:
            score += 0.2
        checks += 1

        # Check 3: Task type is valid
        if hasattr(task, "task_type") and task.task_type in TaskType:
            score += 0.2
        checks += 1

        # Check 4: Task has proper inputs specified
        if task.inputs:
            score += 0.2
        checks += 1

        # Check 5: Task has output file path
        if task.output_file_path:
            score += 0.2
        checks += 1

        return score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if hasattr(self, "error") and self.error is not None:
            return False
        return self.success

    @property
    def __name__(self):
        return "Task Quality Evaluator"
