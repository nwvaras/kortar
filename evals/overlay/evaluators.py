from common.validators import  validate_ffmpeg_filter_complex

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class FFmpegExecutionEvaluator(BaseMetric):
    """Evaluator that actually executes FFmpeg commands to test if they run for at least 2 seconds"""
    
    def __init__(self, threshold: float = 0.8, min_runtime_seconds: float = 2.0):
        self.threshold = threshold
        self.min_runtime_seconds = min_runtime_seconds
    
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            if not isinstance(test_case.actual_output, str):
                self.score = 0.0
                self.success = False
                return 0.0
            
            command = test_case.actual_output.strip()
            
            # Use the shared validator function to test execution with 2-second minimum runtime
            is_valid, error_message, _ = validate_ffmpeg_filter_complex(command, timeout=self.min_runtime_seconds)
            
            self.score = 1.0 if is_valid else 0.0
            self.success = self.score >= self.threshold
            
            if error_message:
                self.reason = error_message
            
            return self.score
        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            raise
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        if hasattr(self, 'error') and self.error is not None:
            return False
        return self.success
    
    @property
    def __name__(self):
        return "FFmpeg Execution Evaluator"
