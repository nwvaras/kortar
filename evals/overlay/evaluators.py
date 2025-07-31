from dataclasses import dataclass
import re
from typing import List

from .cases import dataset

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance

# Add basic string type validation
dataset.add_evaluator(IsInstance(type_name='str'))


@dataclass
class FFmpegOverlayEvaluator(Evaluator):
    """Evaluator for FFmpeg overlay commands that checks for required patterns"""
    
    async def evaluate(self, ctx: EvaluatorContext[dict, str]) -> float:
        if not isinstance(ctx.output, str):
            return 0.0
        
        output = ctx.output.strip()
        
        # Get expected patterns from test case
        expected_patterns = getattr(ctx.case, 'expected_output_contains', [])
        if not expected_patterns:
            # Fallback to basic FFmpeg validation if no patterns specified
            return 1.0 if output.lower().startswith('ffmpeg') else 0.0
        
        # Check each required pattern
        matched_patterns = 0
        total_patterns = len(expected_patterns)
        
        for pattern in expected_patterns:
            if pattern.lower() in output.lower():
                matched_patterns += 1
        
        # Calculate base score based on pattern matching
        pattern_score = matched_patterns / total_patterns if total_patterns > 0 else 0.0
        
        # Bonus points for valid FFmpeg structure
        bonus_score = 0.0
        
        # Check for valid FFmpeg command start
        if output.lower().startswith('ffmpeg'):
            bonus_score += 0.1
        
        # Check for proper -filter_complex usage (if overlay/zoom effects expected)
        overlay_patterns = ['overlay=', 'zoompan=']
        if any(pattern in expected_patterns for pattern in overlay_patterns):
            if '-filter_complex' in output:
                bonus_score += 0.1
        
        # Check for -y flag (overwrite protection)
        if '-y' in output:
            bonus_score += 0.05
        
        # Validate basic FFmpeg syntax patterns
        if self._has_valid_ffmpeg_syntax(output):
            bonus_score += 0.05
        
        # Final score (pattern matching is primary, bonuses are secondary)
        final_score = min(1.0, pattern_score + bonus_score)
        
        return final_score
    
    def _has_valid_ffmpeg_syntax(self, command: str) -> bool:
        """Basic validation of FFmpeg command syntax"""
        # Check for input file specification
        if not re.search(r'-i\s+\S+', command):
            return False
        
        # Check that command doesn't have obvious syntax errors
        if re.search(r'-i\s+(-f\s+null|$|\s+-)', command):
            return False
        
        # Check for output specification (file or /dev/null)
        if not (command.endswith('/dev/null') or re.search(r'\S+\.(mp4|avi|mov|mkv|webm)(\s|$)', command)):
            return False
        
        return True


@dataclass
class OverlaySpecificEvaluator(Evaluator):
    """Specialized evaluator for overlay-specific functionality"""
    
    async def evaluate(self, ctx: EvaluatorContext[dict, str]) -> float:
        if not isinstance(ctx.output, str):
            return 0.0
        
        output = ctx.output.lower()
        metadata = getattr(ctx.case, 'metadata', {})
        effect_type = metadata.get('effect_type', '')
        
        # Score based on effect type requirements
        score = 0.0
        
        if effect_type == 'static_overlay':
            # Should have basic overlay syntax
            if 'overlay=' in output and not ('enable=' in output or '+t*' in output):
                score = 1.0
            elif 'overlay=' in output:
                score = 0.7  # Has overlay but might be more complex than needed
        
        elif effect_type == 'timed_overlay':
            # Should have enable with time conditions
            if "enable='between(t," in output or 'enable="between(t,' in output:
                score = 1.0
            elif 'enable=' in output:
                score = 0.7
        
        elif effect_type == 'moving_overlay':
            # Should have time-based position calculation
            if 'overlay=' in output and '+t*' in output:
                score = 1.0
            elif 'overlay=' in output:
                score = 0.5
        
        elif effect_type == 'fade_overlay':
            # Should have alpha channel manipulation
            if 'alpha=' in output and ('min(' in output or 'max(' in output):
                score = 1.0
            elif 'alpha=' in output:
                score = 0.7
        
        elif effect_type == 'positioned_overlay':
            # Should use FFmpeg positioning variables
            if ('w-w-' in output and 'h-h-' in output) or ('w+' in output and 'h+' in output):
                score = 1.0
            elif 'overlay=' in output:
                score = 0.6
        
        elif effect_type == 'zoom':
            # Should use zoompan filter
            if 'zoompan=' in output and 'z=' in output:
                score = 1.0
            elif 'zoompan=' in output:
                score = 0.7
        
        elif effect_type == 'preserve_filters':
            # Should maintain existing filters in filter_complex
            expected_patterns = getattr(ctx.case, 'expected_output_contains', [])
            filter_patterns = [p for p in expected_patterns if '=' in p and not p.startswith('-')]
            if all(pattern.lower() in output for pattern in filter_patterns):
                score = 1.0
            else:
                score = 0.5
        
        elif effect_type == 'complex_mapping':
            # Should preserve audio/video mapping
            if '-map' in output and 'overlay=' in output:
                score = 1.0
            elif 'overlay=' in output:
                score = 0.6
        
        else:
            # Default scoring for unknown types
            score = 0.8 if 'overlay=' in output or 'zoompan=' in output else 0.0
        
        return score


# Add evaluators to the dataset
dataset.add_evaluator(FFmpegOverlayEvaluator())
dataset.add_evaluator(OverlaySpecificEvaluator())

