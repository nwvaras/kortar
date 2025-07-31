from pydantic_evals import Case, Dataset

# Test cases for overlay agent functionality

case1 = Case(
    name='static_overlay_basic',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4',
        'request': 'Add a logo overlay at position 10,10',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        'overlay=10:10',
        '-y'
    ],
    metadata={'difficulty': 'easy', 'effect_type': 'static_overlay'}
)

case2 = Case(
    name='timed_overlay',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4',
        'request': 'Add overlay that appears between 5 and 15 seconds',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        "enable='between(t,5,15)'",
        '-y'
    ],
    metadata={'difficulty': 'medium', 'effect_type': 'timed_overlay'}
)

case3 = Case(
    name='moving_overlay',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4',
        'request': 'Add overlay that moves horizontally from left to right',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        'overlay=',
        '+t*',
        '-y'
    ],
    metadata={'difficulty': 'medium', 'effect_type': 'moving_overlay'}
)

case4 = Case(
    name='fade_overlay',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4',
        'request': 'Add overlay with fade-in effect',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        "alpha='min(1,t/2)'",
        '-y'
    ],
    metadata={'difficulty': 'medium', 'effect_type': 'fade_overlay'}
)

case5 = Case(
    name='corner_positioned_overlay',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4',
        'request': 'Add overlay in bottom-right corner with 10px padding',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        'overlay=W-w-10:H-h-10',
        '-y'
    ],
    metadata={'difficulty': 'easy', 'effect_type': 'positioned_overlay'}
)

case6 = Case(
    name='zoom_effect',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4',
        'request': 'Add zoom effect that zooms in from 1x to 2x between 5 and 10 seconds',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        'zoompan=z=',
        "enable='between(on,",
        '-y'
    ],
    metadata={'difficulty': 'hard', 'effect_type': 'zoom'}
)

case7 = Case(
    name='preserve_existing_filters',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -vf scale=1920:1080,fps=30 -c:v libx264 output.mp4',
        'request': 'Add static overlay at top-left corner',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        'scale=1920:1080',
        'fps=30',
        'overlay=',
        '-y'
    ],
    metadata={'difficulty': 'medium', 'effect_type': 'preserve_filters'}
)

case8 = Case(
    name='complex_command_with_audio',
    inputs={
        'current_command': 'ffmpeg -i input.mp4 -i audio.mp3 -map 0:v -map 1:a -c:v libx264 -c:a aac output.mp4',
        'request': 'Add watermark overlay in center',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=[
        'ffmpeg',
        '-filter_complex',
        'overlay=',
        '-map',
        '-c:a aac',
        '-y'
    ],
    metadata={'difficulty': 'hard', 'effect_type': 'complex_mapping'}
)

# Validation test cases
validation_case1 = Case(
    name='invalid_command_validation',
    inputs={
        'current_command': 'not_ffmpeg -i input.mp4 output.mp4',
        'request': 'Add overlay',
        'video_path': '/path/to/input.mp4'
    },
    expected_output_contains=['ffmpeg'],
    metadata={'difficulty': 'easy', 'validation': True}
)

validation_case2 = Case(
    name='missing_input_validation',
    inputs={
        'current_command': 'ffmpeg -i -c:v libx264 output.mp4',
        'request': 'Add overlay',
        'video_path': '/path/to/input.mp4'
    },
    should_retry=True,
    metadata={'difficulty': 'medium', 'validation': True}
)

# Create dataset with all test cases
dataset = Dataset(cases=[
    case1, case2, case3, case4, case5, case6, case7, case8,
    validation_case1, validation_case2
])

# Export for easy import
__all__ = ['dataset', 'case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7', 'case8', 'validation_case1', 'validation_case2']
