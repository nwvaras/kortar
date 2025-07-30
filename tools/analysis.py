import subprocess
import json
from pydantic_ai import RunContext
from main import main_agent


@main_agent.tool
async def initial_video_analysis(ctx: RunContext, video_path: str) -> str:
    """Run ffprobe to analyze video technical characteristics"""
    print(f"[LOG] FFPROBE - Analyzing: {video_path}")
    
    try:
        # Run ffprobe to get detailed video information
        ffprobe_cmd = [
            'ffprobe', 
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(
            ffprobe_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            error_msg = f"ffprobe failed: {result}"
            print(f"[LOG] FFPROBE - Error: {result.__dict__}")
            return f"Error analyzing video: {error_msg}"
        
        # Parse JSON output
        probe_data = json.loads(result.stdout)
        
        # Extract relevant information
        format_info = probe_data.get('format', {})
        streams = probe_data.get('streams', [])
        
        # Find video and audio streams
        video_stream = None
        audio_streams = []
        
        for stream in streams:
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_streams.append(stream)
        
        # Build analysis report
        analysis = []
        analysis.append(f"**File:** {video_path}")
        analysis.append(f"**Duration:** {format_info.get('duration', 'unknown')} seconds")
        analysis.append(f"**Size:** {format_info.get('size', 'unknown')} bytes")
        analysis.append(f"**Format:** {format_info.get('format_name', 'unknown')}")
        
        if video_stream:
            width = video_stream.get('width', 'unknown')
            height = video_stream.get('height', 'unknown')
            fps_data = video_stream.get('r_frame_rate', '0/1').split('/')
            fps = round(int(fps_data[0]) / int(fps_data[1]), 2) if len(fps_data) == 2 and fps_data[1] != '0' else 'unknown'
            
            analysis.append(f"**Video Resolution:** {width}x{height}")
            analysis.append(f"**Video FPS:** {fps}")
            analysis.append(f"**Video Codec:** {video_stream.get('codec_name', 'unknown')}")
            analysis.append(f"**Video Bitrate:** {video_stream.get('bit_rate', 'unknown')} bps")
        else:
            analysis.append("**Video:** No video stream found")
        
        if audio_streams:
            analysis.append(f"**Audio Streams:** {len(audio_streams)}")
            for i, audio in enumerate(audio_streams):
                analysis.append(f"  - Stream {i}: {audio.get('codec_name', 'unknown')}, {audio.get('channels', 'unknown')} channels, {audio.get('sample_rate', 'unknown')} Hz")
        else:
            analysis.append("**Audio:** No audio streams found")
        
        final_analysis = "\n".join(analysis)
        print(f"[LOG] FFPROBE - Result: {final_analysis}")
        return final_analysis
        
    except subprocess.TimeoutExpired:
        error_msg = "ffprobe command timed out"
        print(f"[LOG] FFPROBE - Error: {error_msg}")
        return f"Error: {error_msg}"
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse ffprobe output: {str(e)}"
        print(f"[LOG] FFPROBE - Error: {error_msg}")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error during video analysis: {str(e)}"
        print(f"[LOG] FFPROBE - Error: {error_msg}")
        return f"Error: {error_msg}"