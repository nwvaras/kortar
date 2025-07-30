from main import VideoDeps, main_agent, FFmpegCommand
import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

# Import all tools to register them with main_agent
import tools
from tools.analysis import initial_video_analysis
from tools.content_analysis import analyze_video
from tools.doctor import doctor_command


# Initialize rich console for better CLI experience
console = Console()
app = typer.Typer(
    name="ffmpeg-agent",
    help="🎬 FFmpeg Agent v3 - AI-powered video editing assistant",
    add_completion=False,
    rich_markup_mode="rich"
)

@app.command("interactive")
def interactive_mode():
    """🚀 Start interactive mode for conversational video editing"""
    console.print(Panel.fit(
        "[bold blue]🎬 FFmpeg Agent v3 - Interactive Mode[/bold blue]\n\n"
        "[yellow]Available commands:[/yellow]\n"
        "• Analyze video technical details\n"
        "• Apply filters and effects\n"
        "• Find editing opportunities\n"
        "• Fix problematic commands\n\n"
        "[green]Multiline Input Support:[/green]\n"
        "• After typing first line, continue on next lines\n"
        "• Press Enter on empty line to submit\n"
        "• Use '\\' at end of line for forced continuation\n\n"
        "[dim]Commands: 'help' for help, 'clear' to reset, 'quit' to exit[/dim]",
        title="Welcome",
        border_style="blue"
    ))
    
    asyncio.run(_interactive_session())

@app.command("analyze")
def analyze_video_file(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    technical: bool = typer.Option(False, "--technical", "-t", help="Run technical analysis with ffprobe"),
    content: bool = typer.Option(False, "--content", "-c", help="Run content analysis with AI"),
    query: str = typer.Option("", "--query", "-q", help="Specific query for content analysis")
):
    """📊 Analyze video file (technical specs and/or content)"""
    
    if not technical and not content:
        technical = True  # Default to technical analysis
    
    asyncio.run(_analyze_video(video_path, technical, content, query))

@app.command("edit")
def edit_video(
    request: str = typer.Argument(..., help="Video editing request"),
    video_path: str = typer.Option("", "--video", "-v", help="Video file path (if not in request)"),
    output: str = typer.Option("", "--output", "-o", help="Output file name"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show command without executing")
):
    """✨ Apply video editing effects based on natural language request"""
    
    full_request = f"{request} {video_path}".strip() if video_path else request
    if output:
        full_request += f" save as {output}"
    
    asyncio.run(_process_edit_request(full_request, dry_run))

@app.command("doctor")
def doctor_command_cli(
    intent: str = typer.Argument(..., help="Original intent of the command"),
    failing_command: str = typer.Argument(..., help="The FFmpeg command that failed"),
    error: str = typer.Option("", "--error", "-e", help="Error message (optional)")
):
    """🩺 Fix a problematic FFmpeg command"""
    
    asyncio.run(_doctor_fix(intent, failing_command, error))

async def _interactive_session():
    """Internal interactive session handler"""
    console.print("[LOG] Starting FFmpeg Agent v3...", style="dim")
    console.print("[dim]💡 Multiline support: Continue typing on next lines, press Enter on empty line to submit[/dim]")
    console.print("[dim]💡 Commands: 'help' for help, 'clear' to reset, 'quit' to exit[/dim]")
    
    history = []
    while True:
        try:
            # Support multiline input
            console.print()
            console.print("[dim]Enter your request:[/dim]")
            
            user_input = ""
            try:
                # Get first line
                first_line = input("❯ ").strip()
                
                # Check for special commands
                if first_line.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    return
                elif first_line.lower() == 'clear':
                    history = []
                    console.print("[green]✨ Chat history cleared![/green]")
                    continue
                elif first_line.lower() in ['help', '?']:
                    console.print(Panel.fit(
                        "[bold yellow]🎬 FFmpeg Agent v3 - Help[/bold yellow]\n\n"
                        "[green]Available Commands:[/green]\n"
                        "• quit/exit/q - Exit the program\n"
                        "• clear - Clear chat history\n"
                        "• help/? - Show this help\n\n"
                        "[green]Multiline Input:[/green]\n"
                        "• After first line, continue typing on next lines\n"
                        "• Press Enter on empty line to submit\n"
                        "• Use '\\' at end of line for forced continuation\n"
                        "• Example:\n"
                        "  [dim]❯ Analyze video.mp4 and\n"
                        "  ... find all the moments where\n"
                        "  ... nothing is happening\n"
                        "  ... [press Enter on empty line][/dim]\n\n"
                        "[green]Common Requests:[/green]\n"
                        "• Analyze video for editing opportunities\n"
                        "• Crop/trim specific sections\n"
                        "• Add overlays, text, transitions\n"
                        "• Fix problematic FFmpeg commands",
                        title="Help",
                        border_style="yellow"
                    ))
                    continue
                
                user_input = first_line
                
                # Simple multiline support - always check for more input
                console.print("[dim](Press Enter on empty line to submit, or continue typing)[/dim]")
                while True:
                    try:
                        if user_input.endswith('\\'):
                            # Remove backslash and continue
                            user_input = user_input[:-1] + " "
                            next_line = input("... ").strip()
                            user_input += next_line
                        else:
                            # Check for additional lines
                            next_line = input("... ")
                            if not next_line.strip():  # Empty line = done
                                break
                            user_input += " " + next_line.strip()
                    except EOFError:
                        break
                    
            except EOFError:
                # Handle Ctrl+D
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Show processing indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing request...", total=None)
                
                result = await main_agent.run(
                    user_input, message_history=history
                )
                progress.update(task, description="Complete!")
            
            history = result.all_messages()
            
            # Display results in a nice format
            _display_result(result.output)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")

async def _analyze_video(video_path: str, technical: bool, content: bool, query: str):
    """Internal video analysis handler"""
    console.print(f"[LOG] Analyzing video: {video_path}", style="dim")
    
    try:
        if technical:
            console.print("[blue]🔍 Running technical analysis...[/blue]")
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                task = progress.add_task("Analyzing with ffprobe...", total=None)
                tech_result = await initial_video_analysis(None, video_path)
                
            console.print(f"\n[bold blue]🔍 Technical Analysis:[/bold blue]")
            console.print(f"[dim]{'─' * 60}[/dim]")
            # Print analysis with no formatting for easy copying
            print(tech_result)
            console.print(f"[dim]{'─' * 60}[/dim]")
        
        if content:
            content_query = query if query else "Analyze this video for editing opportunities"
            console.print(f"[green]🎯 Running content analysis: {content_query}[/green]")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                task = progress.add_task("Analyzing with AI...", total=None)
                content_result = await analyze_video(None, video_path, content_query)
                
            console.print(f"\n[bold green]🎯 Content Analysis:[/bold green]")
            console.print(f"[dim]{'─' * 60}[/dim]")
            # Print analysis with no formatting for easy copying
            print(content_result)
            console.print(f"[dim]{'─' * 60}[/dim]")
            
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")

async def _process_edit_request(request: str, dry_run: bool):
    """Internal edit request handler"""
    console.print(f"[LOG] Processing edit request: {request}", style="dim")
    
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("Generating FFmpeg command...", total=None)
            result = await main_agent.run(request)
        
        _display_result(result.output)
        
        if dry_run:
            console.print("[yellow]📋 Dry run mode - command not executed[/yellow]")
        else:
            if Confirm.ask("Execute this command?"):
                console.print("[green]🚀 Executing command...[/green]")
                # Here you could add actual command execution
                console.print("[yellow]⚠️ Command execution not implemented yet[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Edit request failed: {str(e)}[/red]")

async def _doctor_fix(intent: str, failing_command: str, error: str):
    """Internal doctor fix handler"""
    console.print("[blue]🩺 Analyzing and fixing command...[/blue]")
    
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("Diagnosing issues...", total=None)
            fixed_command = await doctor_command(None, intent, failing_command, error)
        
        console.print(f"\n[bold red]❌ Original Command:[/bold red]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        # Print commands with no formatting for easy copying
        print(failing_command)
        console.print(f"[dim]{'─' * 60}[/dim]")
        
        console.print(f"\n[bold green]✅ Fixed Command (copy-ready):[/bold green]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        print(fixed_command)
        console.print(f"[dim]{'─' * 60}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Doctor fix failed: {str(e)}[/red]")

def _display_result(output):
    """Display command result in a copy-friendly format"""
    
    # Display the command in a copy-friendly format first
    console.print("\n[bold cyan]📋 FFmpeg Command (copy-ready):[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    # Print command with no formatting at all for easy copying
    print(output.command)
    console.print(f"[dim]{'─' * 60}[/dim]")
    
    # Then show additional details in a formatted way
    console.print(f"\n[bold yellow]📝 Explanation:[/bold yellow]")
    # Print explanation with no formatting for easy copying if needed
    print(output.explanation)
    
    if output.filters_used:
        console.print(f"\n[bold magenta]🔧 Filters Used:[/bold magenta]")
        print(", ".join(output.filters_used))

# Legacy main function for backwards compatibility
async def main():
    console.print("[LOG] Starting legacy interactive mode...", style="dim")
    await _interactive_session()

if __name__ == "__main__":
    # Check if any arguments were passed, if not, start interactive mode
    import sys
    if len(sys.argv) == 1:
        console.print("[yellow]No command specified, starting interactive mode...[/yellow]")
        asyncio.run(_interactive_session())
    else:
        app() 