from video_assistant import main_agent
from planner import plan_video_editing, print_execution_plan, ExecutionPlan
import asyncio
import subprocess
import typer
from rich.console import Console
from rich.panel import Panel
from common.logger import get_logger
from common.progress import progress_manager, add_task, update_task, confirm_user

logger = get_logger("kortar.initial")


# Initialize rich console for better CLI experience
console = Console()
app = typer.Typer(
    name="ffmpeg-agent",
    help="🎬 FFmpeg Agent v3 - AI-powered video editing assistant",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command("interactive")
def interactive_mode():
    """🚀 Start interactive mode for conversational video editing"""
    console.print(
        Panel.fit(
            "[bold blue]🎬 FFmpeg Agent v3 - Interactive Mode[/bold blue]\n\n"
            "[yellow]Available commands:[/yellow]\n"
            "• Analyze video technical details\n"
            "• Apply filters and effects\n"
            "• Find editing opportunities\n"
            "• After typing first line, continue on next lines\n"
            "• Press Enter on empty line to submit\n"
            "• Use '\\' at end of line for forced continuation\n\n"
            "[dim]Commands: 'help' for help, 'clear' to reset, 'quit' to exit[/dim]",
            title="Welcome",
            border_style="blue",
        )
    )

    asyncio.run(_interactive_session())


async def _interactive_session():
    """Internal interactive session handler"""
    console.print("[LOG] Starting FFmpeg Agent v3...", style="dim")
    console.print(
        "[dim]💡 Multiline support: Continue typing on next lines, press Enter on empty line to submit[/dim]"
    )
    console.print(
        "[dim]💡 Commands: 'help' for help, 'clear' to reset, 'quit' to exit[/dim]"
    )

    history = []
    plan_history = []
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
                if first_line.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    return
                elif first_line.lower() == "clear":
                    history = []
                    plan_history = []
                    console.print("[green]✨ Chat history cleared![/green]")
                    continue
                elif first_line.lower() in ["help", "?"]:
                    console.print(
                        Panel.fit(
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
                            border_style="yellow",
                        )
                    )
                    continue

                user_input = first_line

                # Simple multiline support - always check for more input
                console.print(
                    "[dim](Press Enter on empty line to submit, or continue typing)[/dim]"
                )
                while True:
                    try:
                        if user_input.endswith("\\"):
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

            # First, create execution plan
            plan = None
            with progress_manager.progress_context():
                task = add_task("Creating execution plan...")

                try:
                    plan_response = await plan_video_editing(user_input, plan_history)
                    plan = plan_response.output
                    plan_history = plan_response.all_messages()
                    update_task(task, description="Plan created!")
                except Exception as e:
                    console.print(f"[red]❌ Failed to create plan: {str(e)}[/red]")
                    console.print(
                        "[yellow]Falling back to direct execution...[/yellow]"
                    )

                    # Fallback to direct main_agent execution
                    task = add_task("Processing request directly...")
                    result = await main_agent.run(user_input, message_history=history)
                    history = result.all_messages()
                    update_task(task, description="Complete!")

                    # Handle direct execution result
                    _display_result(result.output)

                    # Ask user if they want to execute this command
                    execute_command = confirm_user(
                        "\n[bold yellow]Execute this FFmpeg command?[/bold yellow]",
                        default=True,
                    )

                    if execute_command:
                        await _run_ffmpeg_command(result.output.command)
                    else:
                        console.print("[yellow]⏭️  Command skipped by user[/yellow]")
                    continue

            # If we have a plan, proceed with plan execution
            if plan:
                # Display the execution plan
                print_execution_plan(plan)

                # Ask user for confirmation
                if not confirm_user(
                    "\n[bold blue]Execute this plan?[/bold blue]", default=True
                ):
                    console.print("[yellow]Plan cancelled by user.[/yellow]")
                    continue

                # Execute the plan
                try:
                    await _execute_plan(plan, history)
                except Exception as e:
                    console.print(f"[red]❌ Plan execution failed: {str(e)}[/red]")
                    console.print(
                        "[yellow]You can try a simpler request or modify your input.[/yellow]"
                    )

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")


async def _run_ffmpeg_command(command: str) -> bool:
    """Execute an FFmpeg command and return success status"""
    try:
        console.print("\n[bold blue]🎬 Executing FFmpeg Command...[/bold blue]")
        console.print(f"[dim]Command: {command}[/dim]\n")

        with progress_manager.progress_context():
            task = add_task("Running FFmpeg...")

            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            update_task(task, description="FFmpeg execution complete!")

        if result.returncode == 0:
            console.print(
                "[bold green]✅ FFmpeg command executed successfully![/bold green]"
            )
            if result.stdout.strip():
                console.print(f"[dim]Output: {result.stdout.strip()}[/dim]")
            return True
        else:
            console.print("[bold red]❌ FFmpeg command failed![/bold red]")
            console.print(f"[red]Error: {result.stderr.strip()}[/red]")
            return False

    except subprocess.TimeoutExpired:
        console.print("[bold red]❌ FFmpeg command timed out (5 minutes)[/bold red]")
        return False
    except Exception as e:
        console.print(f"[bold red]❌ Error executing FFmpeg: {str(e)}[/bold red]")
        return False


async def _execute_plan(plan: ExecutionPlan, history: list) -> list:
    """Execute an execution plan by running each task through main_agent"""
    console.print(f"\n[bold blue]🎬 Executing Plan: {plan.description}[/bold blue]")
    console.print(f"[dim]Total tasks: {len(plan.tasks)}[/dim]\n")

    for i, task in enumerate(plan.tasks, 1):
        console.print(
            f"[bold yellow]Task {i}/{len(plan.tasks)}: {task.name}[/bold yellow]"
        )
        console.print(f"[dim]{task.description}[/dim]")

        # Create task request with current context
        task_request = f"""
Task: {task.description}
Inputs file: {task.inputs}
Expected output: {task.output_file_path or f"task_{i}_output.mp4"}
Task type: {task.task_type.value}
"""

        if task.time_interval:
            task_request += f"\nTime interval: {task.time_interval}"

        console.print(f"[dim]Processing task {i}...[/dim]")

        result = await main_agent.run(task_request, message_history=history)
        history = result.all_messages()

        # Display task result
        _display_result(result.output)

        # Ask user if they want to execute this command
        execute_command = confirm_user(
            f"\n[bold yellow]Execute this FFmpeg command for task {i}?[/bold yellow]",
            default=True,
        )

        if execute_command:
            # Execute the FFmpeg command
            success = await _run_ffmpeg_command(result.output.command)
            if not success:
                console.print(f"[red]❌ Task {i} execution failed[/red]")

                # Ask if user wants to continue with remaining tasks
                continue_plan = confirm_user(
                    "\n[bold red]Continue with remaining tasks despite this failure?[/bold red]",
                    default=False,
                )
                if not continue_plan:
                    console.print("[yellow]Plan execution cancelled by user.[/yellow]")
                    return history
            else:
                # Update current video path for next task
                if task.output_file_path:
                    pass
        else:
            console.print(f"[yellow]⏭️  Task {i} command skipped by user[/yellow]")
            # Still update the path as if the command was executed (for planning continuity)
            if task.output_file_path:
                pass

        console.print(f"[green]✅ Task {i} completed[/green]\n")

    console.print("[bold green]🎉 All tasks completed!")
    return history


def _display_result(output):
    """Display command result in a copy-friendly format"""

    # Display the command in a copy-friendly format first
    console.print("\n[bold cyan]📋 FFmpeg Command (copy-ready):[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    # Print command with no formatting at all for easy copying
    console.print(output.command, highlight=False)
    console.print(f"[dim]{'─' * 60}[/dim]")

    # Then show additional details in a formatted way
    console.print("\n[bold yellow]📝 Explanation:[/bold yellow]")
    # Print explanation with no formatting for easy copying if needed
    console.print(output.explanation, highlight=False)

    if output.filters_used:
        console.print("\n[bold magenta]🔧 Filters Used:[/bold magenta]")
        console.print(", ".join(output.filters_used), highlight=False)


if __name__ == "__main__":
    # Check if any arguments were passed, if not, start interactive mode
    import sys

    if len(sys.argv) == 1:
        console.print(
            "[yellow]No command specified, starting interactive mode...[/yellow]"
        )
        asyncio.run(_interactive_session())
    else:
        app()
