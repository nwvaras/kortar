"""
Progress Manager for sharing Rich Progress instances across the application.
"""

from contextlib import contextmanager
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from typing import Generator, Optional

from rich.prompt import Prompt


class ProgressManager:
    """Manages a shared Rich Progress instance that can be accessed from anywhere in the application."""
    
    def __init__(self):
        self._progress: Optional[Progress] = None
    
    @contextmanager
    def progress_context(self) -> Generator[Progress, None, None]:
        """
        Context manager that creates a Progress instance and makes it available globally.
        
        Usage:
            with progress_manager.progress_context() as progress:
                # Progress is now available globally
                task = progress.add_task("Main task", total=100)
                # Other modules can access it via get_progress()
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            expand=True
        )
            
        try:
            with self._progress:
                yield self._progress
        finally:
            self._progress = None
    
    def get_progress(self) -> Optional[Progress]:
        """
        Get the current global Progress instance.
        
        Returns:
            The current Progress instance if one is active, None otherwise.
        """
        return self._progress
    
    def add_task(self, description: str, total: Optional[int] = None, **kwargs) -> Optional[int]:
        """
        Add a task to the current progress if one exists.
        
        Args:
            description: Task description
            total: Total number of steps (None for indeterminate progress)
            **kwargs: Additional arguments passed to Progress.add_task
            
        Returns:
            Task ID if progress is active, None otherwise
        """
        progress = self.get_progress()
        if progress:
            return progress.add_task(description, total=total, **kwargs)
        return None
    
    def update_task(self, task_id: Optional[int], **kwargs) -> bool:
        """
        Update a task in the current progress if one exists.
        
        Args:
            task_id: Task ID returned from add_task
            **kwargs: Arguments passed to Progress.update
            
        Returns:
            True if update was successful, False otherwise
        """
        if task_id is None:
            return False
            
        progress = self.get_progress()
        if progress:
            progress.update(task_id, **kwargs)
            return True
        return False
    
    def remove_task(self, task_id: Optional[int]) -> bool:
        """
        Remove a task from the current progress if one exists.
        
        Args:
            task_id: Task ID returned from add_task
            
        Returns:
            True if removal was successful, False otherwise
        """
        if task_id is None:
            return False
            
        progress = self.get_progress()
        if progress:
            progress.remove_task(task_id)
            return True
        return False
    def stop_progress(self) -> bool:
        """
        Stop the current progress if one exists.
        """
        progress = self.get_progress()
        if progress:
            progress.stop()
            return True
        return False

    def start_progress(self) -> bool:
        """
        Start the current progress if one exists.
        """
        progress = self.get_progress()
        if progress:
            progress.start()
            return True
        return False


# Global instance
progress_manager = ProgressManager()


def get_progress() -> Optional[Progress]:
    """Convenience function to get the global progress instance."""
    return progress_manager.get_progress()


def add_task(description: str, total: Optional[int] = None, **kwargs) -> Optional[int]:
    """Convenience function to add a task to the global progress."""
    return progress_manager.add_task(description, total=total, **kwargs)


def update_task(task_id: Optional[int], **kwargs) -> bool:
    """Convenience function to update a task in the global progress."""
    return progress_manager.update_task(task_id, **kwargs)


def remove_task(task_id: Optional[int]) -> bool:
    """Convenience function to remove a task from the global progress."""
    return progress_manager.remove_task(task_id)

def prompt_user(question: str) -> str:
    """Prompt the user for input."""
    return Prompt.ask(question)


def confirm_user(question: str, default: bool = True) -> bool:
    """Confirm with the user."""
    from rich.prompt import Confirm
    return Confirm.ask(question, default=default)