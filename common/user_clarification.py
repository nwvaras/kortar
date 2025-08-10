from rich.console import Console
from common.logger import get_logger
from common.progress import prompt_user

console = Console()
logger = get_logger("kortar.common.user_clarification")


async def get_user_clarification(question: str, context: str = "") -> str:
    """Core function to ask the user for missing information or clarification when needed"""
    logger.info("Requesting user clarification", question=question, context=context)

    # Format the question for the user
    formatted_question = f"\n{'=' * 60}\n"
    formatted_question += "🤔 CLARIFICATION NEEDED\n"
    formatted_question += f"{'=' * 60}\n"

    if context:
        formatted_question += f"Context: {context}\n\n"

    formatted_question += f"Question: {question}\n"
    formatted_question += f"{'=' * 60}\n"

    # Print the formatted question (this is user interface, not logging)
    console.print(formatted_question)

    # Get user input
    try:
        user_response = prompt_user("Your response")

        if not user_response:
            user_response = "No response provided"

        logger.info("User response received", response=user_response)
        return user_response

    except KeyboardInterrupt:
        logger.info("User interrupted the input process")
        return "User cancelled the operation"
    except Exception as e:
        logger.error("Error getting user input", error=str(e))
        return f"Error getting user input: {str(e)}"
