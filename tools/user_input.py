from pydantic_ai import RunContext
from main import main_agent
from common.logger import get_logger
from common.user_clarification import get_user_clarification
from tools.content_analysis import gemini_agent

logger = get_logger("kortar.tools.user_input")


@gemini_agent.tool
async def ask_user_for_clarification_gemini(
    ctx: RunContext, question: str, context: str = ""
) -> str:
    """Ask the user for missing information or clarification when needed"""
    return await get_user_clarification(question, context)


@main_agent.tool
async def ask_user_for_clarification(
    ctx: RunContext, question: str, context: str = ""
) -> str:
    """Ask the user for missing information or clarification when needed"""
    return await get_user_clarification(question, context)
