from pydantic_ai import RunContext
from main import main_agent


@main_agent.tool
async def ask_user_for_clarification(ctx: RunContext, question: str, context: str = "") -> str:
    """Ask the user for missing information or clarification when needed"""
    print(f"[LOG] USER_INPUT - Question: {question}")

    
    # Format the question for the user
    formatted_question = f"\n{'='*60}\n"
    formatted_question += "🤔 CLARIFICATION NEEDED\n"
    formatted_question += f"{'='*60}\n"
    
    if context:
        formatted_question += f"Context: {context}\n\n"
    
    formatted_question += f"Question: {question}\n"
    formatted_question += f"{'='*60}\n"
    
    # Print the formatted question
    print(formatted_question)
    
    # Get user input
    try:
        user_response = input("Your response: ").strip()
        
        if not user_response:
            user_response = "No response provided"
        
        print(f"[LOG] USER_INPUT - Response: {user_response}")
        return user_response
        
    except KeyboardInterrupt:
        print(f"\n[LOG] USER_INPUT - User interrupted")
        return "User cancelled the operation"
    except Exception as e:
        print(f"[LOG] USER_INPUT - Error getting input: {str(e)}")
        return f"Error getting user input: {str(e)}"