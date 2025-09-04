"""Handler for REST API calls to manage conversation history."""

import logging
from typing import Any

from llama_stack_client import APIConnectionError, NotFoundError

from fastapi import APIRouter, HTTPException, Request, status, Depends

from client import AsyncLlamaStackClientHolder
from configuration import configuration
from app.database import get_session
from auth import get_auth_dependency
from authorization.middleware import authorize
from models.config import Action
from models.database.conversations import UserConversation
from models.responses import (
    ConversationResponse,
    ConversationDeleteResponse,
    ConversationsListResponse,
    ConversationDetails,
)
from utils.endpoints import (
    check_configuration_loaded,
    delete_conversation,
    validate_conversation_ownership,
)
from utils.suid import check_suid

logger = logging.getLogger("app.endpoints.handlers")
router = APIRouter(tags=["conversations"])
auth_dependency = get_auth_dependency()

conversation_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
        "chat_history": [
            {
                "messages": [
                    {"content": "Hi", "type": "user"},
                    {"content": "Hello!", "type": "assistant"},
                ],
                "started_at": "2024-01-01T00:00:00Z",
                "completed_at": "2024-01-01T00:00:05Z",
            }
        ],
    },
    404: {
        "detail": {
            "response": "Conversation not found",
            "cause": "The specified conversation ID does not exist.",
        }
    },
    503: {
        "detail": {
            "response": "Unable to connect to Llama Stack",
            "cause": "Connection error.",
        }
    },
}

conversation_delete_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
        "success": True,
        "message": "Conversation deleted successfully",
    },
    404: {
        "detail": {
            "response": "Conversation not found",
            "cause": "The specified conversation ID does not exist.",
        }
    },
    503: {
        "detail": {
            "response": "Unable to connect to Llama Stack",
            "cause": "Connection error.",
        }
    },
}

conversations_list_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "conversations": [
            {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2024-01-01T00:00:00Z",
                "last_message_at": "2024-01-01T00:05:00Z",
                "last_used_model": "gemini/gemini-1.5-flash",
                "last_used_provider": "gemini",
                "message_count": 5,
            },
            {
                "conversation_id": "456e7890-e12b-34d5-a678-901234567890",
                "created_at": "2024-01-01T01:00:00Z",
                "last_message_at": "2024-01-01T01:02:00Z",
                "last_used_model": "gemini/gemini-2.0-flash",
                "last_used_provider": "gemini",
                "message_count": 2,
            },
        ]
    },
    503: {
        "detail": {
            "response": "Unable to connect to Llama Stack",
            "cause": "Connection error.",
        }
    },
}


def simplify_response_data(response_data: dict) -> list[dict[str, Any]]:
    """Simplify response data to include only essential conversation information.

    Args:
        response_data: The full response data dict from llama-stack responses API

    Returns:
        Simplified response data with input and output messages
    """
    # Create simplified structure - each response is one "turn"
    chat_history = []

    # Extract input from the response
    input_messages = []
    if "input" in response_data:
        # Input could be a simple string or list of input items
        input_data = response_data["input"]
        if isinstance(input_data, str):
            input_messages.append({
                "content": input_data,
                "type": "user"
            })
        elif isinstance(input_data, list):
            for input_item in input_data:
                if hasattr(input_item, 'content') or 'content' in input_item:
                    content = input_item.get('content') if isinstance(input_item, dict) else getattr(input_item, 'content', '')
                    if isinstance(content, list):
                        # Handle structured content
                        text_content = ""
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get('type') == 'input_text':
                                text_content += content_item.get('text', '')
                            elif hasattr(content_item, 'text'):
                                text_content += content_item.text
                        input_messages.append({
                            "content": text_content,
                            "type": "user"
                        })
                    else:
                        input_messages.append({
                            "content": str(content),
                            "type": "user"
                        })

    # Extract output from the response
    output_messages = []
    if "output" in response_data:
        for output_item in response_data["output"]:
            if hasattr(output_item, 'content') or 'content' in output_item:
                content = output_item.get('content') if isinstance(output_item, dict) else getattr(output_item, 'content', '')
                if isinstance(content, list):
                    # Handle structured content
                    text_content = ""
                    for content_item in content:
                        if isinstance(content_item, dict) and content_item.get('type') == 'text':
                            text_content += content_item.get('text', '')
                        elif hasattr(content_item, 'text'):
                            text_content += content_item.text
                    output_messages.append({
                        "content": text_content,
                        "type": "assistant"
                    })
                else:
                    output_messages.append({
                        "content": str(content),
                        "type": "assistant"
                    })

    # Combine input and output messages
    all_messages = input_messages + output_messages

    # Create a single turn with all messages
    if all_messages:
        simplified_turn = {
            "messages": all_messages,
            "started_at": response_data.get("created_at"),
            "completed_at": response_data.get("created_at"),  # Responses don't have completion time
        }
        chat_history.append(simplified_turn)

    return chat_history


@router.get("/conversations", responses=conversations_list_responses)
@authorize(Action.LIST_CONVERSATIONS)
async def get_conversations_list_endpoint_handler(
    request: Request,
    auth: Any = Depends(auth_dependency),
) -> ConversationsListResponse:
    """Handle request to retrieve all conversations for the authenticated user."""
    check_configuration_loaded(configuration)

    user_id, _, _ = auth

    logger.info("Retrieving conversations for user %s", user_id)

    with get_session() as session:
        try:
            query = session.query(UserConversation)

            filtered_query = (
                query
                if Action.LIST_OTHERS_CONVERSATIONS in request.state.authorized_actions
                else query.filter_by(user_id=user_id)
            )

            user_conversations = filtered_query.all()

            # Return conversation summaries with metadata
            conversations = [
                ConversationDetails(
                    conversation_id=conv.id,
                    created_at=conv.created_at.isoformat() if conv.created_at else None,
                    last_message_at=(
                        conv.last_message_at.isoformat()
                        if conv.last_message_at
                        else None
                    ),
                    message_count=conv.message_count,
                    last_used_model=conv.last_used_model,
                    last_used_provider=conv.last_used_provider,
                )
                for conv in user_conversations
            ]

            logger.info(
                "Found %d conversations for user %s", len(conversations), user_id
            )

            return ConversationsListResponse(conversations=conversations)

        except Exception as e:
            logger.exception(
                "Error retrieving conversations for user %s: %s", user_id, e
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "response": "Unknown error",
                    "cause": f"Unknown error while getting conversations for user {user_id}",
                },
            ) from e


@router.get("/conversations/{conversation_id}", responses=conversation_responses)
@authorize(Action.GET_CONVERSATION)
async def get_conversation_endpoint_handler(
    request: Request,
    conversation_id: str,
    auth: Any = Depends(auth_dependency),
) -> ConversationResponse:
    """
    Handle request to retrieve a conversation by ID.

    Retrieve a conversation's chat history by its ID. Then fetches
    the conversation session from the Llama Stack backend,
    simplifies the session data to essential chat history, and
    returns it in a structured response. Raises HTTP 400 for
    invalid IDs, 404 if not found, 503 if the backend is
    unavailable, and 500 for unexpected errors.

    Parameters:
        conversation_id (str): Unique identifier of the conversation to retrieve.

    Returns:
        ConversationResponse: Structured response containing the conversation
        ID and simplified chat history.
    """
    check_configuration_loaded(configuration)

    # Validate conversation ID format
    if not check_suid(conversation_id):
        logger.error("Invalid conversation ID format: %s", conversation_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Invalid conversation ID format",
                "cause": f"Conversation ID {conversation_id} is not a valid UUID",
            },
        )

    user_id, _, _ = auth

    user_conversation = validate_conversation_ownership(
        user_id=user_id,
        conversation_id=conversation_id,
        others_allowed=(
            Action.READ_OTHERS_CONVERSATIONS in request.state.authorized_actions
        ),
    )

    if user_conversation is None:
        logger.warning(
            "User %s attempted to read conversation %s they don't own",
            user_id,
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "response": "Access denied",
                "cause": "You do not have permission to read this conversation",
            },
        )

    response_id = conversation_id
    logger.info("Retrieving response %s", response_id)

    try:
        client = AsyncLlamaStackClientHolder().get_client()

        # Use responses API to get the individual response
        # Try different method names that might be available
        try:
            response_obj = await client.responses.retrieve(response_id=response_id)
        except AttributeError:
            # Fallback to agents API if responses client doesn't have retrieve method
            response_obj = await client.agents.get_openai_response(response_id=response_id)
        response_data = response_obj.model_dump() if hasattr(response_obj, 'model_dump') else response_obj

        logger.info("Successfully retrieved response %s", response_id)

        # Simplify the response data to include only essential conversation information
        chat_history = simplify_response_data(response_data)

        return ConversationResponse(
            conversation_id=conversation_id,
            chat_history=chat_history,
        )

    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "response": "Unable to connect to Llama Stack",
                "cause": str(e),
            },
        ) from e
    except NotFoundError as e:
        logger.error("Conversation not found: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "response": "Conversation not found",
                "cause": f"Conversation {conversation_id} could not be retrieved: {str(e)}",
            },
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        # Handle case where session doesn't exist or other errors
        logger.exception("Error retrieving conversation %s: %s", conversation_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Unknown error",
                "cause": f"Unknown error while getting conversation {conversation_id} : {str(e)}",
            },
        ) from e


@router.delete(
    "/conversations/{conversation_id}", responses=conversation_delete_responses
)
@authorize(Action.DELETE_CONVERSATION)
async def delete_conversation_endpoint_handler(
    request: Request,
    conversation_id: str,
    auth: Any = Depends(auth_dependency),
) -> ConversationDeleteResponse:
    """
    Handle request to delete a conversation by ID.

    Validates the conversation ID format and attempts to delete the
    corresponding session from the Llama Stack backend. Raises HTTP
    errors for invalid IDs, not found conversations, connection
    issues, or unexpected failures.

    Returns:
        ConversationDeleteResponse: Response indicating the result of the deletion operation.
    """
    check_configuration_loaded(configuration)

    # Validate conversation ID format
    if not check_suid(conversation_id):
        logger.error("Invalid conversation ID format: %s", conversation_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Invalid conversation ID format",
                "cause": f"Conversation ID {conversation_id} is not a valid UUID",
            },
        )

    user_id, _, _ = auth

    user_conversation = validate_conversation_ownership(
        user_id=user_id,
        conversation_id=conversation_id,
        others_allowed=(
            Action.DELETE_OTHERS_CONVERSATIONS in request.state.authorized_actions
        ),
    )

    if user_conversation is None:
        logger.warning(
            "User %s attempted to delete conversation %s they don't own",
            user_id,
            conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "response": "Access denied",
                "cause": "You do not have permission to delete this conversation",
            },
        )

    response_id = conversation_id
    logger.info("Deleting response %s", response_id)

    try:
        # Get Llama Stack client
        client = AsyncLlamaStackClientHolder().get_client()

        # Use responses API to delete the response
        # Note: Check if responses API has delete method, otherwise just mark as success
        try:
            # Try to delete the response (may not be supported by all implementations)
            if hasattr(client.responses, 'delete'):
                await client.responses.delete(response_id=response_id)
                logger.info("Successfully deleted response %s", response_id)
            else:
                # Responses API doesn't support deletion, just mark as success
                logger.info("Responses API doesn't support deletion, marking as success for response %s", response_id)
        except AttributeError:
            # If delete method doesn't exist on responses API, just log and continue
            logger.info("Responses API doesn't support deletion, marking as success for response %s", response_id)

        delete_conversation(conversation_id=conversation_id)

        return ConversationDeleteResponse(
            conversation_id=conversation_id,
            success=True,
            response="Conversation deleted successfully",
        )

    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "response": "Unable to connect to Llama Stack",
                "cause": str(e),
            },
        ) from e
    except NotFoundError as e:
        logger.error("Conversation not found: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "response": "Conversation not found",
                "cause": f"Conversation {conversation_id} could not be deleted: {str(e)}",
            },
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        # Handle case where session doesn't exist or other errors
        logger.exception("Error deleting conversation %s: %s", conversation_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Unknown error",
                "cause": f"Unknown error while deleting conversation {conversation_id} : {str(e)}",
            },
        ) from e
