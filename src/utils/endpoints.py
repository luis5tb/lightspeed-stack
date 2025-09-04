"""Utility functions for endpoint handlers."""

import logging
from fastapi import HTTPException, status

import constants
from models.requests import QueryRequest
from models.database.conversations import UserConversation
from models.config import Action
from app.database import get_session
from configuration import AppConfig


logger = logging.getLogger("utils.endpoints")


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation according to its ID."""
    with get_session() as session:
        db_conversation = (
            session.query(UserConversation).filter_by(id=conversation_id).first()
        )
        if db_conversation:
            session.delete(db_conversation)
            session.commit()
            logger.info("Deleted conversation %s from local database", conversation_id)
        else:
            logger.info(
                "Conversation %s not found in local database, it may have already been deleted",
                conversation_id,
            )


def validate_conversation_ownership(
    user_id: str, conversation_id: str, others_allowed: bool = False
) -> UserConversation | None:
    """Validate that the conversation belongs to the user.

    Validates that the conversation with the given ID belongs to the user with the given ID.
    If `others_allowed` is True, it allows conversations that do not belong to the user,
    which is useful for admin access.
    """
    with get_session() as session:
        conversation_query = session.query(UserConversation)

        filtered_conversation_query = (
            conversation_query.filter_by(id=conversation_id)
            if others_allowed
            else conversation_query.filter_by(id=conversation_id, user_id=user_id)
        )

        conversation: UserConversation | None = filtered_conversation_query.first()

        return conversation


def check_configuration_loaded(config: AppConfig) -> None:
    """Check that configuration is loaded and raise exception when it is not."""
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"response": "Configuration is not loaded"},
        )


def get_system_prompt(query_request: QueryRequest, config: AppConfig, has_mcp_tools: bool = False) -> str:
    """Get the system prompt: the provided one, configured one, or default one."""
    system_prompt_disabled = (
        config.customization is not None
        and config.customization.disable_query_system_prompt
    )
    if system_prompt_disabled and query_request.system_prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "response": (
                    "This instance does not support customizing the system prompt in the "
                    "query request (disable_query_system_prompt is set). Please remove the "
                    "system_prompt field from your request."
                )
            },
        )

    base_prompt = ""
    if query_request.system_prompt:
        # Query taking precedence over configuration is the only behavior that
        # makes sense here - if the configuration wants precedence, it can
        # disable query system prompt altogether with disable_system_prompt.
        base_prompt = query_request.system_prompt
    elif (
        config.customization is not None
        and config.customization.system_prompt is not None
    ):
        base_prompt = config.customization.system_prompt
    else:
        # default system prompt has the lowest precedence
        base_prompt = constants.DEFAULT_SYSTEM_PROMPT

    # Add tool usage instructions when MCP tools are available
    if has_mcp_tools:
        tool_instruction = (
            "\n\nWhen answering questions, use the available tools to get accurate, "
            "real-time information. If a user asks about clusters, resources, status, "
            "or other infrastructure-related topics, use the tools to query the actual "
            "system state rather than providing generic responses."
        )
        base_prompt += tool_instruction

    return base_prompt


def validate_model_provider_override(
    query_request: QueryRequest, authorized_actions: set[Action] | frozenset[Action]
) -> None:
    """Validate whether model/provider overrides are allowed by RBAC.

    Raises HTTP 403 if the request includes model or provider and the caller
    lacks Action.MODEL_OVERRIDE permission.
    """
    if (query_request.model is not None or query_request.provider is not None) and (
        Action.MODEL_OVERRIDE not in authorized_actions
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "response": (
                    "This instance does not permit overriding model/provider in the query request "
                    "(missing permission: MODEL_OVERRIDE). Please remove the model and provider "
                    "fields from your request."
                )
            },
        )


def get_rag_tools(vector_db_ids: list[str]) -> list[dict] | None:
    """Convert vector DB IDs to tools format for responses API."""
    if not vector_db_ids:
        return None

    return [{
        "type": "file_search",
        "vector_store_ids": vector_db_ids,
        "max_num_results": 10
    }]


def get_mcp_tools(mcp_servers: list) -> list[dict]:
    """Convert MCP servers to tools format for responses API."""
    tools = []
    for mcp_server in mcp_servers:
        tools.append({
            "type": "mcp",
            "server_label": mcp_server.name,
            "server_url": mcp_server.url,
            "require_approval": "never"
        })
    return tools
