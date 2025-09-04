"""Handler for REST API call to provide answer to streaming query."""

import ast
import json
import re
import logging
from typing import Annotated, Any, AsyncIterator, Iterator, cast

from llama_stack_client import APIConnectionError
from llama_stack_client import AsyncLlamaStackClient  # type: ignore
from llama_stack_client.lib.agents.event_logger import interleaved_content_as_str
# Removed unused AgentTurnResponseStreamChunk import - no longer needed for responses API
from llama_stack_client.types.shared import ToolCall
from llama_stack_client.types.shared.interleaved_content_item import TextContentItem
from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseObjectStream,
)

from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse

from auth import get_auth_dependency
from auth.interface import AuthTuple
from authorization.middleware import authorize
from client import AsyncLlamaStackClientHolder
from configuration import configuration
import metrics
from models.config import Action
from models.requests import QueryRequest
from models.database.conversations import UserConversation
from utils.endpoints import (
    check_configuration_loaded, 
    get_system_prompt,
    get_rag_tools,
    get_mcp_tools,
)
from utils.mcp_headers import mcp_headers_dependency
from utils.transcripts import store_transcript
from utils.types import TurnSummary
from utils.endpoints import validate_model_provider_override

from app.endpoints.query import (
    is_input_shield,
    is_output_shield,
    is_transcripts_enabled,
    select_model_and_provider_id,
    validate_attachments_metadata,
    validate_conversation_ownership,
    persist_user_conversation_details,
    evaluate_model_hints,
)

logger = logging.getLogger("app.endpoints.handlers")
router = APIRouter(tags=["streaming_query"])
auth_dependency = get_auth_dependency()


METADATA_PATTERN = re.compile(r"\nMetadata: (\{.+})\n")


def format_stream_data(d: dict) -> str:
    """
    Format a dictionary as a Server-Sent Events (SSE) data string.

    Parameters:
        d (dict): The data to be formatted as an SSE event.

    Returns:
        str: The formatted SSE data string.
    """
    data = json.dumps(d)
    return f"data: {data}\n\n"


def stream_start_event(conversation_id: str) -> str:
    """
    Yield the start of the data stream.

    Format a Server-Sent Events (SSE) start event containing the
    conversation ID.

    Parameters:
        conversation_id (str): Unique identifier for the
        conversation.

    Returns:
        str: SSE-formatted string representing the start event.
    """
    return format_stream_data(
        {
            "event": "start",
            "data": {
                "conversation_id": conversation_id,
            },
        }
    )


def stream_end_event(metadata_map: dict) -> str:
    """
    Yield the end of the data stream.

    Format and return the end event for a streaming response,
    including referenced document metadata and placeholder token
    counts.

    Parameters:
        metadata_map (dict): A mapping containing metadata about
        referenced documents.

    Returns:
        str: A Server-Sent Events (SSE) formatted string
        representing the end of the data stream.
    """
    return format_stream_data(
        {
            "event": "end",
            "data": {
                "referenced_documents": [
                    {
                        "doc_url": v["docs_url"],
                        "doc_title": v["title"],
                    }
                    for v in filter(
                        lambda v: ("docs_url" in v) and ("title" in v),
                        metadata_map.values(),
                    )
                ],
                "truncated": None,  # TODO(jboos): implement truncated
                "input_tokens": 0,  # TODO(jboos): implement input tokens
                "output_tokens": 0,  # TODO(jboos): implement output tokens
            },
            "available_quotas": {},  # TODO(jboos): implement available quotas
        }
    )


def stream_build_event(chunk: Any, chunk_id: int, metadata_map: dict) -> Iterator[str]:
    """Build a streaming event from a chunk response.

    This function processes chunks from the Llama Stack streaming response and
    formats them into Server-Sent Events (SSE) format for the client. It
    dispatches on (event_type, step_type):

    1. turn_start, turn_awaiting_input -> start token
    2. turn_complete -> final output message
    3. step_* with step_type in {"shield_call", "inference", "tool_execution"} -> delegated handlers
    4. anything else -> heartbeat

    Args:
        chunk: The streaming chunk from Llama Stack containing event data
        chunk_id: The current chunk ID counter (gets incremented for each token)

    Returns:
        Iterator[str]: An iterable list of formatted SSE data strings with event information
    """
    if hasattr(chunk, "error"):
        yield from _handle_error_event(chunk, chunk_id)

    event_type = chunk.event.payload.event_type
    step_type = getattr(chunk.event.payload, "step_type", None)

    match (event_type, step_type):
        case (("turn_start" | "turn_awaiting_input"), _):
            yield from _handle_turn_start_event(chunk_id)
        case ("turn_complete", _):
            yield from _handle_turn_complete_event(chunk, chunk_id)
        case (_, "shield_call"):
            yield from _handle_shield_event(chunk, chunk_id)
        case (_, "inference"):
            yield from _handle_inference_event(chunk, chunk_id)
        case (_, "tool_execution"):
            yield from _handle_tool_execution_event(chunk, chunk_id, metadata_map)
        case _:
            logger.debug(
                "Unhandled event combo: event_type=%s, step_type=%s",
                event_type,
                step_type,
            )
            yield from _handle_heartbeat_event(chunk_id)


# -----------------------------------
# Error handling
# -----------------------------------
def _handle_error_event(chunk: Any, chunk_id: int) -> Iterator[str]:
    """
    Yield error event.

    Yield a formatted Server-Sent Events (SSE) error event
    containing the error message from a streaming chunk.

    Parameters:
        chunk_id (int): The unique identifier for the current
        streaming chunk.
    """
    yield format_stream_data(
        {
            "event": "error",
            "data": {
                "id": chunk_id,
                "token": chunk.error["message"],
            },
        }
    )


# -----------------------------------
# Turn handling
# -----------------------------------
def _handle_turn_start_event(chunk_id: int) -> Iterator[str]:
    """
    Yield turn start event.

    Yield a Server-Sent Event (SSE) token event indicating the
    start of a new conversation turn.

    Parameters:
        chunk_id (int): The unique identifier for the current
        chunk.

    Yields:
        str: SSE-formatted token event with an empty token to
        signal turn start.
    """
    yield format_stream_data(
        {
            "event": "token",
            "data": {
                "id": chunk_id,
                "token": "",
            },
        }
    )


def _handle_turn_complete_event(chunk: Any, chunk_id: int) -> Iterator[str]:
    """
    Yield turn complete event.

    Yields a Server-Sent Event (SSE) indicating the completion of a
    conversation turn, including the full output message content.

    Parameters:
        chunk_id (int): The unique identifier for the current
        chunk.

    Yields:
        str: SSE-formatted string containing the turn completion
        event and output message content.
    """
    yield format_stream_data(
        {
            "event": "turn_complete",
            "data": {
                "id": chunk_id,
                "token": interleaved_content_as_str(
                    chunk.event.payload.turn.output_message.content
                ),
            },
        }
    )


# -----------------------------------
# Shield handling
# -----------------------------------
def _handle_shield_event(chunk: Any, chunk_id: int) -> Iterator[str]:
    """
    Yield shield event.

    Processes a shield event chunk and yields a formatted SSE token
    event indicating shield validation results.

    Yields a "No Violation" token if no violation is detected, or a
    violation message if a shield violation occurs. Increments
    validation error metrics when violations are present.
    """
    if chunk.event.payload.event_type == "step_complete":
        violation = chunk.event.payload.step_details.violation
        if not violation:
            yield format_stream_data(
                {
                    "event": "token",
                    "data": {
                        "id": chunk_id,
                        "role": chunk.event.payload.step_type,
                        "token": "No Violation",
                    },
                }
            )
        else:
            # Metric for LLM validation errors
            metrics.llm_calls_validation_errors_total.inc()
            violation = (
                f"Violation: {violation.user_message} (Metadata: {violation.metadata})"
            )
            yield format_stream_data(
                {
                    "event": "token",
                    "data": {
                        "id": chunk_id,
                        "role": chunk.event.payload.step_type,
                        "token": violation,
                    },
                }
            )


# -----------------------------------
# Inference handling
# -----------------------------------
def _handle_inference_event(chunk: Any, chunk_id: int) -> Iterator[str]:
    """
    Yield inference step event.

    Yield formatted Server-Sent Events (SSE) strings for inference
    step events during streaming.

    Processes inference-related streaming chunks, yielding SSE
    events for step start, text token deltas, and tool call deltas.
    Supports both string and ToolCall object tool calls.
    """
    if chunk.event.payload.event_type == "step_start":
        yield format_stream_data(
            {
                "event": "token",
                "data": {
                    "id": chunk_id,
                    "role": chunk.event.payload.step_type,
                    "token": "",
                },
            }
        )

    elif chunk.event.payload.event_type == "step_progress":
        if chunk.event.payload.delta.type == "tool_call":
            if isinstance(chunk.event.payload.delta.tool_call, str):
                yield format_stream_data(
                    {
                        "event": "tool_call",
                        "data": {
                            "id": chunk_id,
                            "role": chunk.event.payload.step_type,
                            "token": chunk.event.payload.delta.tool_call,
                        },
                    }
                )
            elif isinstance(chunk.event.payload.delta.tool_call, ToolCall):
                yield format_stream_data(
                    {
                        "event": "tool_call",
                        "data": {
                            "id": chunk_id,
                            "role": chunk.event.payload.step_type,
                            "token": chunk.event.payload.delta.tool_call.tool_name,
                        },
                    }
                )

        elif chunk.event.payload.delta.type == "text":
            yield format_stream_data(
                {
                    "event": "token",
                    "data": {
                        "id": chunk_id,
                        "role": chunk.event.payload.step_type,
                        "token": chunk.event.payload.delta.text,
                    },
                }
            )


# -----------------------------------
# Tool Execution handling
# -----------------------------------
# pylint: disable=R1702,R0912
def _handle_tool_execution_event(
    chunk: Any, chunk_id: int, metadata_map: dict
) -> Iterator[str]:
    """
    Yield tool call event.

    Processes tool execution events from a streaming chunk and
    yields formatted Server-Sent Events (SSE) strings.

    Handles both tool call initiation and completion, including
    tool call arguments, responses, and summaries. Extracts and
    updates document metadata from knowledge search tool responses
    when present.

    Parameters:
        chunk_id (int): Unique identifier for the current streaming
        chunk.  metadata_map (dict): Dictionary to be updated with
        document metadata extracted from tool responses.

    Yields:
        str: SSE-formatted event strings representing tool call
        events and responses.
    """
    if chunk.event.payload.event_type == "step_start":
        logger.info("MCP DEBUGGING: Tool execution step started - chunk %d", chunk_id)
        yield format_stream_data(
            {
                "event": "tool_call",
                "data": {
                    "id": chunk_id,
                    "role": chunk.event.payload.step_type,
                    "token": "",
                },
            }
        )

    elif chunk.event.payload.event_type == "step_complete":
        logger.info("MCP DEBUGGING: Tool execution step complete - %d tool calls, %d responses", 
                   len(chunk.event.payload.step_details.tool_calls),
                   len(chunk.event.payload.step_details.tool_responses))
        
        for t in chunk.event.payload.step_details.tool_calls:
            logger.info("MCP DEBUGGING: Tool call - Name: %s, Args: %s", 
                       t.tool_name, str(t.arguments)[:100])
            yield format_stream_data(
                {
                    "event": "tool_call",
                    "data": {
                        "id": chunk_id,
                        "role": chunk.event.payload.step_type,
                        "token": {
                            "tool_name": t.tool_name,
                            "arguments": t.arguments,
                        },
                    },
                }
            )

        for r in chunk.event.payload.step_details.tool_responses:
            response_text = interleaved_content_as_str(r.content)
            logger.info("MCP DEBUGGING: Tool response - Name: %s, Response: '%s'", 
                       r.tool_name, response_text[:200] + "..." if len(response_text) > 200 else response_text)
            if r.tool_name == "query_from_memory":
                inserted_context = interleaved_content_as_str(r.content)
                yield format_stream_data(
                    {
                        "event": "tool_call",
                        "data": {
                            "id": chunk_id,
                            "role": chunk.event.payload.step_type,
                            "token": {
                                "tool_name": r.tool_name,
                                "response": f"Fetched {len(inserted_context)} bytes from memory",
                            },
                        },
                    }
                )

            elif r.tool_name == "knowledge_search" and r.content:
                summary = ""
                for i, text_content_item in enumerate(r.content):
                    if isinstance(text_content_item, TextContentItem):
                        if i == 0:
                            summary = text_content_item.text
                            newline_pos = summary.find("\n")
                            if newline_pos > 0:
                                summary = summary[:newline_pos]
                        for match in METADATA_PATTERN.findall(text_content_item.text):
                            try:
                                meta = ast.literal_eval(match)
                                if "document_id" in meta:
                                    metadata_map[meta["document_id"]] = meta
                            except Exception:  # pylint: disable=broad-except
                                logger.debug(
                                    "An exception was thrown in processing %s",
                                    match,
                                )

                yield format_stream_data(
                    {
                        "event": "tool_call",
                        "data": {
                            "id": chunk_id,
                            "role": chunk.event.payload.step_type,
                            "token": {
                                "tool_name": r.tool_name,
                                "summary": summary,
                            },
                        },
                    }
                )

            else:
                yield format_stream_data(
                    {
                        "event": "tool_call",
                        "data": {
                            "id": chunk_id,
                            "role": chunk.event.payload.step_type,
                            "token": {
                                "tool_name": r.tool_name,
                                "response": interleaved_content_as_str(r.content),
                            },
                        },
                    }
                )


# -----------------------------------
# Catch-all for everything else
# -----------------------------------
def _handle_heartbeat_event(chunk_id: int) -> Iterator[str]:
    """
    Yield a heartbeat event.

    Yield a heartbeat event as a Server-Sent Event (SSE) for the
    given chunk ID.

    Parameters:
        chunk_id (int): The identifier for the current streaming
        chunk.

    Yields:
        str: SSE-formatted heartbeat event string.
    """
    yield format_stream_data(
        {
            "event": "heartbeat",
            "data": {
                "id": chunk_id,
                "token": "heartbeat",
            },
        }
    )


@router.post("/streaming_query")
@authorize(Action.STREAMING_QUERY)
async def streaming_query_endpoint_handler(  # pylint: disable=too-many-locals
    request: Request,
    query_request: QueryRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> StreamingResponse:
    """
    Handle request to the /streaming_query endpoint.

    This endpoint receives a query request, authenticates the user,
    selects the appropriate model and provider, and streams
    incremental response events from the Llama Stack backend to the
    client. Events include start, token updates, tool calls, turn
    completions, errors, and end-of-stream metadata. Optionally
    stores the conversation transcript if enabled in configuration.

    Returns:
        StreamingResponse: An HTTP streaming response yielding
        SSE-formatted events for the query lifecycle.

    Raises:
        HTTPException: Returns HTTP 500 if unable to connect to the
        Llama Stack server.
    """
    # Nothing interesting in the request
    _ = request

    check_configuration_loaded(configuration)

    # Enforce RBAC: optionally disallow overriding model/provider in requests
    validate_model_provider_override(query_request, request.state.authorized_actions)

    # log Llama Stack configuration
    logger.info("Llama stack config: %s", configuration.llama_stack_configuration)

    user_id, _user_name, token = auth

    user_conversation: UserConversation | None = None
    if query_request.conversation_id:
        user_conversation = validate_conversation_ownership(
            user_id=user_id, conversation_id=query_request.conversation_id
        )

        if user_conversation is None:
            logger.warning(
                "User %s attempted to query conversation %s they don't own",
                user_id,
                query_request.conversation_id,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "response": "Access denied",
                    "cause": "You do not have permission to access this conversation",
                },
            )

    try:
        # try to get Llama Stack client
        client = AsyncLlamaStackClientHolder().get_client()
        llama_stack_model_id, model_id, provider_id = select_model_and_provider_id(
            await client.models.list(),
            *evaluate_model_hints(
                user_conversation=user_conversation, query_request=query_request
            ),
        )
        response, conversation_id = await retrieve_response(
            client,
            llama_stack_model_id,
            query_request,
            token,
            mcp_headers=mcp_headers,
        )
        metadata_map: dict[str, dict[str, Any]] = {}

        async def response_generator(
            turn_response: AsyncIterator[OpenAIResponseObjectStream],
        ) -> AsyncIterator[str]:
            """
            Generate SSE formatted streaming response.

            Asynchronously generates a stream of Server-Sent Events
            (SSE) representing incremental responses from a
            language model turn.

            Yields start, token, tool call, turn completion, and
            end events as SSE-formatted strings. Collects the
            complete response for transcript storage if enabled.
            """
            chunk_id = 0
            summary = TurnSummary(
                llm_response="No response from the model", tool_calls=[]
            )

            # Send start event - use the NEW response ID from the response object
            yield stream_start_event(conversation_id)
            
            logger.info("MCP DEBUGGING: Starting streaming response processing")

            async for chunk in turn_response:
                p = chunk.event.payload
                
                logger.info("MCP DEBUGGING: Processing chunk %d, event_type: %s, step_type: %s", 
                           chunk_id, p.event_type, getattr(p, "step_type", "N/A"))
                
                if p.event_type == "turn_complete":
                    summary.llm_response = interleaved_content_as_str(
                        p.turn.output_message.content
                    )
                    logger.info("MCP DEBUGGING: Turn complete, final response: '%s'", 
                               summary.llm_response[:200] + "..." if len(summary.llm_response) > 200 else summary.llm_response)
                    # Debug the output message structure for MCP tools
                    logger.info("MCP DEBUGGING: Turn output message type: %s", type(p.turn.output_message).__name__)
                    if hasattr(p.turn.output_message, 'content'):
                        logger.info("MCP DEBUGGING: Output message has content: %s", type(p.turn.output_message.content))
                    else:
                        logger.info("MCP DEBUGGING: Output message does NOT have content attribute!")
                        logger.info("MCP DEBUGGING: Output message attributes: %s", dir(p.turn.output_message))
                elif p.event_type == "step_complete":
                    if p.step_details.step_type == "tool_execution":
                        logger.info("MCP DEBUGGING: Tool execution completed - %s", 
                                   getattr(p.step_details, 'tool_call', {}).get('function', {}).get('name', 'unknown'))
                        summary.append_tool_calls_from_llama(p.step_details)

                for event in stream_build_event(chunk, chunk_id, metadata_map):
                    chunk_id += 1
                    yield event
            
            logger.info("MCP DEBUGGING: Streaming complete - Tool calls: %d, Response chars: %d", 
                       len(summary.tool_calls), len(summary.llm_response))

            yield stream_end_event(metadata_map)

            if not is_transcripts_enabled():
                logger.debug("Transcript collection is disabled in the configuration")
            else:
                store_transcript(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                    provider_id=provider_id,
                    query_is_valid=True,  # TODO(lucasagomes): implement as part of query validation
                    query=query_request.query,
                    query_request=query_request,
                    summary=summary,
                    rag_chunks=[],  # TODO(lucasagomes): implement rag_chunks
                    truncated=False,  # TODO(lucasagomes): implement truncation as part
                    # of quota work
                    attachments=query_request.attachments or [],
                )

        persist_user_conversation_details(
            user_id=user_id,
            conversation_id=conversation_id,
            model=model_id,
            provider_id=provider_id,
        )

        # Update metrics for the LLM call
        metrics.llm_calls_total.labels(provider_id, model_id).inc()

        return StreamingResponse(response_generator(response))
    # connection to Llama Stack server
    except APIConnectionError as e:
        # Update metrics for the LLM call failure
        metrics.llm_calls_failures_total.inc()
        logger.error("Unable to connect to Llama Stack: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Unable to connect to Llama Stack",
                "cause": str(e),
            },
        ) from e


async def retrieve_response(
    client: AsyncLlamaStackClient,
    model_id: str,
    query_request: QueryRequest,
    token: str,
    mcp_headers: dict[str, dict[str, str]] | None = None,
) -> tuple[AsyncIterator[OpenAIResponseObjectStream], str]:
    """
    Retrieve response from LLMs and agents.

    Asynchronously retrieves a streaming response and conversation
    ID from the Llama Stack agent for a given user query.

    This function configures input/output shields, system prompt,
    and tool usage based on the request and environment. It
    prepares the agent with appropriate headers and toolgroups,
    validates attachments if present, and initiates a streaming
    turn with the user's query and any provided documents.

    Parameters:
        model_id (str): Identifier of the model to use for the query.
        query_request (QueryRequest): The user's query and associated metadata.
        token (str): Authentication token for downstream services.
        mcp_headers (dict[str, dict[str, str]], optional):
        Multi-cluster proxy headers for tool integrations.

    Returns:
        tuple: A tuple containing the streaming response object
        and the conversation ID.
    """
    available_input_shields = [
        shield.identifier
        for shield in filter(is_input_shield, await client.shields.list())
    ]
    available_output_shields = [
        shield.identifier
        for shield in filter(is_output_shield, await client.shields.list())
    ]
    if not available_input_shields and not available_output_shields:
        logger.info("No available shields. Disabling safety")
    else:
        logger.info(
            "Available input shields: %s, output shields: %s",
            available_input_shields,
            available_output_shields,
        )
    # Prepare tools for responses API
    tools = []
    has_mcp_tools = False
    if not query_request.no_tools:
        # Get vector databases for RAG tools
        vector_db_ids = [
            vector_db.identifier for vector_db in await client.vector_dbs.list()
        ]
        
        # Add RAG tools if vector databases are available
        rag_tools = get_rag_tools(vector_db_ids)
        if rag_tools:
            tools.extend(rag_tools)
        
        # Add MCP server tools
        mcp_tools = get_mcp_tools(configuration.mcp_servers, token)
        if mcp_tools:
            tools.extend(mcp_tools)
            has_mcp_tools = True
            logger.info("MCP DEBUGGING: Configured %d MCP tools: %s", 
                       len(mcp_tools), [tool.get("server_label", "unknown") for tool in mcp_tools])

    # use system prompt from request or default one, enhanced for tool usage
    system_prompt = get_system_prompt(query_request, configuration, has_mcp_tools)
    logger.debug("Using system prompt: %s", system_prompt)

    # TODO(lucasagomes): redact attachments content before sending to LLM
    # if attachments are provided, validate them
    if query_request.attachments:
        validate_attachments_metadata(query_request.attachments)
    
    # Create streaming OpenAI response using responses API
    # Try to use conversation chaining with tools - fallback if it fails
    logger.info("MCP DEBUGGING: Attempting conversation chaining - conversation_id: %s, tools: %d",
               query_request.conversation_id, len(tools) if tools else 0)
    
    logger.info("MCP DEBUGGING: Creating streaming response with query: '%s' and %d tools", 
               query_request.query[:100] + "..." if len(query_request.query) > 100 else query_request.query,
               len(tools) if tools else 0)
    
    # Create streaming response - always try to use conversation chaining
    logger.info("MCP DEBUGGING: Creating response with conversation_id=%s, tools=%d", 
                query_request.conversation_id, len(tools) if tools else 0)
    
    response = await client.responses.create(
        input=query_request.query,
        model=model_id,
        instructions=system_prompt,
        previous_response_id=query_request.conversation_id,
        tools=tools if tools else None,
        stream=True,
        store=True,
    )
    response = cast(AsyncIterator[OpenAIResponseObjectStream], response)
    
    logger.info("MCP DEBUGGING: Started streaming response, will extract ID from first chunk")
    
    # The response object has the ID directly, just like non-streaming
    # This is the NEW response ID to use for the NEXT interaction
    conversation_id = response.id if hasattr(response, 'id') else ""
    
    logger.info("MCP DEBUGGING: Created streaming response with ID: %s", conversation_id)

    return response, conversation_id
