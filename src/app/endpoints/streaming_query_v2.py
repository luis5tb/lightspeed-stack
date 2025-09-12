"""Handler for REST API call to provide answer to streaming query using Response API."""

import logging
from typing import Annotated, AsyncIterator

from llama_stack_client import APIConnectionError
from llama_stack_client import AsyncLlamaStackClient  # type: ignore

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
from utils.endpoints import get_system_prompt
from utils.mcp_headers import mcp_headers_dependency
from utils.transcripts import store_transcript
from utils.types import TurnSummary
from utils.streaming_query import (
    format_stream_data,
    stream_start_event,
    stream_end_event,
)
from utils.query import (
    select_model_and_provider_id,
    validate_attachments_metadata,
    validate_query_request,
    evaluate_model_hints,
    is_transcripts_enabled,
    persist_user_conversation_details,
)
from app.endpoints.query_v2 import (
    get_last_response_id,
    store_response_id, 
    build_tools_for_responses_api,
)

logger = logging.getLogger("app.endpoints.handlers")
router = APIRouter(tags=["streaming_query_v2"])
auth_dependency = get_auth_dependency()


def stream_end_event_v2(metadata_map: dict, response_id: str | None = None) -> str:
    """
    Yield the end of the data stream for Response API v2.

    Format and return the end event for a streaming response,
    including referenced document metadata, placeholder token
    counts, and response_id for conversation chaining.

    Parameters:
        metadata_map (dict): A mapping containing metadata about referenced documents.
        response_id (str | None): The response ID for conversation chaining.

    Returns:
        str: A Server-Sent Events (SSE) formatted string representing the end of the data stream.
    """
    data = {
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
    }
    
    # Add response_id for conversation chaining if available
    if response_id:
        data["response_id"] = response_id
    
    return format_stream_data(
        {
            "event": "end",
            "data": data,
            "available_quotas": {},  # TODO(jboos): implement available quotas
        }
    )


async def retrieve_response_v2(
    client: AsyncLlamaStackClient,
    model_id: str,
    query_request: QueryRequest,
    token: str,
    mcp_headers: dict[str, dict[str, str]] | None = None,
) -> tuple[AsyncIterator, str]:
    """
    Retrieve streaming response from LLMs using Response API.

    Uses the correct client.responses.create() API call with stream=True,
    proper conversation chaining through response IDs and authentication
    headers for tools.

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
    # use system prompt from request or default one
    system_prompt = get_system_prompt(query_request, configuration)
    logger.debug("Using system prompt: %s", system_prompt)

    # if attachments are provided, validate them
    if query_request.attachments:
        validate_attachments_metadata(query_request.attachments)

    # Generate conversation ID for Response API
    import uuid
    conversation_id = query_request.conversation_id or str(uuid.uuid4())

    logger.debug("Using Response API for streaming conversation ID: %s", conversation_id)


    # Build user input with attachments if present
    user_input = query_request.query
    if query_request.attachments:
        attachment_content = "\n\n".join([
            f"[Attachment: {att.attachment_type}]\n{att.content}"
            for att in query_request.attachments
        ])
        user_input = f"{query_request.query}\n\n{attachment_content}"

    # Handle conversation chaining - get last response ID if continuing conversation
    logger.info("STREAMING: Getting last response ID for conversation: %s", conversation_id)
    last_response_id = await get_last_response_id(conversation_id)
    use_chaining = last_response_id is not None
    logger.info("STREAMING: Conversation chaining status - last_response_id: %s, use_chaining: %s", last_response_id, use_chaining)
    
    # Build tools for Response API format
    tools = None
    if not query_request.no_tools:
        tools = await build_tools_for_responses_api(token, mcp_headers)
    
    # Build the response request with correct streaming parameters
    response_request = {
        "input": user_input,
        "model": model_id,
        "instructions": system_prompt,
        "previous_response_id": last_response_id if use_chaining else None,
        "tools": tools,
        "stream": True,
        "store": True,
    }
    
    # Add sampling parameters if available
    if hasattr(configuration, 'inference') and configuration.inference:
        if hasattr(configuration.inference, 'temperature'):
            response_request["temperature"] = configuration.inference.temperature
        if hasattr(configuration.inference, 'max_tokens'):
            response_request["max_tokens"] = configuration.inference.max_tokens
        if hasattr(configuration.inference, 'top_p'):
            response_request["top_p"] = configuration.inference.top_p

        logger.info("Making Response API streaming call with model %s for conversation %s (chaining: %s)", 
                   model_id, conversation_id, use_chaining)
        if use_chaining:
            logger.debug("Chaining from previous response ID: %s", last_response_id)
        
        # Log the actual request being sent for debugging
        import json
        sanitized_request = {k: v for k, v in response_request.items()}
        # Don't log potentially large tool definitions in full
        if 'tools' in sanitized_request and sanitized_request['tools']:
            sanitized_request['tools'] = f"[{len(sanitized_request['tools'])} tools configured]"
        logger.debug("Streaming Response API request parameters: %s", json.dumps(sanitized_request, indent=2, default=str))
        
        # Use correct Response API streaming call
        response = await client.responses.create(**response_request)
        logger.info("Successfully initiated Response API streaming for conversation %s", conversation_id)
        logger.debug("Streaming response object type: %s", type(response))
        
        return response, conversation_id


def stream_build_event_v2(chunk, chunk_id: int, metadata_map: dict):
    """Build a streaming event from a Response API chunk."""
    
    # Handle Response API streaming chunks based on OpenAI-style format
    content = ""
    is_complete = False
    
    # Extract content from different possible chunk structures
    if hasattr(chunk, 'choices') and chunk.choices:
        choice = chunk.choices[0]
        if hasattr(choice, 'delta') and choice.delta:
            if hasattr(choice.delta, 'content') and choice.delta.content:
                content = choice.delta.content
        if hasattr(choice, 'finish_reason') and choice.finish_reason:
            is_complete = True
    elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
        content = chunk.delta.content or ""
    elif hasattr(chunk, 'content'):
        content = chunk.content or ""
    elif hasattr(chunk, 'event_type'):
        # Handle mock chunks in fallback scenarios
        if chunk.event_type == "start":
            content = ""
        elif chunk.event_type == "token":
            content = getattr(chunk, 'content', '')
        elif chunk.event_type == "complete":
            content = getattr(chunk, 'content', '')
            is_complete = True
            
    # Determine event type based on content and completion status
    if chunk_id == 0 or content == "":
        # Start of stream
        yield format_stream_data(
            {
                "event": "token",
                "data": {
                    "id": chunk_id,
                    "token": "",
                },
            }
        )
    elif is_complete:
        # End of stream - turn complete
        yield format_stream_data(
            {
                "event": "turn_complete",
                "data": {
                    "id": chunk_id,
                    "token": content,
                },
            }
        )
    elif content:
        # Regular content chunk
        yield format_stream_data(
            {
                "event": "token",
                "data": {
                    "id": chunk_id,
                    "token": content,
                },
            }
        )
    else:
        # Fallback heartbeat
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
async def streaming_query_endpoint_handler_v2(  # pylint: disable=too-many-locals
    request: Request,
    query_request: QueryRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> StreamingResponse:
    """
    Handle request to the /streaming_query endpoint using Response API.

    This endpoint receives a query request, authenticates the user,
    selects the appropriate model and provider, and streams
    incremental response events from the Llama Stack backend using
    Response API to the client. Events include start, token updates,
    turn completions, errors, and end-of-stream metadata.
    Optionally stores the conversation transcript if enabled in configuration.

    Returns:
        StreamingResponse: An HTTP streaming response yielding
        SSE-formatted events for the query lifecycle.

    Raises:
        HTTPException: Returns HTTP 500 if unable to connect to the
        Llama Stack server.
    """
    # Nothing interesting in the request
    _ = request

    # Validate request and get user info
    user_id, user_conversation = validate_query_request(request, query_request, auth)
    
    _, _user_name, _skip_userid_check, token = auth

    try:
        # try to get Llama Stack client
        client = AsyncLlamaStackClientHolder().get_client()
        llama_stack_model_id, model_id, provider_id = select_model_and_provider_id(
            await client.models.list(),
            *evaluate_model_hints(
                user_conversation=user_conversation, query_request=query_request
            ),
        )
        response, conversation_id = await retrieve_response_v2(
            client,
            llama_stack_model_id,
            query_request,
            token,
            mcp_headers=mcp_headers,
        )

        async def response_generator_v2() -> AsyncIterator[str]:
            """
            Generate SSE formatted streaming response for Response API.

            Asynchronously generates a stream of Server-Sent Events
            (SSE) representing incremental responses from Response API.

            Yields start, token, turn completion, and end events as
            SSE-formatted strings. Collects the complete response for
            transcript storage if enabled.
            """
            chunk_id = 0
            summary = TurnSummary(
                llm_response="No response from the model", tool_calls=[]
            )
            metadata_map: dict[str, dict[str, any]] = {}

            # Send start event
            yield stream_start_event(conversation_id)

            full_response = ""
            response_id = None
            tool_calls_found = []  # Collect tool calls during streaming
            
            async for chunk in response:
                # Log chunk details for debugging
                logger.debug("Received streaming chunk %d: type=%s", chunk_id, type(chunk))
                try:
                    if hasattr(chunk, 'model_dump'):
                        chunk_dict = chunk.model_dump()
                        logger.debug("Chunk %d model_dump: %s", chunk_id, str(chunk_dict)[:200] + "..." if len(str(chunk_dict)) > 200 else str(chunk_dict))
                    elif hasattr(chunk, 'dict'):
                        chunk_dict = chunk.dict()
                        logger.debug("Chunk %d dict: %s", chunk_id, str(chunk_dict)[:200] + "..." if len(str(chunk_dict)) > 200 else str(chunk_dict))
                    else:
                        logger.debug("Chunk %d attributes: %s", chunk_id, [attr for attr in dir(chunk) if not attr.startswith('_')][:10])
                except Exception as chunk_debug_e:
                    logger.debug("Could not debug chunk %d: %s", chunk_id, chunk_debug_e)
                
                # Extract content from Response API chunk for summary
                chunk_content = ""
                
                # Handle Response API streaming format - content in output array
                if hasattr(chunk, 'output') and chunk.output:
                    logger.debug("Chunk %d: Found output array with %d items", chunk_id, len(chunk.output))
                    for item in chunk.output:
                        # Extract tool calls
                        if hasattr(item, 'type') and item.type == 'mcp_call':
                            from utils.types import ToolCallSummary
                            import json
                            
                            tool_id = getattr(item, 'id', 'unknown-id')
                            tool_name = getattr(item, 'name', 'unknown_tool')
                            tool_args = getattr(item, 'arguments', '{}')
                            tool_response = getattr(item, 'output', None)
                            
                            # Parse arguments if it's a JSON string
                            try:
                                parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                            except json.JSONDecodeError:
                                parsed_args = tool_args
                            
                            tool_call_summary = ToolCallSummary(
                                id=tool_id,
                                name=tool_name,
                                args=parsed_args,
                                response=tool_response
                            )
                            tool_calls_found.append(tool_call_summary)
                            logger.debug("Chunk %d: Found tool call: %s (id=%s)", chunk_id, tool_name, tool_id)
                        
                        # Extract assistant message content
                        elif (hasattr(item, 'role') and item.role == 'assistant' and 
                            hasattr(item, 'type') and item.type == 'message' and
                            hasattr(item, 'content') and item.content):
                            
                            # Extract text from content array
                            for content_item in item.content:
                                if (hasattr(content_item, 'type') and content_item.type == 'output_text' and
                                    hasattr(content_item, 'text')):
                                    chunk_content = str(content_item.text)
                                    full_response += chunk_content
                                    try:
                                        preview = chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content
                                        logger.debug("Chunk %d: Extracted content from output[].content[].text: %s", chunk_id, preview)
                                    except Exception as e:
                                        logger.debug("Chunk %d: Error getting content preview: %s", chunk_id, e)
                                    break
                            break
                # Handle OpenAI-style response chunks (fallback)
                elif hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta:
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            chunk_content = choice.delta.content
                            full_response += chunk_content
                            logger.debug("Chunk %d: Extracted content from choices.delta.content: %s", chunk_id, chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content)
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content') and chunk.delta.content:
                    chunk_content = chunk.delta.content
                    full_response += chunk_content
                    logger.debug("Chunk %d: Extracted content from delta.content: %s", chunk_id, chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content)
                elif hasattr(chunk, 'content') and chunk.content:
                    chunk_content = chunk.content
                    full_response += chunk_content
                    logger.debug("Chunk %d: Extracted content from content: %s", chunk_id, chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content)
                elif hasattr(chunk, 'completion_message') and hasattr(chunk.completion_message, 'content'):
                    # LlamaStack Inference API style streaming
                    from llama_stack_client.lib.inference.event_logger import interleaved_content_as_str
                    raw_content = interleaved_content_as_str(chunk.completion_message.content)
                    chunk_content = str(raw_content)  # Convert to string if it's a Text object
                    full_response += chunk_content
                    try:
                        preview = chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content
                        logger.debug("Chunk %d: Extracted content from completion_message.content: %s", chunk_id, preview)
                    except Exception as e:
                        logger.debug("Chunk %d: Extracted content from completion_message.content (error getting preview): %s", chunk_id, e)
                elif hasattr(chunk, 'output_message') and hasattr(chunk.output_message, 'content'):
                    # LlamaStack Agent API style streaming
                    from llama_stack_client.lib.agents.event_logger import interleaved_content_as_str
                    raw_content = interleaved_content_as_str(chunk.output_message.content)
                    chunk_content = str(raw_content)  # Convert to string if it's a Text object
                    full_response += chunk_content
                    try:
                        preview = chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content
                        logger.debug("Chunk %d: Extracted content from output_message.content: %s", chunk_id, preview)
                    except Exception as e:
                        logger.debug("Chunk %d: Extracted content from output_message.content (error getting preview): %s", chunk_id, e)
                else:
                    logger.debug("Chunk %d: No content found in expected locations", chunk_id)
                
                # Extract response ID for conversation chaining - enhanced debugging
                if not response_id:  # Only try extraction if we don't have one yet
                    if hasattr(chunk, 'id'):
                        response_id = chunk.id
                        logger.info("STREAMING: Chunk %d: SUCCESS: Found chunk.id: %s", chunk_id, response_id)
                    elif hasattr(chunk, 'response_id'):
                        response_id = chunk.response_id
                        logger.info("STREAMING: Chunk %d: SUCCESS: Found chunk.response_id: %s", chunk_id, response_id)
                    # Also try to get ID from assistant message in output
                    elif hasattr(chunk, 'output') and chunk.output:
                        logger.debug("STREAMING: Chunk %d: Looking for response ID in output array with %d items", chunk_id, len(chunk.output))
                        for i, item in enumerate(chunk.output):
                            if (hasattr(item, 'role') and item.role == 'assistant' and 
                                hasattr(item, 'type') and item.type == 'message' and
                                hasattr(item, 'id')):
                                response_id = item.id
                                logger.info("STREAMING: Chunk %d: SUCCESS: Found response ID from assistant message: %s", chunk_id, response_id)
                                break
                    else:
                        # Only log this once, not for every chunk
                        if chunk_id == 1:
                            logger.info("STREAMING: Chunk %d type: %s", chunk_id, type(chunk))
                            logger.debug("STREAMING: Chunk %d attributes: %s", chunk_id, [attr for attr in dir(chunk) if not attr.startswith('_')])
                
                # Generate streaming events
                for event in stream_build_event_v2(chunk, chunk_id, metadata_map):
                    chunk_id += 1
                    yield event
            
            # Update summary with complete response and tool calls
            if full_response:
                summary.llm_response = full_response
            
            # Update tool calls in summary
            if tool_calls_found:
                summary.tool_calls = tool_calls_found
                logger.debug("Updated streaming summary with %d tool calls", len(tool_calls_found))
                
            # Store response ID for conversation chaining - enhanced debugging
            if response_id:
                logger.info("STREAMING: CALLING store_response_id")
                logger.info("STREAMING:   conversation_id: %s", conversation_id)
                logger.info("STREAMING:   response_id: %s", response_id)
                try:
                    await store_response_id(conversation_id, response_id)
                    logger.debug("STREAMING: Finished store_response_id call")
                except Exception as e:
                    logger.error("STREAMING: Could not store response ID for streaming: %s", e)
                    import traceback
                    logger.error("STREAMING: Full traceback: %s", traceback.format_exc())
            else:
                logger.error("STREAMING: CRITICAL: No response_id extracted from streaming chunks - cannot store for conversation chaining!")
                logger.error("STREAMING: This means the second query will lose context!")

            yield stream_end_event_v2(metadata_map, response_id)

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

        return StreamingResponse(response_generator_v2())
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
