"""Handler for REST API call to provide answer to query using Response API."""

import logging
from typing import Annotated

from llama_stack_client import APIConnectionError
from llama_stack_client import AsyncLlamaStackClient  # type: ignore

from fastapi import APIRouter, Request, Depends

from auth import get_auth_dependency
from auth.interface import AuthTuple
from client import AsyncLlamaStackClientHolder
from configuration import configuration
from app.database import get_session
import metrics
from authorization.middleware import authorize
from models.config import Action
from models.database.conversations import UserConversation
from models.requests import QueryRequest
from models.responses import QueryResponse
from utils.endpoints import (
    get_system_prompt,
)
from utils.mcp_headers import mcp_headers_dependency
from utils.types import TurnSummary
from utils.query import (
    query_response,
    evaluate_model_hints,
    select_model_and_provider_id,
    validate_attachments_metadata,
    validate_query_request,
    handle_api_connection_error,
    process_transcript_and_persist_conversation,
)

logger = logging.getLogger("app.endpoints.handlers")
router = APIRouter(tags=["query_v2"])
auth_dependency = get_auth_dependency()


async def get_last_response_id(conversation_id: str) -> str | None:
    """Get the last response ID for a conversation from database or storage."""
    logger.debug("Attempting to get last response ID for conversation: %s", conversation_id)
    
    with get_session() as session:
        try:
            conversation = session.query(UserConversation).filter_by(id=conversation_id).first()
            if conversation:
                logger.debug("Found conversation record for ID: %s", conversation_id)
                if hasattr(conversation, 'last_response_id'):
                    response_id = conversation.last_response_id
                    logger.debug("Retrieved last response ID: %s", response_id)
                    return response_id
                else:
                    logger.warning("UserConversation model missing last_response_id field")
                    return None
            else:
                logger.debug("No conversation record found for ID: %s", conversation_id)
                return None
        except Exception as e:
            logger.error("Error retrieving last response ID for conversation %s: %s", conversation_id, e)
            return None


async def store_response_id(conversation_id: str, response_id: str) -> None:
    """Store the response ID for a conversation."""
    logger.info("STORE_RESPONSE_ID called")
    logger.info("  conversation_id: %s", conversation_id)
    logger.info("  response_id: %s", response_id)
    
    # Validate inputs
    if not response_id:
        logger.error("Cannot store empty response_id for conversation %s", conversation_id)
        return
        
    # Store the response ID for conversation chaining
    with get_session() as session:
        try:
            conversation = session.query(UserConversation).filter_by(id=conversation_id).first()
            if conversation:
                logger.debug("Found conversation record for storing response ID")
                
                # Check if field exists
                if hasattr(conversation, 'last_response_id'):
                    # Check current value
                    old_value = conversation.last_response_id
                    logger.debug("Current last_response_id: %s", old_value)
                    
                    # Update value
                    conversation.last_response_id = response_id
                    logger.debug("Updated last_response_id to: %s", response_id)
                    
                    # Commit transaction
                    session.commit()
                    logger.info("Successfully stored response ID")
                    logger.info("  conversation_id: %s", conversation_id)  
                    logger.info("  response_id: %s", response_id)
                    logger.info("  old_value: %s", old_value)
                    
                    # Verify the storage worked
                    session.refresh(conversation)
                    final_value = conversation.last_response_id
                    logger.info("Verified stored value: %s", final_value)
                    
                else:
                    logger.error("UserConversation model missing last_response_id field - cannot store response ID")
            else:
                logger.warning("No conversation record found for ID %s - cannot store response ID", conversation_id)
        except Exception as e:
            logger.error("Failed to store response ID %s for conversation %s: %s", response_id, conversation_id, e)
            import traceback
            logger.error("Full traceback: %s", traceback.format_exc())


def get_rag_tools(vector_db_ids: list[str]) -> list[dict] | None:
    """Convert vector DB IDs to tools format for responses API."""
    if not vector_db_ids:
        return None

    return [{
        "type": "file_search",
        "vector_store_ids": vector_db_ids,
        "max_num_results": 10
    }]


def get_mcp_tools(mcp_servers: list, token: str | None = None) -> list[dict]:
    """Convert MCP servers to tools format for responses API."""
    tools = []
    for mcp_server in mcp_servers:
        tool_def = {
            "type": "mcp",
            "server_label": mcp_server.name,
            "server_url": mcp_server.url,
            "require_approval": "never"
        }
        
        # Add authentication if token provided (Response API format)
        if token:
            tool_def["headers"] = {
                "Authorization": f"Bearer {token}"
            }
            
        tools.append(tool_def)
    return tools


async def build_tools_for_responses_api(token: str, mcp_headers: dict[str, dict[str, str]] | None = None) -> list[dict] | None:
    """Build tools definition for Response API format."""
    logger.info("BUILDING TOOLS: token provided: %s, mcp_headers provided: %s", 
                bool(token), bool(mcp_headers))
    
    tools = []
    
    # Add RAG tools if vector DBs are available
    try:
        client = AsyncLlamaStackClientHolder().get_client()
        vector_db_ids = [
            vector_db.identifier for vector_db in await client.vector_dbs.list()
        ]
        if vector_db_ids:
            rag_tools = get_rag_tools(vector_db_ids)
            if rag_tools:
                tools.extend(rag_tools)
                logger.info("Added %d RAG tools from vector DBs", len(rag_tools))
    except Exception as e:
        logger.debug("Could not retrieve vector DBs for RAG tools: %s", e)
    
    # CRITICAL FIX: Build MCP headers from configuration if not provided (like Agent API)
    if mcp_headers is None:
        mcp_headers = {}
    
    # If no MCP headers provided, build them from configuration + token (like Agent API does)
    if not mcp_headers and token and hasattr(configuration, 'mcp_servers') and configuration.mcp_servers:
        logger.info("BUILDING MCP HEADERS: No headers provided, building from %d configured servers + token", 
                   len(configuration.mcp_servers))
        for mcp_server in configuration.mcp_servers:
            mcp_headers[mcp_server.url] = {
                "Authorization": f"Bearer {token}",
            }
            logger.debug("Built MCP header for %s: Authorization=Bearer ***", mcp_server.url)
    
    # Add MCP tools from configuration
    if hasattr(configuration, 'mcp_servers') and configuration.mcp_servers:
        logger.info("Found %d MCP servers in configuration", len(configuration.mcp_servers))
        for server in configuration.mcp_servers:
            logger.debug("MCP Server: name=%s, url=%s", getattr(server, 'name', 'N/A'), getattr(server, 'url', 'N/A'))
        mcp_tools = get_mcp_tools(configuration.mcp_servers, token)
        tools.extend(mcp_tools)
        logger.info("Added %d MCP tools from configuration", len(mcp_tools))
    else:
        logger.warning("No MCP servers found in configuration - tools will not be available")
    
    # Add MCP tools from headers ONLY if they are different servers (not from configuration)
    if mcp_headers:
        logger.info("Found MCP headers with %d servers: %s", len(mcp_headers), list(mcp_headers.keys()))
        config_server_urls = set()
        if hasattr(configuration, 'mcp_servers') and configuration.mcp_servers:
            config_server_urls = {server.url for server in configuration.mcp_servers}
        
        custom_mcp_servers = []
        for mcp_url, headers in mcp_headers.items():
            if mcp_url not in config_server_urls:
                logger.debug("MCP Header Server: url=%s, headers=%s (NEW - not in configuration)", mcp_url, headers)
                # Create a mock server object for get_mcp_tools
                mock_server = type('MCPServer', (), {
                    'name': f"custom_mcp_{len(custom_mcp_servers)}",
                    'url': mcp_url
                })()
                custom_mcp_servers.append(mock_server)
            else:
                logger.debug("MCP Header Server: url=%s (SKIPPED - already in configuration)", mcp_url)
        
        if custom_mcp_servers:
            custom_tools = get_mcp_tools(custom_mcp_servers, token)
            tools.extend(custom_tools)
            logger.info("Added %d additional MCP tools from headers", len(custom_tools))
        else:
            logger.debug("No additional MCP servers found in headers (all already in configuration)")
    else:
        logger.debug("No MCP headers provided")
    
    final_tools = tools if tools else None
    logger.info("Built %d tools for Response API: %s", 
                len(tools) if tools else 0, 
                [f"{t.get('type', 'unknown')}:{t.get('server_label', t.get('vector_store_ids', 'N/A'))}" for t in (tools or [])])
    return final_tools


async def retrieve_response_v2(  # pylint: disable=too-many-locals,too-many-branches,too-many-arguments
    client: AsyncLlamaStackClient,
    model_id: str,
    query_request: QueryRequest,
    token: str,
    mcp_headers: dict[str, dict[str, str]] | None = None,
    *,
    provider_id: str = "",
) -> tuple[TurnSummary, str, str | None]:
    """
    Retrieve response from LLMs using Response API.

    Uses the correct client.responses.create() API call with proper
    conversation chaining through response IDs and authentication
    headers for tools.

    Parameters:
        client: The LlamaStack client
        model_id (str): The identifier of the LLM model to use.
        query_request (QueryRequest): The user's query and associated metadata.
        token (str): The authentication token for authorization.
        mcp_headers (dict[str, dict[str, str]], optional): Headers for multi-component processing.
        provider_id (str): The identifier of the LLM provider to use.

    Returns:
        tuple[TurnSummary, str, str | None]: A tuple containing a summary of the LLM response content,
        the conversation ID, and the response ID (for conversation chaining).
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
    
    logger.debug("Using Response API for conversation ID: %s", conversation_id)

    # Initialize response_id for error handling
    response_id = None

    try:
        # Build user input with attachments if present
        user_input = query_request.query
        if query_request.attachments:
            attachment_content = "\n\n".join([
                f"[Attachment: {att.attachment_type}]\n{att.content}"
                for att in query_request.attachments
            ])
            user_input = f"{query_request.query}\n\n{attachment_content}"

        # Handle conversation chaining - get last response ID if continuing conversation
        last_response_id = await get_last_response_id(conversation_id)
        use_chaining = last_response_id is not None
        
        logger.info("Conversation chaining for %s: last_response_id=%s, use_chaining=%s", 
                   conversation_id, last_response_id, use_chaining)
        
        # Build tools for Response API format
        tools = None
        if not query_request.no_tools:
            tools = await build_tools_for_responses_api(token, mcp_headers)
        
        # Build the response request with correct parameters
        response_request = {
            "input": user_input,
            "model": model_id,
            "instructions": system_prompt,
            "previous_response_id": last_response_id if use_chaining else None,
            "tools": tools,
            "stream": False,
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

        logger.info("Making Response API call with model %s for conversation %s (chaining: %s)", 
                   model_id, conversation_id, use_chaining)
        if use_chaining:
            logger.debug("Chaining from previous response ID: %s", last_response_id)
        
        # Log the actual request being sent for debugging
        import json
        sanitized_request = {k: v for k, v in response_request.items()}
        
        # Log tools in detail first
        if 'tools' in sanitized_request and sanitized_request['tools']:
            logger.info("Sending %d tools to Response API:", len(sanitized_request['tools']))
            for i, tool in enumerate(sanitized_request['tools']):
                logger.info("  Tool %d: type=%s, server_label=%s", 
                           i+1, tool.get('type', 'unknown'), tool.get('server_label', 'N/A'))
            sanitized_request['tools'] = f"[{len(sanitized_request['tools'])} tools configured]"
        else:
            logger.warning("No tools being sent to Response API - this may be the problem!")
            
        logger.debug("Response API request parameters: %s", json.dumps(sanitized_request, indent=2, default=str))
        
        # Use correct Response API call
        response = await client.responses.create(**response_request)
        
        # Add comprehensive logging to debug response structure
        logger.info("Response API call completed for conversation %s", conversation_id)
        logger.debug("Response object type: %s", type(response))
        logger.debug("Response object attributes: %s", dir(response))
        
        # Log the response object structure for debugging
        try:
            import json
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
                logger.debug("Response model_dump: %s", json.dumps(response_dict, indent=2, default=str))
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
                logger.debug("Response dict: %s", json.dumps(response_dict, indent=2, default=str))
            else:
                logger.debug("Response object (raw): %s", str(response))
        except Exception as debug_e:
            logger.debug("Could not serialize response object for debugging: %s", debug_e)
        
        # Extract response content and ID with extensive logging
        response_content = ""
        response_id = None
        
        # Try different response formats and log attempts
        logger.debug("Attempting to extract response content...")
        
        def safe_log_content(content, source_name):
            """Safely log content, handling different types of objects."""
            try:
                content_str = str(content)
                if len(content_str) > 100:
                    preview = content_str[:100] + "..."
                else:
                    preview = content_str
                logger.debug("Extracted content from %s: %s", source_name, preview)
                return content_str
            except Exception as e:
                logger.debug("Error extracting content from %s: %s", source_name, e)
                return str(content)
        
        if hasattr(response, 'output') and response.output:
            # Response API format: content is in response.output array
            logger.debug("Found response.output with %d items", len(response.output))
            
            # Debug: log each item in the output
            for i, item in enumerate(response.output):
                item_type = getattr(item, 'type', 'no-type')
                item_role = getattr(item, 'role', 'no-role')
                logger.debug("Output item %d: type=%s, role=%s", i, item_type, item_role)
            
            # Look for the assistant message in the output
            for i, item in enumerate(response.output):
                if (hasattr(item, 'role') and item.role == 'assistant' and 
                    hasattr(item, 'type') and item.type == 'message' and
                    hasattr(item, 'content') and item.content):
                    
                    logger.debug("Found assistant message at output[%d] with %d content items", i, len(item.content))
                    
                    # Extract text from content array
                    for j, content_item in enumerate(item.content):
                        content_type = getattr(content_item, 'type', 'no-type')
                        logger.debug("Content item %d: type=%s", j, content_type)
                        
                        if (hasattr(content_item, 'type') and content_item.type == 'output_text' and
                            hasattr(content_item, 'text')):
                            logger.debug("Extracting text from content item %d", j)
                            logger.debug("Raw text object type: %s", type(content_item.text))
                            logger.debug("Raw text object value: %s", repr(content_item.text))
                            response_content = safe_log_content(content_item.text, "response.output[].content[].text")
                            logger.debug("Successfully extracted response content: length=%d", len(response_content))
                            logger.debug("Final response content type: %s", type(response_content))
                            break
                    
                    if response_content:
                        break
            
            # If no response content found in structured format, fall back to string representation
            if not response_content:
                logger.warning("No response content found in structured format, trying fallback")
                response_content = safe_log_content(response.output, "response.output (fallback)")
        elif hasattr(response, 'choices') and response.choices:
            logger.debug("Found response.choices with %d choices", len(response.choices))
            choice = response.choices[0]
            logger.debug("First choice attributes: %s", dir(choice))
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                response_content = safe_log_content(choice.message.content, "choice.message.content")
        elif hasattr(response, 'content'):
            response_content = safe_log_content(response.content, "response.content")
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            response_content = safe_log_content(response.message.content, "response.message.content")
        elif hasattr(response, 'text'):
            response_content = safe_log_content(response.text, "response.text")
        elif hasattr(response, 'completion'):
            response_content = safe_log_content(response.completion, "response.completion")
        elif hasattr(response, 'output_message') and hasattr(response.output_message, 'content'):
            # LlamaStack Agent API style response
            from llama_stack_client.lib.agents.event_logger import interleaved_content_as_str
            raw_content = interleaved_content_as_str(response.output_message.content)
            response_content = safe_log_content(raw_content, "response.output_message.content (LlamaStack style)")
        elif hasattr(response, 'completion_message') and hasattr(response.completion_message, 'content'):
            # LlamaStack Inference API style response
            from llama_stack_client.lib.inference.event_logger import interleaved_content_as_str
            raw_content = interleaved_content_as_str(response.completion_message.content)
            response_content = safe_log_content(raw_content, "response.completion_message.content")
        else:
            logger.warning("Could not find response content in any expected location")
            # Try to find any string-like attributes
            for attr in dir(response):
                if not attr.startswith('_'):
                    try:
                        value = getattr(response, attr)
                        if isinstance(value, str) and len(value) > 10:
                            safe_log_content(value, f"potential content in {attr}")
                        elif hasattr(value, 'content'):
                            logger.debug("Found object with content in %s", attr)
                            # Try to extract content safely
                            try:
                                content_val = getattr(value, 'content')
                                safe_log_content(content_val, f"nested content in {attr}.content")
                            except Exception:
                                pass
                    except Exception:
                        pass
            response_content = "No response content available"
            
        # Get response ID for chaining with logging
        logger.info("EXTRACTING RESPONSE ID - response type: %s", type(response))
        logger.info("Response object attributes: %s", [attr for attr in dir(response) if not attr.startswith('_')])
        
        # First try to get ID from the main response object
        if hasattr(response, 'id'):
            response_id = response.id
            logger.info("SUCCESS: Found response.id: %s", response_id)
        elif hasattr(response, 'response_id'):
            response_id = response.response_id
            logger.info("SUCCESS: Found response.response_id: %s", response_id)
        # Also try to get ID from the assistant message in output
        elif hasattr(response, 'output') and response.output:
            logger.info("Looking for response ID in output array with %d items", len(response.output))
            for i, item in enumerate(response.output):
                logger.debug("Output item %d: type=%s, role=%s, has_id=%s", 
                           i, getattr(item, 'type', 'N/A'), getattr(item, 'role', 'N/A'), hasattr(item, 'id'))
                if (hasattr(item, 'role') and item.role == 'assistant' and 
                    hasattr(item, 'type') and item.type == 'message' and
                    hasattr(item, 'id')):
                    response_id = item.id
                    logger.info("SUCCESS: Found response ID from assistant message: %s", response_id)
                    break
        elif hasattr(response, 'uuid'):
            response_id = response.uuid
            logger.info("SUCCESS: Found response.uuid: %s", response_id)
        else:
            logger.error("FAILED: No response ID found in any expected locations")
            logger.error("Available attributes: %s", [attr for attr in dir(response) if not attr.startswith('_')])
            
        logger.info("Response extraction completed - Content length: %d, Response ID: %s", 
                   len(response_content), response_id)
        
        # Store response ID for conversation chaining
        if response_id:
            logger.info("CALLING store_response_id")
            logger.info("  conversation_id: %s", conversation_id)
            logger.info("  response_id: %s", response_id)
            await store_response_id(conversation_id, response_id)
            logger.debug("Finished store_response_id call")
        else:
            logger.error("CRITICAL: No response_id extracted - cannot store for conversation chaining!")
            logger.error("This means the second query will lose context!")
        
        # Extract tool calls from Response API format
        tool_calls = []
        if hasattr(response, 'output') and response.output:
            for item in response.output:
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
                    tool_calls.append(tool_call_summary)
                    logger.debug("Found tool call: %s (id=%s)", tool_name, tool_id)
        
        summary = TurnSummary(
            llm_response=response_content,
            tool_calls=tool_calls,
        )
        
        logger.debug("Created summary with response length %d and %d tool calls", 
                    len(response_content), len(tool_calls))

    except Exception as e:
        logger.error("Error during Response API call: %s", e)
        # Fallback response for development/testing
        summary = TurnSummary(
            llm_response=f"Response API call failed: {str(e)}. Please check the API configuration.",
            tool_calls=[],
        )

    return summary, conversation_id, response_id


@router.post("/query", responses=query_response)
@authorize(Action.QUERY)
async def query_endpoint_handler_v2(
    request: Request,
    query_request: QueryRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> QueryResponse:
    """
    Handle request to the /query endpoint using Response API.

    Processes a POST request to the /query endpoint, forwarding the
    user's query to a selected Llama Stack LLM using Response API
    and returning the generated response.

    Validates configuration and authentication, selects the appropriate model
    and provider, retrieves the LLM response, updates metrics, and optionally
    stores a transcript of the interaction. Handles connection errors to the
    Llama Stack service by returning an HTTP 500 error.

    Returns:
        QueryResponse: Contains the conversation ID and the LLM-generated response.
    """
    # Validate request and get user info
    user_id, user_conversation = validate_query_request(request, query_request, auth)
    
    _, _, _, token = auth

    # Initialize response_id for error handling
    response_id = None

    try:
        # try to get Llama Stack client
        client = AsyncLlamaStackClientHolder().get_client()
        llama_stack_model_id, model_id, provider_id = select_model_and_provider_id(
            await client.models.list(),
            *evaluate_model_hints(
                user_conversation=user_conversation, query_request=query_request
            ),
        )
        summary, conversation_id, response_id = await retrieve_response_v2(
            client,
            llama_stack_model_id,
            query_request,
            token,
            mcp_headers=mcp_headers,
            provider_id=provider_id,
        )
        # Update metrics for the LLM call
        metrics.llm_calls_total.labels(provider_id, model_id).inc()

        process_transcript_and_persist_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
            provider_id=provider_id,
            query_request=query_request,
            summary=summary,
        )

        return QueryResponse(
            conversation_id=conversation_id,
            response=summary.llm_response,
            response_id=response_id,  # Include response ID for chaining
        )

    # connection to Llama Stack server
    except APIConnectionError as e:
        handle_api_connection_error(e)
