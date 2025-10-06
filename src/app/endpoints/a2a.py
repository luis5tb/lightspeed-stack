"""Handler for A2A (Agent-to-Agent) protocol endpoints."""

import logging
import json
import uuid
from typing import Annotated, Any, Dict
from datetime import datetime

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from authentication.interface import AuthTuple
from authentication import get_auth_dependency
from authorization.middleware import authorize
from configuration import configuration
from models.config import Action
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentProvider,
    AgentCapabilities,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCErrorResponse,
    JSONRPCError,
    Message,
    TextPart,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from models.requests import QueryRequest
from app.endpoints.query import (
    query_endpoint_handler,
    select_model_and_provider_id,
    evaluate_model_hints,
)
from client import AsyncLlamaStackClientHolder
from app.endpoints.streaming_query import (
    retrieve_response as streaming_retrieve_response,
    stream_start_event,
    stream_end_event,
    stream_build_event,
)
from utils.mcp_headers import mcp_headers_dependency
from version import __version__

logger = logging.getLogger("app.endpoints.a2a")
router = APIRouter(tags=["a2a"])

auth_dependency = get_auth_dependency()


# -----------------------------
# Helpers to reduce duplication
# -----------------------------
def _extract_text_from_message_parts(message: Message) -> str:
    """Extract plain text from a Message.parts, supporting dict and model parts."""
    text_chunks: list[str] = []
    for part in getattr(message, "parts", []) or []:
        if isinstance(part, dict):
            if part.get("kind") == "text" and "text" in part:
                text_chunks.append(part.get("text", ""))
        else:
            if getattr(part, "kind", None) == "text":
                txt = getattr(part, "text", "")
                if txt:
                    text_chunks.append(txt)
    return " ".join(text_chunks)


def _build_enhanced_query_from_message(message: Message) -> tuple[str, dict]:
    """Pass-through query (no augmentation) and original metadata.

    We rely on the system prompt for domain guidance; only forward the user's
    message text and metadata needed for routing (conversation_id/model/provider).
    """
    message_text = _extract_text_from_message_parts(message)
    md = message.metadata or {}
    return message_text, md


def _make_query_request_from_enhanced_query(enhanced_query: str, md: dict) -> QueryRequest:
    return QueryRequest(
        query=enhanced_query,
        conversation_id=md.get("conversation_id") if md else None,
        model=md.get("model") if md else None,
        provider=md.get("provider") if md else None,
    )


async def _start_llama_stream(query_request: QueryRequest, token: str, mcp_headers: dict[str, dict[str, str]] | None) -> tuple[Any, str]:
    """Start llama streaming for a given QueryRequest, returning (stream, conversation_id)."""
    client = AsyncLlamaStackClientHolder().get_client()
    llama_stack_model_id, _model_id, _provider_id = select_model_and_provider_id(
        await client.models.list(),
        *evaluate_model_hints(user_conversation=None, query_request=query_request),
    )
    return await streaming_retrieve_response(
        client,
        llama_stack_model_id,
        query_request,
        token,
        mcp_headers=mcp_headers,
    )


async def _aggregate_stream_text(stream: Any) -> str:
    """Aggregate streamed SSE events into a plain text string of tokens."""
    text_chunks: list[str] = []
    metadata_map: dict[str, dict[str, Any]] = {}
    chunk_id = 0
    async for chunk in stream:
        for evt in stream_build_event(chunk, chunk_id, metadata_map):
            chunk_id += 1
            if not evt.startswith("data: "):
                continue
            try:
                payload = json.loads(evt[len("data: "):].strip())
                event = payload.get("event")
                if event in ("token", "turn_complete"):
                    token = payload.get("data", {}).get("token", "")
                    if isinstance(token, str):
                        text_chunks.append(token)
            except Exception:
                # ignore malformed events
                pass
    return "".join(text_chunks)


def _streaming_sse_response(stream: Any, conversation_id: str) -> StreamingResponse:
    async def response_generator() -> Any:
        yield stream_start_event(conversation_id)
        chunk_id = 0
        async for chunk in stream:
            for evt in stream_build_event(chunk, chunk_id, {}):
                chunk_id += 1
                yield evt
        yield stream_end_event({})

    return StreamingResponse(response_generator(), media_type="text/event-stream")

def _enhance_query_for_capability(capability: str, original_query: str, parameters: Dict[str, Any]) -> str:
    """Enhance the user query with capability-specific context for OpenShift installation.

    Args:
        capability: The A2A capability being invoked.
        original_query: The original user query.
        parameters: Additional parameters from the A2A request.

    Returns:
        Enhanced query string with capability-specific context.
    """
    capability_contexts = {
        "cluster_installation_guidance": (
            "You are an OpenShift cluster installation expert using the assisted-installer. "
            "Provide step-by-step guidance for OpenShift cluster installation. "
            "Focus on assisted-installer workflow, prerequisites, and best practices. "
            f"User question: {original_query}"
        ),
        "cluster_configuration_validation": (
            "You are an OpenShift configuration validation expert. "
            "Analyze and validate OpenShift cluster configuration parameters. "
            "Check for common misconfigurations, resource requirements, and compatibility issues. "
            "Provide specific recommendations for improvement. "
            f"Configuration query: {original_query}"
        ),
        "installation_troubleshooting": (
            "You are an OpenShift installation troubleshooting specialist. "
            "Help diagnose and resolve OpenShift cluster installation issues. "
            "Focus on assisted-installer specific problems, common failure points, and solutions. "
            "Provide actionable troubleshooting steps. "
            f"Installation issue: {original_query}"
        ),
        "cluster_requirements_analysis": (
            "You are an OpenShift infrastructure requirements analyst. "
            "Analyze infrastructure requirements for OpenShift cluster deployment. "
            "Consider hardware specs, network requirements, storage needs, and platform-specific considerations. "
            "Provide detailed requirements analysis and recommendations. "
            f"Requirements analysis for: {original_query}"
        )
    }

    base_context = capability_contexts.get(capability, original_query)

    # Add platform-specific context if provided
    platform = parameters.get("platform")
    if platform:
        base_context += f"\n\nTarget platform: {platform}"

    # Add cluster size context if provided
    cluster_size = parameters.get("cluster_size")
    if cluster_size:
        base_context += f"\nCluster size: {cluster_size}"

    # Add version context if provided
    openshift_version = parameters.get("openshift_version")
    if openshift_version:
        base_context += f"\nOpenShift version: {openshift_version}"

    return base_context


def get_lightspeed_agent_card() -> AgentCard:
    """Generate the A2A Agent Card for Lightspeed.

    Returns:
        AgentCard: The agent card describing Lightspeed's capabilities.
    """
    # Get base URL from configuration or construct it
    service_config = configuration.service_configuration
    base_url = getattr(service_config, 'base_url', 'http://localhost:8080')

    # Define Lightspeed's skills for OpenShift cluster installation
    skills = [
        AgentSkill(
            id="cluster_installation_guidance",
            name="Cluster Installation Guidance",
            description="Provide guidance and assistance for OpenShift cluster installation using assisted-installer",
            tags=["openshift", "installation", "assisted-installer"],
            inputModes=["text/plain", "application/json"],
            outputModes=["text/plain", "application/json"],
            examples=[
                "How do I install OpenShift using assisted-installer?",
                "What are the prerequisites for OpenShift installation?"
            ]
        ),
        AgentSkill(
            id="cluster_configuration_validation",
            name="Cluster Configuration Validation",
            description="Validate and provide recommendations for OpenShift cluster configuration parameters",
            tags=["openshift", "configuration", "validation"],
            inputModes=["application/json", "text/plain"],
            outputModes=["application/json", "text/plain"],
            examples=[
                "Validate my cluster configuration",
                "Check if my OpenShift setup meets requirements"
            ]
        ),
        AgentSkill(
            id="installation_troubleshooting",
            name="Installation Troubleshooting",
            description="Help troubleshoot OpenShift cluster installation issues and provide solutions",
            tags=["openshift", "troubleshooting", "support"],
            inputModes=["text/plain", "application/json"],
            outputModes=["text/plain", "application/json"],
            examples=[
                "My cluster installation is failing",
                "How do I fix installation errors?"
            ]
        ),
        AgentSkill(
            id="cluster_requirements_analysis",
            name="Cluster Requirements Analysis",
            description="Analyze infrastructure requirements for OpenShift cluster deployment",
            tags=["openshift", "requirements", "planning"],
            inputModes=["application/json", "text/plain"],
            outputModes=["application/json", "text/plain"],
            examples=[
                "What hardware do I need for OpenShift?",
                "Analyze requirements for a 5-node cluster"
            ]
        )
    ]

    # Provider information
    provider = AgentProvider(
        organization="Red Hat",
        url="https://redhat.com"
    )

    # Agent capabilities
    capabilities = AgentCapabilities(
        streaming=True,
        pushNotifications=False,
        stateTransitionHistory=False
    )

    return AgentCard(
        name="OpenShift Assisted Installer AI Assistant",
        description="AI-powered assistant specialized in OpenShift cluster installation, configuration, and troubleshooting using assisted-installer backend",
        version=__version__,
        url=f"{base_url}/a2a",
        documentationUrl=f"{base_url}/docs",
        provider=provider,
        skills=skills,
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        capabilities=capabilities,
        protocolVersion="0.2.1",
    )


@router.get("/.well-known/agent-card.json", response_model=AgentCard)
@authorize(Action.A2A_AGENT_CARD)
async def get_agent_card(
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
) -> AgentCard:
    """
    Serve the A2A Agent Card at the well-known location.

    This endpoint provides the agent card that describes Lightspeed's
    capabilities according to the A2A protocol specification.

    Returns:
        AgentCard: The agent card describing this agent's capabilities.
    """
    try:
        logger.info("Serving A2A Agent Card")
        return get_lightspeed_agent_card()
    except Exception as e:
        logger.error("Error serving A2A Agent Card: %s", str(e))
        raise


@router.post("/a2a/task", response_model=Task)
@authorize(Action.A2A_TASK_EXECUTION)
async def execute_a2a_task(
    request: Request,
    task_request: Dict[str, Any],
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> Task:
    """
    Execute an A2A task request (custom endpoint, not JSON-RPC).

    This endpoint handles A2A task execution, mapping A2A requests
    to Lightspeed's internal query processing capabilities.

    Args:
        request: The FastAPI request object.
        task_request: Dictionary with skill, input message, and parameters.
        auth: Authentication tuple.
        mcp_headers: MCP headers for the request.

    Returns:
        Task: The official A2A SDK Task object.
    """
    # Extract request parameters
    skill_id = task_request.get("skill", task_request.get("capability", ""))
    input_content = task_request.get("input", {}).get("content", "")
    parameters = task_request.get("parameters", {})
    task_id = task_request.get("task_id", str(uuid.uuid4()))
    context_id = parameters.get("conversation_id", str(uuid.uuid4()))

    logger.info("Executing A2A task for skill: %s", skill_id)

    try:
        # Map A2A skill to internal processing for OpenShift installation tasks
        supported_skills = [
            "cluster_installation_guidance",
            "cluster_configuration_validation",
            "installation_troubleshooting",
            "cluster_requirements_analysis"
        ]

        if skill_id in supported_skills:
            # Enhance the query with skill-specific context
            enhanced_query = input_content or ""

            # Convert A2A request to internal QueryRequest and log
            query_request = QueryRequest(
                query=enhanced_query,
                conversation_id=parameters.get("conversation_id"),
                model=parameters.get("model"),
                provider=parameters.get("provider")
            )
            preview = enhanced_query[:200] + ("..." if len(enhanced_query) > 200 else "")
            logger.info(
                "A2A task query: '%s' | skill=%s conversation_id=%s model=%s provider=%s",
                preview,
                skill_id,
                parameters.get("conversation_id"),
                parameters.get("model"),
                parameters.get("provider"),
            )

            # Stream tokens using shared helpers; aggregate for Task history message
            stream, conversation_id = await _start_llama_stream(query_request, auth[3], mcp_headers)
            content = await _aggregate_stream_text(stream)
            rag_chunks: list[Any] = []
            referenced_documents: list[Any] = []
            tool_calls: list[Any] | None = None

            # Create official SDK Message object for the response
            response_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                contextId=conversation_id or context_id,
                taskId=task_id,
                parts=[TextPart(text=content or "", kind="text")],
                metadata={
                    "rag_chunks_count": len(rag_chunks or []),
                    "referenced_documents_count": len(referenced_documents or []),
                    "tool_calls_count": len(tool_calls or []) if tool_calls else 0,
                    "skill_type": skill_id,
                    "openshift_context": True,
                },
            )

            # Create official SDK Task object
            return Task(
                id=task_id,
                contextId=conversation_id or context_id,
                kind="task",
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp=datetime.now().isoformat()
                ),
                history=[response_message],
                metadata={
                    "skill": skill_id,
                    "specialization": "openshift_installation"
                }
            )

        else:
            # Unsupported skill - return failed task
            error_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                contextId=context_id,
                taskId=task_id,
                parts=[TextPart(text=f"Unsupported skill: {skill_id}", kind="text")],
                metadata={"error": True}
            )

            return Task(
                id=task_id,
                contextId=context_id,
                kind="task",
                status=TaskStatus(
                    state=TaskState.failed,
                    message=error_message,
                    timestamp=datetime.now().isoformat()
                ),
                metadata={"skill": skill_id}
            )

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error executing A2A task %s: %s", task_id, str(e))

        error_message = Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            contextId=context_id,
            taskId=task_id,
            parts=[TextPart(text=f"Task execution failed: {str(e)}", kind="text")],
            metadata={"error": True}
        )

        return Task(
            id=task_id,
            contextId=context_id,
            kind="task",
            status=TaskStatus(
                state=TaskState.failed,
                message=error_message,
                timestamp=datetime.now().isoformat()
            ),
            metadata={"skill": skill_id if 'skill_id' in locals() else "unknown"}
        )


@router.post("/a2a/message", response_model=Message)
@authorize(Action.A2A_MESSAGE)
async def handle_a2a_message(
    request: Request,
    message: Message,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> Message:
    """
    Handle simple A2A message interactions.

    This endpoint provides a lightweight alternative to the full task execution
    endpoint for simple, direct message exchanges that don't require task
    management overhead.

    Args:
        request: The FastAPI request object.
        message: The A2A message to process.
        auth: Authentication tuple.
        mcp_headers: MCP headers for the request.

    Returns:
        Message: The official A2A SDK response message.
    """
    logger.info("Processing A2A message with ID: %s", message.message_id)

    try:
        # Use shared helpers: build query, stream, aggregate
        enhanced_query, md = _build_enhanced_query_from_message(message)
        # Log A2A message query preview and routing metadata
        preview = enhanced_query[:200] + ("..." if len(enhanced_query) > 200 else "")
        logger.info(
            "A2A message query: '%s' | conversation_id=%s model=%s provider=%s",
            preview,
            md.get("conversation_id") if md else None,
            md.get("model") if md else None,
            md.get("provider") if md else None,
        )
        query_request = _make_query_request_from_enhanced_query(enhanced_query, md)
        stream, conversation_id = await _start_llama_stream(query_request, auth[3], mcp_headers)
        content = await _aggregate_stream_text(stream)
        rag_chunks: list[Any] = []
        referenced_documents: list[Any] = []
        tool_calls: list[Any] | None = None

        # Return official SDK Message response
        return Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            contextId=message.context_id or conversation_id or str(uuid.uuid4()),
            taskId=message.task_id,
            parts=[TextPart(text=content or "", kind="text")],
            metadata={
                "conversation_id": conversation_id,
                "rag_chunks_count": len(rag_chunks or []),
                "referenced_documents_count": len(referenced_documents or []),
                "tool_calls_count": len(tool_calls or []) if tool_calls else 0,
                "interaction_type": "simple_message",
                "openshift_context": True
            }
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error processing A2A message: %s", str(e))
        # Return proper HTTP error instead of 200 with error payload
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error processing A2A message",
                "cause": str(e),
            },
        ) from e


@router.post("/a2a/message/stream")
@authorize(Action.A2A_MESSAGE)
async def stream_a2a_message(
    request: Request,
    message: Message,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> StreamingResponse:
    """
    Stream A2A message responses via SSE.

    Produces token/tool_call/turn_complete events for the given message input.
    """
    enhanced_query, md = _build_enhanced_query_from_message(message)
    query_request = _make_query_request_from_enhanced_query(enhanced_query, md)
    stream, conversation_id = await _start_llama_stream(query_request, auth[3], mcp_headers)
    return _streaming_sse_response(stream, conversation_id)


@router.post("/a2a/task/stream")
@authorize(Action.A2A_TASK_EXECUTION)
async def stream_a2a_task(
    request: Request,
    task_request: Dict[str, Any],
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> StreamingResponse:
    """
    Stream A2A task execution via SSE.

    Accepts the same payload as /a2a/task (skill, input.content, parameters),
    but streams incremental generation chunks and tool events.
    """
    skill_id = task_request.get("skill", task_request.get("capability", ""))
    input_content = task_request.get("input", {}).get("content", "")
    parameters = task_request.get("parameters", {})

    enhanced_query = _enhance_query_for_capability(skill_id, input_content, parameters)
    query_request = QueryRequest(
        query=enhanced_query,
        conversation_id=parameters.get("conversation_id"),
        model=parameters.get("model"),
        provider=parameters.get("provider"),
    )
    stream, conversation_id = await _start_llama_stream(query_request, auth[3], mcp_headers)
    return _streaming_sse_response(stream, conversation_id)


@router.post("/a2a/jsonrpc", response_model=JSONRPCResponse)
@authorize(Action.A2A_JSONRPC)
async def handle_a2a_jsonrpc(
    request: Request,
    jsonrpc_request: JSONRPCRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> JSONRPCResponse:
    """
    Handle A2A JSON-RPC 2.0 requests.

    This endpoint provides JSON-RPC 2.0 compatibility for A2A communication,
    allowing other agents to interact with Lightspeed using the standard
    JSON-RPC protocol.

    Args:
        request: The FastAPI request object.
        jsonrpc_request: The JSON-RPC request.
        auth: Authentication tuple.
        mcp_headers: MCP headers for the request.

    Returns:
        JSONRPCResponse: The JSON-RPC response.
    """
    logger.info("Handling A2A JSON-RPC method: %s", jsonrpc_request.method)

    try:
        if jsonrpc_request.method == "execute_task" or jsonrpc_request.method == "tasks/create":
            # Execute the task - pass params as dict
            task_response = await execute_a2a_task(
                request=request,
                task_request=jsonrpc_request.params or {},
                auth=auth,
                mcp_headers=mcp_headers
            )

            return JSONRPCResponse(
                id=jsonrpc_request.id,
                result=task_response.model_dump()
            )

        elif jsonrpc_request.method == "get_capabilities" or jsonrpc_request.method == "agent/capabilities":
            # Return agent capabilities/skills
            agent_card = get_lightspeed_agent_card()

            return JSONRPCResponse(
                id=jsonrpc_request.id,
                result={
                    "skills": [skill.model_dump() for skill in agent_card.skills],
                    "agent_info": {
                        "name": agent_card.name,
                        "version": agent_card.version,
                        "description": agent_card.description
                    }
                }
            )

        elif jsonrpc_request.method == "send_message" or jsonrpc_request.method == "message/send":
            # Handle message via JSON-RPC - convert params to Message object
            message = Message(**jsonrpc_request.params)

            # Process the message using the message handler
            response_message = await handle_a2a_message(
                request=request,
                message=message,
                auth=auth,
                mcp_headers=mcp_headers
            )

            return JSONRPCResponse(
                id=jsonrpc_request.id,
                result=response_message.model_dump()
            )

        else:
            # Unsupported method - return error response
            error = JSONRPCError(
                code=-32601,
                message="Method not found",
                data={"method": jsonrpc_request.method}
            )

            return JSONRPCErrorResponse(
                id=jsonrpc_request.id,
                error=error
            )

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error handling JSON-RPC request: %s", str(e))

        error = JSONRPCError(
            code=-32603,
            message="Internal error",
            data={"error": str(e)}
        )

        return JSONRPCErrorResponse(
            id=jsonrpc_request.id,
            error=error
        )


@router.get("/a2a/health")
async def a2a_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for A2A service.

    Returns:
        Dict[str, Any]: Health status information.
    """
    return {
        "status": "healthy",
        "service": "lightspeed-a2a",
        "version": __version__,
        "a2a_sdk_version": "0.2.1",
        "timestamp": datetime.now().isoformat()
    }
