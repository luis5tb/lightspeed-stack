"""Handler for A2A (Agent-to-Agent) protocol endpoints."""

import logging
import uuid
from typing import Annotated, Any, Dict
from datetime import datetime

from fastapi import APIRouter, Request, Depends, HTTPException, status

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
from app.endpoints.query import query_endpoint_handler
from utils.mcp_headers import mcp_headers_dependency
from version import __version__

logger = logging.getLogger("app.endpoints.a2a")
router = APIRouter(tags=["a2a"])

auth_dependency = get_auth_dependency()


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


@router.get("/.well-known/agent.json", response_model=AgentCard)
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
            enhanced_query = _enhance_query_for_capability(
                skill_id,
                input_content,
                parameters
            )

            # Convert A2A request to internal QueryRequest
            query_request = QueryRequest(
                query=enhanced_query,
                conversation_id=parameters.get("conversation_id"),
                model=parameters.get("model"),
                provider=parameters.get("provider")
            )

            # Execute the query using existing Agents API handler
            query_response = await query_endpoint_handler(
                request=request,
                query_request=query_request,
                auth=auth,
                mcp_headers=mcp_headers
            )

            # Convert response back to A2A format with capability-specific metadata
            # Be robust in case the handler returns a dict instead of a Pydantic model
            qr = (
                query_response.model_dump()  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(query_response, "model_dump")
                else (query_response if isinstance(query_response, dict) else {})
            )
            content = (
                getattr(query_response, "response", None)
                if not isinstance(qr, dict)
                else qr.get("response")
            )
            conversation_id = (
                getattr(query_response, "conversation_id", None)
                if not isinstance(qr, dict)
                else qr.get("conversation_id")
            )
            rag_chunks = (
                getattr(query_response, "rag_chunks", [])
                if not isinstance(qr, dict)
                else qr.get("rag_chunks", [])
            )
            referenced_documents = (
                getattr(query_response, "referenced_documents", [])
                if not isinstance(qr, dict)
                else qr.get("referenced_documents", [])
            )
            tool_calls = (
                getattr(query_response, "tool_calls", None)
                if not isinstance(qr, dict)
                else qr.get("tool_calls")
            )

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
        # Extract text content from message parts
        message_text = " ".join(
            part.text for part in message.parts
            if isinstance(part, TextPart) or (isinstance(part, dict) and part.get("kind") == "text")
        )

        # For simple messages, we'll use a default skill based on content
        # This provides a more conversational, less structured interaction
        enhanced_query = (
            "You are an OpenShift cluster installation expert using the assisted-installer. "
            "Provide helpful, concise responses about OpenShift installation, configuration, "
            "troubleshooting, or requirements analysis. Keep responses focused and practical. "
            f"User message: {message_text}"
        )

        # Add any metadata context
        if message.metadata:
            platform = message.metadata.get("platform")
            if platform:
                enhanced_query += f"\nTarget platform: {platform}"

            cluster_size = message.metadata.get("cluster_size")
            if cluster_size:
                enhanced_query += f"\nCluster size: {cluster_size}"

            openshift_version = message.metadata.get("openshift_version")
            if openshift_version:
                enhanced_query += f"\nOpenShift version: {openshift_version}"

        # Convert to internal QueryRequest
        query_request = QueryRequest(
            query=enhanced_query,
            conversation_id=message.metadata.get("conversation_id") if message.metadata else None,
            model=message.metadata.get("model") if message.metadata else None,
            provider=message.metadata.get("provider") if message.metadata else None
        )

        # Execute the query using existing Agents API handler
        query_response = await query_endpoint_handler(
            request=request,
            query_request=query_request,
            auth=auth,
            mcp_headers=mcp_headers
        )

        # Return simple A2A message response
        # Be robust in case the handler returns a dict instead of a Pydantic model
        qr = (
            query_response.model_dump()  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(query_response, "model_dump")
            else (query_response if isinstance(query_response, dict) else {})
        )
        content = (
            getattr(query_response, "response", None)
            if not isinstance(qr, dict)
            else qr.get("response")
        )
        conversation_id = (
            getattr(query_response, "conversation_id", None)
            if not isinstance(qr, dict)
            else qr.get("conversation_id")
        )
        rag_chunks = (
            getattr(query_response, "rag_chunks", [])
            if not isinstance(qr, dict)
            else qr.get("rag_chunks", [])
        )
        referenced_documents = (
            getattr(query_response, "referenced_documents", [])
            if not isinstance(qr, dict)
            else qr.get("referenced_documents", [])
        )
        tool_calls = (
            getattr(query_response, "tool_calls", None)
            if not isinstance(qr, dict)
            else qr.get("tool_calls")
        )

        # Return official SDK Message response
        return Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            contextId=message.contextId or conversation_id or str(uuid.uuid4()),
            taskId=message.taskId,
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
