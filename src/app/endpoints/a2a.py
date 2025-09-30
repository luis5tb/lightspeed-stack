"""Handler for A2A (Agent-to-Agent) protocol endpoints."""

import logging
import uuid
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Request, Depends

from authentication.interface import AuthTuple
from authentication import get_auth_dependency
from authorization.middleware import authorize
from configuration import configuration
from models.config import Action
from models.a2a import (
    A2AAgentCard,
    A2ACapability,
    A2AProvider,
    A2AAuthentication,
    A2ATaskRequest,
    A2ATaskResponse,
    A2AJsonRpcRequest,
    A2AJsonRpcResponse,
    A2AJsonRpcError,
    A2AMessage,
    A2AVersion,
    AuthScheme,
    ContentType,
    TaskStatus,
)
from models.requests import QueryRequest
from app.endpoints.query_v2 import query_endpoint_handler_v2
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


def get_lightspeed_agent_card() -> A2AAgentCard:
    """Generate the A2A Agent Card for Lightspeed.
    
    Returns:
        A2AAgentCard: The agent card describing Lightspeed's capabilities.
    """
    # Get base URL from configuration or construct it
    service_config = configuration.service_configuration
    base_url = getattr(service_config, 'base_url', 'http://localhost:8080')
    
    # Define Lightspeed's capabilities for OpenShift cluster installation
    capabilities = [
        A2ACapability(
            name="cluster_installation_guidance",
            description="Provide guidance and assistance for OpenShift cluster installation using assisted-installer",
            input_types=[ContentType.TEXT, ContentType.JSON],
            output_types=[ContentType.TEXT, ContentType.JSON],
            streaming=True,
            async_execution=False
        ),
        A2ACapability(
            name="cluster_configuration_validation",
            description="Validate and provide recommendations for OpenShift cluster configuration parameters",
            input_types=[ContentType.JSON, ContentType.TEXT],
            output_types=[ContentType.JSON, ContentType.TEXT],
            streaming=False,
            async_execution=False
        ),
        A2ACapability(
            name="installation_troubleshooting",
            description="Help troubleshoot OpenShift cluster installation issues and provide solutions",
            input_types=[ContentType.TEXT, ContentType.JSON],
            output_types=[ContentType.TEXT, ContentType.JSON],
            streaming=True,
            async_execution=False
        ),
        A2ACapability(
            name="cluster_requirements_analysis",
            description="Analyze infrastructure requirements for OpenShift cluster deployment",
            input_types=[ContentType.JSON, ContentType.TEXT],
            output_types=[ContentType.JSON, ContentType.TEXT],
            streaming=False,
            async_execution=False
        )
    ]
    
    # Provider information
    provider = A2AProvider(
        name="Red Hat",
        url="https://redhat.com",
        contact="lightspeed-support@redhat.com"
    )
    
    # Authentication configuration
    authentication = A2AAuthentication(
        scheme=AuthScheme.BEARER,
        required=True,
        description="Bearer token authentication required for API access"
    )
    
    return A2AAgentCard(
        name="OpenShift Assisted Installer AI Assistant",
        description="AI-powered assistant specialized in OpenShift cluster installation, configuration, and troubleshooting using assisted-installer backend",
        version=__version__,
        a2a_version=A2AVersion.V0_3_0,
        service_url=f"{base_url}/a2a",
        documentation_url=f"{base_url}/docs",
        provider=provider,
        capabilities=capabilities,
        default_input_type=ContentType.TEXT,
        default_output_type=ContentType.TEXT,
        supports_streaming=True,
        supports_push_notifications=False,
        authentication=authentication,
        tags=["openshift", "assisted-installer", "cluster-installation", "ai-assistant", "infrastructure"],
        metadata={
            "service_name": configuration.configuration.name,
            "specialization": "OpenShift cluster installation and management",
            "backend_integration": "assisted-installer",
            "supported_platforms": ["bare-metal", "vsphere", "aws", "azure", "gcp"],
            "installation_methods": ["assisted-installer", "agent-based-installer"],
            "rag_enabled": True,
            "conversation_support": True
        }
    )


@router.get("/.well-known/agent.json", response_model=A2AAgentCard)
@authorize(Action.A2A_AGENT_CARD)
async def get_agent_card() -> A2AAgentCard:
    """
    Serve the A2A Agent Card at the well-known location.
    
    This endpoint provides the agent card that describes Lightspeed's
    capabilities according to the A2A protocol specification.
    
    Returns:
        A2AAgentCard: The agent card describing this agent's capabilities.
    """
    logger.info("Serving A2A Agent Card")
    return get_lightspeed_agent_card()


@router.post("/a2a/task", response_model=A2ATaskResponse)
@authorize(Action.A2A_TASK_EXECUTION)
async def execute_a2a_task(
    request: Request,
    task_request: A2ATaskRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> A2ATaskResponse:
    """
    Execute an A2A task request.
    
    This endpoint handles A2A task execution, mapping A2A requests
    to Lightspeed's internal query processing capabilities.
    
    Args:
        request: The FastAPI request object.
        task_request: The A2A task request.
        auth: Authentication tuple.
        mcp_headers: MCP headers for the request.
        
    Returns:
        A2ATaskResponse: The task execution response.
    """
    logger.info("Executing A2A task: %s", task_request.capability)
    
    # Generate task ID if not provided
    task_id = task_request.task_id or str(uuid.uuid4())
    
    try:
        # Map A2A capability to internal processing for OpenShift installation tasks
        supported_capabilities = [
            "cluster_installation_guidance",
            "cluster_configuration_validation", 
            "installation_troubleshooting",
            "cluster_requirements_analysis"
        ]
        
        if task_request.capability in supported_capabilities:
            # Enhance the query with capability-specific context
            enhanced_query = _enhance_query_for_capability(
                task_request.capability, 
                task_request.input.content,
                task_request.parameters
            )
            
            # Convert A2A request to internal QueryRequest
            query_request = QueryRequest(
                query=enhanced_query,
                conversation_id=task_request.parameters.get("conversation_id"),
                model=task_request.parameters.get("model"),
                provider=task_request.parameters.get("provider")
            )
            
            # Execute the query using existing query handler
            query_response = await query_endpoint_handler_v2(
                request=request,
                query_request=query_request,
                auth=auth,
                mcp_headers=mcp_headers
            )
            
            # Convert response back to A2A format with capability-specific metadata
            output_message = A2AMessage(
                content=query_response.response,
                content_type=ContentType.TEXT,
                metadata={
                    "conversation_id": query_response.conversation_id,
                    "rag_chunks_count": len(query_response.rag_chunks),
                    "referenced_documents_count": len(query_response.referenced_documents),
                    "tool_calls_count": len(query_response.tool_calls) if query_response.tool_calls else 0,
                    "capability_type": task_request.capability,
                    "openshift_context": True
                }
            )
            
            return A2ATaskResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output=output_message,
                metadata={
                    "execution_time": "completed",
                    "capability": task_request.capability,
                    "specialization": "openshift_installation"
                }
            )
            
        else:
            # Unsupported capability
            return A2ATaskResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=f"Unsupported capability: {task_request.capability}",
                metadata={"capability": task_request.capability}
            )
            
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error executing A2A task %s: %s", task_id, str(e))
        return A2ATaskResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=f"Task execution failed: {str(e)}",
            metadata={"capability": task_request.capability}
        )


@router.post("/a2a/message", response_model=A2AMessage)
@authorize(Action.A2A_MESSAGE)
async def handle_a2a_message(
    request: Request,
    message: A2AMessage,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> A2AMessage:
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
        A2AMessage: The response message.
    """
    logger.info("Processing A2A message: %s", message.content_type)
    
    try:
        # For simple messages, we'll use a default capability based on content
        # This provides a more conversational, less structured interaction
        enhanced_query = (
            "You are an OpenShift cluster installation expert using the assisted-installer. "
            "Provide helpful, concise responses about OpenShift installation, configuration, "
            "troubleshooting, or requirements analysis. Keep responses focused and practical. "
            f"User message: {message.content}"
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
        
        # Execute the query using existing query handler
        query_response = await query_endpoint_handler_v2(
            request=request,
            query_request=query_request,
            auth=auth,
            mcp_headers=mcp_headers
        )
        
        # Return simple A2A message response
        return A2AMessage(
            content=query_response.response,
            content_type=ContentType.TEXT,
            metadata={
                "conversation_id": query_response.conversation_id,
                "rag_chunks_count": len(query_response.rag_chunks),
                "referenced_documents_count": len(query_response.referenced_documents),
                "tool_calls_count": len(query_response.tool_calls) if query_response.tool_calls else 0,
                "interaction_type": "simple_message",
                "openshift_context": True
            }
        )
        
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error processing A2A message: %s", str(e))
        
        # Return error as A2A message
        return A2AMessage(
            content=f"Sorry, I encountered an error processing your message: {str(e)}",
            content_type=ContentType.TEXT,
            metadata={
                "error": True,
                "error_type": "processing_error",
                "interaction_type": "simple_message"
            }
        )


@router.post("/a2a/jsonrpc", response_model=A2AJsonRpcResponse)
@authorize(Action.A2A_JSONRPC)
async def handle_a2a_jsonrpc(
    request: Request,
    jsonrpc_request: A2AJsonRpcRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> A2AJsonRpcResponse:
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
        A2AJsonRpcResponse: The JSON-RPC response.
    """
    logger.info("Handling A2A JSON-RPC method: %s", jsonrpc_request.method)
    
    try:
        if jsonrpc_request.method == "execute_task":
            # Extract task request from JSON-RPC params
            task_request = A2ATaskRequest(**jsonrpc_request.params)
            
            # Execute the task
            task_response = await execute_a2a_task(
                request=request,
                task_request=task_request,
                auth=auth,
                mcp_headers=mcp_headers
            )
            
            return A2AJsonRpcResponse(
                id=jsonrpc_request.id,
                result=task_response.model_dump()
            )
            
        elif jsonrpc_request.method == "get_capabilities":
            # Return agent capabilities
            agent_card = get_lightspeed_agent_card()
            
            return A2AJsonRpcResponse(
                id=jsonrpc_request.id,
                result={
                    "capabilities": [cap.model_dump() for cap in agent_card.capabilities],
                    "agent_info": {
                        "name": agent_card.name,
                        "version": agent_card.version,
                        "description": agent_card.description
                    }
                }
            )
            
        elif jsonrpc_request.method == "send_message":
            # Handle simple message via JSON-RPC
            message = A2AMessage(**jsonrpc_request.params)
            
            # Process the message using the message handler
            response_message = await handle_a2a_message(
                request=request,
                message=message,
                auth=auth,
                mcp_headers=mcp_headers
            )
            
            return A2AJsonRpcResponse(
                id=jsonrpc_request.id,
                result=response_message.model_dump()
            )
            
        else:
            # Unsupported method
            error = A2AJsonRpcError(
                code=-32601,
                message="Method not found",
                data={"method": jsonrpc_request.method}
            )
            
            return A2AJsonRpcResponse(
                id=jsonrpc_request.id,
                error=error.model_dump()
            )
            
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error handling JSON-RPC request: %s", str(e))
        
        error = A2AJsonRpcError(
            code=-32603,
            message="Internal error",
            data={"error": str(e)}
        )
        
        return A2AJsonRpcResponse(
            id=jsonrpc_request.id,
            error=error.model_dump()
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
        "a2a_version": A2AVersion.V0_3_0.value,
        "timestamp": "2024-01-01T00:00:00Z"  # This would be actual timestamp in production
    }
