"""Handler for A2A (Agent-to-Agent) protocol endpoints."""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentProvider,
    AgentCapabilities,
    Task,
    TaskState,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from a2a.utils import new_agent_text_message, new_task

from authentication.interface import AuthTuple
from authentication import get_auth_dependency
from authorization.middleware import authorize
from configuration import configuration
from models.config import Action
from models.requests import QueryRequest
from app.endpoints.query import (
    select_model_and_provider_id,
    evaluate_model_hints,
)
from app.endpoints.streaming_query import retrieve_response
from client import AsyncLlamaStackClientHolder
from utils.mcp_headers import mcp_headers_dependency
from version import __version__

logger = logging.getLogger("app.endpoints.a2a")
router = APIRouter(tags=["a2a"])

auth_dependency = get_auth_dependency()


# -----------------------------
# Agent Executor Implementation
# -----------------------------
class LightspeedAgentExecutor(AgentExecutor):
    """
    Lightspeed Agent Executor for OpenShift Assisted Chat Installer.

    This executor implements the A2A AgentExecutor interface and handles
    routing queries to the appropriate LLM backend.
    """

    def __init__(self, auth_token: str, mcp_headers: dict[str, dict[str, str]] | None = None):
        """
        Initialize the Lightspeed agent executor.

        Args:
            auth_token: Authentication token for the request
            mcp_headers: MCP headers for context propagation
        """
        self.auth_token = auth_token
        self.mcp_headers = mcp_headers or {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute the agent with the given context and send results to the event queue.

        Args:
            context: The request context containing user input and metadata
            event_queue: Queue for sending response events
        """
        # Get or create task
        task = await self._prepare_task(context, event_queue)

        # Process the task with streaming
        await self._process_task_streaming(context, event_queue, task.context_id, task.id)

    async def _prepare_task(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> Task:
        """
        Get existing task or create a new one.

        Args:
            context: The request context
            event_queue: Queue for sending events

        Returns:
            Task object
        """
        task = context.current_task
        if not task:
            if not context.message:
                raise Exception("No message provided in context")
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        return task

    async def _process_task_streaming(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        context_id: str,
        task_id: str
    ) -> None:
        """
        Process the task with streaming updates.

        Args:
            context: The request context
            event_queue: Queue for sending events
            context_id: Context ID for the task
            task_id: Task ID
        """
        task_updater = TaskUpdater(event_queue, task_id, context_id)

        try:
            # Extract user input using SDK utility
            user_input = context.get_user_input()
            if not user_input:
                await task_updater.failed(
                    message=new_agent_text_message(
                        "I didn't receive any input. How can I help you with OpenShift installation?",
                        context_id=context_id,
                        task_id=task_id
                    )
                )
                return

            preview = user_input[:200] + ("..." if len(user_input) > 200 else "")
            logger.info("Processing A2A request: %s", preview)

            # Extract routing metadata from context
            metadata = context.message.metadata if context.message else {}
            conversation_id = metadata.get("conversation_id") if metadata else None
            model = metadata.get("model") if metadata else None
            provider = metadata.get("provider") if metadata else None

            # Build internal query request
            query_request = QueryRequest(
                query=user_input,
                conversation_id=conversation_id,
                model=model,
                provider=provider,
            )

            # Get LLM client and select model
            client = AsyncLlamaStackClientHolder().get_client()
            llama_stack_model_id, _model_id, _provider_id = select_model_and_provider_id(
                await client.models.list(),
                *evaluate_model_hints(user_conversation=None, query_request=query_request),
            )

            # Stream response from LLM with status updates
            stream, conversation_id = await retrieve_response(
                client,
                llama_stack_model_id,
                query_request,
                self.auth_token,
                mcp_headers=self.mcp_headers,
            )

            # Process stream and send incremental updates
            text_chunks = []
            is_complete = False
            streamed_any_delta = False

            async for chunk in stream:
                # Extract text from chunk - llama-stack structure
                if hasattr(chunk, 'event') and chunk.event is not None:
                    payload = chunk.event.payload
                    event_type = payload.event_type

                    # Handle turn_complete - final message
                    if event_type == "turn_complete":
                        from llama_stack_client.lib.agents.event_logger import interleaved_content_as_str
                        final_text = interleaved_content_as_str(
                            payload.turn.output_message.content
                        )
                        # If we already streamed deltas, avoid re-appending full content
                        if final_text and not streamed_any_delta:
                            text_chunks.append(final_text)

                        # Mark that we've received the complete turn
                        is_complete = True

                        # Send final message and mark task as complete
                        # If deltas were streamed, avoid duplicating content by sending empty final text
                        await task_updater.complete(
                            message=new_agent_text_message(
                                "".join(text_chunks) if (text_chunks and not streamed_any_delta) else "",
                                context_id=context_id,
                                task_id=task_id
                            )
                        )
                        logger.info("Task %s completed successfully", task_id)

                    # Handle streaming inference tokens
                    elif event_type == "step_progress":
                        if hasattr(payload, 'delta') and payload.delta.type == "text":
                            delta_text = payload.delta.text
                            if delta_text:
                                # Stream token delta only to avoid duplication downstream
                                streamed_any_delta = True
                                # Send incremental update with working status
                                await task_updater.update_status(
                                    TaskState.working,
                                    message=new_agent_text_message(
                                        delta_text,
                                        context_id=context_id,
                                        task_id=task_id
                                    )
                                )

            # Fallback: If we exited the loop without completing, mark as complete anyway
            if not is_complete:
                logger.warning("Stream ended without turn_complete event, marking task as complete")
                # If deltas were streamed, avoid duplicating content by sending empty final text
                final_content = "" if streamed_any_delta else ("".join(text_chunks) if text_chunks else "Task completed")
                await task_updater.complete(
                    message=new_agent_text_message(
                        final_content,
                        context_id=context_id,
                        task_id=task_id
                    )
                )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error executing agent: %s", str(exc), exc_info=True)
            await task_updater.failed(
                message=new_agent_text_message(
                    f"Sorry, I encountered an error: {str(exc)}",
                    context_id=context_id,
                    task_id=task_id
                )
            )

    async def cancel(
        self,
        context: RequestContext,  # pylint: disable=unused-argument
        event_queue: EventQueue,  # pylint: disable=unused-argument
    ) -> None:
        """
        Handle task cancellation.

        Args:
            context: The request context
            event_queue: Queue for sending cancellation events

        Raises:
            NotImplementedError: Task cancellation is not currently supported
        """
        logger.info("Cancellation requested but not currently supported")
        raise NotImplementedError("Task cancellation not currently supported")


# -----------------------------
# Agent Card Configuration
# -----------------------------
def get_lightspeed_agent_card() -> AgentCard:
    """
    Generate the A2A Agent Card for Lightspeed.

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
        security=[
            {
                "bearer": []
            }
        ],
        security_schemes={
            "bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearer_format": "JWT",
                "description": "Bearer token for authentication with OpenShift services"
            }
        }
    )


# -----------------------------
# FastAPI Endpoints
# -----------------------------
@router.get("/.well-known/agent.json", response_model=AgentCard)
@router.get("/.well-known/agent-card.json", response_model=AgentCard)
@authorize(Action.A2A_AGENT_CARD)
async def get_agent_card(
    auth: Annotated[AuthTuple, Depends(auth_dependency)],  # pylint: disable=unused-argument
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
    except Exception as exc:
        logger.error("Error serving A2A Agent Card: %s", str(exc))
        raise


def _create_a2a_app(auth_token: str, mcp_headers: dict[str, dict[str, str]]):
    """
    Create an A2A Starlette application instance with auth context.

    Args:
        auth_token: Authentication token for the request
        mcp_headers: MCP headers for context propagation

    Returns:
        A2A Starlette ASGI application
    """
    agent_executor = LightspeedAgentExecutor(
        auth_token=auth_token,
        mcp_headers=mcp_headers
    )

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=get_lightspeed_agent_card(),
        http_handler=request_handler,
    )

    return a2a_app.build()


@router.api_route("/a2a", methods=["GET", "POST"])
@authorize(Action.A2A_JSONRPC)
async def handle_a2a_jsonrpc(
    request: Request,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
):
    """
    Main A2A JSON-RPC endpoint following the A2A protocol specification.

    This endpoint uses the DefaultRequestHandler from the A2A SDK to handle
    all JSON-RPC requests including message/send, message/stream, etc.

    The A2A SDK application is created per-request to include authentication
    context while still leveraging FastAPI's authorization middleware.

    Args:
        request: FastAPI request object
        auth: Authentication tuple
        mcp_headers: MCP headers for context propagation

    Returns:
        JSON-RPC response or streaming response
    """
    # Extract auth token
    auth_token = auth[3] if len(auth) > 3 else ""

    # Create A2A app with auth context
    a2a_app = _create_a2a_app(auth_token, mcp_headers)

    # Forward the request to the A2A app
    # The A2A SDK will handle all JSON-RPC protocol details
    from starlette.responses import Response

    scope = request.scope.copy()
    scope['path'] = '/'  # A2A app expects root path

    async def receive():
        return await request.receive()

    response_started = False
    response_body = []
    status_code = 200
    headers = []

    async def send(message):
        nonlocal response_started, status_code, headers
        if message['type'] == 'http.response.start':
            response_started = True
            status_code = message['status']
            headers = message.get('headers', [])
        elif message['type'] == 'http.response.body':
            response_body.append(message.get('body', b''))

    await a2a_app(scope, receive, send)

    # Return the response from A2A app
    return Response(
        content=b''.join(response_body),
        status_code=status_code,
        headers=dict((k.decode(), v.decode()) for k, v in headers)
    )


@router.get("/a2a/health")
async def a2a_health_check():
    """
    Health check endpoint for A2A service.

    Returns:
        Dict with health status information.
    """
    return {
        "status": "healthy",
        "service": "lightspeed-a2a",
        "version": __version__,
        "a2a_sdk_version": "0.2.1",
        "timestamp": datetime.now().isoformat()
    }
