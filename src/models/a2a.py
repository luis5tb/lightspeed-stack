"""Models for Agent-to-Agent (A2A) protocol support."""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, AnyUrl


class A2AVersion(str, Enum):
    """Supported A2A protocol versions."""
    
    V0_3_0 = "0.3.0"


class AuthScheme(str, Enum):
    """Supported authentication schemes."""
    
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "apiKey"
    OAUTH2 = "oauth2"


class TaskStatus(str, Enum):
    """Task execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentType(str, Enum):
    """Supported content types for A2A communication."""
    
    TEXT = "text"
    JSON = "json"
    FORM = "form"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


class A2ACapability(BaseModel):
    """Model representing an A2A agent capability."""
    
    name: str = Field(description="Name of the capability")
    description: str = Field(description="Description of what this capability does")
    input_types: List[ContentType] = Field(description="Supported input content types")
    output_types: List[ContentType] = Field(description="Supported output content types")
    streaming: bool = Field(default=False, description="Whether this capability supports streaming")
    async_execution: bool = Field(default=False, description="Whether this capability supports async execution")


class A2AAuthentication(BaseModel):
    """Model representing authentication requirements for A2A communication."""
    
    scheme: AuthScheme = Field(description="Authentication scheme")
    required: bool = Field(default=True, description="Whether authentication is required")
    description: Optional[str] = Field(None, description="Description of authentication requirements")


class A2AProvider(BaseModel):
    """Model representing the agent provider information."""
    
    name: str = Field(description="Provider name")
    url: Optional[AnyUrl] = Field(None, description="Provider website URL")
    contact: Optional[str] = Field(None, description="Contact information")


class A2AAgentCard(BaseModel):
    """Model representing an A2A Agent Card.
    
    The Agent Card is the core identity document for an A2A agent,
    describing its capabilities, endpoints, and metadata.
    """
    
    # Core identification
    name: str = Field(description="Human-readable name of the agent")
    description: str = Field(description="Description of what the agent does")
    version: str = Field(description="Agent version")
    a2a_version: A2AVersion = Field(description="A2A protocol version supported")
    
    # Service endpoints
    service_url: AnyUrl = Field(description="Base URL for the agent's A2A service")
    documentation_url: Optional[AnyUrl] = Field(None, description="URL to agent documentation")
    
    # Provider information
    provider: A2AProvider = Field(description="Information about the agent provider")
    
    # Capabilities
    capabilities: List[A2ACapability] = Field(description="List of agent capabilities")
    
    # Communication settings
    default_input_type: ContentType = Field(default=ContentType.TEXT, description="Default input content type")
    default_output_type: ContentType = Field(default=ContentType.TEXT, description="Default output content type")
    supports_streaming: bool = Field(default=False, description="Whether agent supports streaming responses")
    supports_push_notifications: bool = Field(default=False, description="Whether agent supports push notifications")
    
    # Authentication
    authentication: Optional[A2AAuthentication] = Field(None, description="Authentication requirements")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "OpenShift Assisted Installer AI Assistant",
                    "description": "AI-powered assistant specialized in OpenShift cluster installation, configuration, and troubleshooting using assisted-installer backend",
                    "version": "1.0.0",
                    "a2a_version": "0.3.0",
                    "service_url": "https://lightspeed.example.com/a2a",
                    "documentation_url": "https://lightspeed.example.com/docs",
                    "provider": {
                        "name": "Red Hat",
                        "url": "https://redhat.com",
                        "contact": "lightspeed-support@redhat.com"
                    },
                    "capabilities": [
                        {
                            "name": "cluster_installation_guidance",
                            "description": "Provide guidance and assistance for OpenShift cluster installation using assisted-installer",
                            "input_types": ["text", "json"],
                            "output_types": ["text", "json"],
                            "streaming": True,
                            "async_execution": False
                        },
                        {
                            "name": "cluster_configuration_validation",
                            "description": "Validate and provide recommendations for OpenShift cluster configuration parameters",
                            "input_types": ["json", "text"],
                            "output_types": ["json", "text"],
                            "streaming": False,
                            "async_execution": False
                        },
                        {
                            "name": "installation_troubleshooting",
                            "description": "Help troubleshoot OpenShift cluster installation issues and provide solutions",
                            "input_types": ["text", "json"],
                            "output_types": ["text", "json"],
                            "streaming": True,
                            "async_execution": False
                        }
                    ],
                    "default_input_type": "text",
                    "default_output_type": "text",
                    "supports_streaming": True,
                    "supports_push_notifications": False,
                    "authentication": {
                        "scheme": "bearer",
                        "required": True,
                        "description": "Bearer token authentication required"
                    },
                    "tags": ["openshift", "assisted-installer", "cluster-installation", "ai-assistant", "infrastructure"],
                    "metadata": {
                        "specialization": "OpenShift cluster installation and management",
                        "backend_integration": "assisted-installer",
                        "supported_platforms": ["bare-metal", "vsphere", "aws", "azure", "gcp"],
                        "installation_methods": ["assisted-installer", "agent-based-installer"]
                    }
                }
            ]
        }
    }


class A2AMessage(BaseModel):
    """Model representing an A2A message."""
    
    content: str = Field(description="Message content")
    content_type: ContentType = Field(default=ContentType.TEXT, description="Content type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")


class A2ATaskRequest(BaseModel):
    """Model representing an A2A task request."""
    
    task_id: Optional[str] = Field(None, description="Optional task identifier")
    capability: str = Field(description="Capability to invoke")
    input: A2AMessage = Field(description="Input message")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    streaming: bool = Field(default=False, description="Whether to stream the response")
    async_execution: bool = Field(default=False, description="Whether to execute asynchronously")


class A2ATaskResponse(BaseModel):
    """Model representing an A2A task response."""
    
    task_id: str = Field(description="Task identifier")
    status: TaskStatus = Field(description="Task execution status")
    output: Optional[A2AMessage] = Field(None, description="Output message if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100) for running tasks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")


class A2AJsonRpcRequest(BaseModel):
    """Model representing a JSON-RPC 2.0 request for A2A communication."""
    
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    method: str = Field(description="Method name")
    params: Dict[str, Any] = Field(description="Method parameters")
    id: Union[str, int] = Field(description="Request identifier")


class A2AJsonRpcResponse(BaseModel):
    """Model representing a JSON-RPC 2.0 response for A2A communication."""
    
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")
    id: Union[str, int] = Field(description="Request identifier")


class A2AJsonRpcError(BaseModel):
    """Model representing a JSON-RPC 2.0 error."""
    
    code: int = Field(description="Error code")
    message: str = Field(description="Error message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error data")
