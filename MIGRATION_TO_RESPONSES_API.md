# Migration from Agent API to Responses API

This document summarizes the changes made to switch lightspeed-stack from using the llama-stack Agent API to the Responses API.

## Overview

The migration replaces the stateful agent-based approach with a stateless response-chaining approach that is more compatible with OpenAI's API patterns.

## Files Modified

### 1. `lightspeed-stack/src/utils/endpoints.py`

**Changes Made:**
- **Removed**: `get_agent()` function and all agent creation logic
- **Removed**: Agent-specific imports (`AsyncAgent`, `GraniteToolParser`)
- **Added**: `get_rag_tools()` function to convert vector DBs to tools format
- **Added**: `get_mcp_tools()` function to convert MCP servers to tools format

**New Functions:**
```python
def get_rag_tools(vector_db_ids: list[str]) -> list[dict] | None
def get_mcp_tools(mcp_servers: list) -> list[dict]
```

These functions create tools in the correct format for llama-stack's responses API:
- RAG tools use `"type": "file_search"` with `vector_store_ids`
- MCP tools use `"type": "mcp"` with `server_label` and `server_url`

### 2. `lightspeed-stack/src/app/endpoints/query.py`

**Changes Made:**
- **Updated imports**: Replaced agent imports with responses API imports
- **Replaced**: `get_agent()` call with tool preparation logic
- **Replaced**: `agent.create_turn()` with `client.create_openai_response()`
- **Updated**: Response processing to handle `OpenAIResponseObject` format
- **Removed**: `get_rag_toolgroups()` function (replaced with `get_rag_tools()`)

**Key Logic Changes:**
```python
# Old: Agent creation and turn
agent, conversation_id, session_id = await get_agent(...)
response = await agent.create_turn(...)

# New: Response creation with tools
tools = []
if not query_request.no_tools:
    # Add RAG tools (file search)
    rag_tools = get_rag_tools(vector_db_ids)  # Returns [{"type": "file_search", ...}]
    if rag_tools:
        tools.extend(rag_tools)

    # Add MCP tools
    mcp_tools = get_mcp_tools(mcp_servers)    # Returns [{"type": "mcp", ...}]
    if mcp_tools:
        tools.extend(mcp_tools)

response = await client.responses.create(
    input=query_request.query,
    model=model_id,
    instructions=system_prompt,
    previous_response_id=query_request.conversation_id,
    tools=tools if tools else None,
    stream=False,
    store=True,
)
```

### 3. `lightspeed-stack/src/app/endpoints/streaming_query.py`

**Changes Made:**
- **Updated imports**: Added responses API imports
- **Replaced**: Agent logic with responses API logic
- **Updated**: Function signatures to use `OpenAIResponseObjectStream`
- **Note**: Streaming event processing may need additional updates for full compatibility

### 4. `lightspeed-stack/src/utils/suid.py`

**Changes Made:**
- **Updated**: `check_suid()` function to accept both UUID format and response ID format
- **Backward Compatible**: Still accepts regular UUIDs for existing sessions

**Fix Details:**
The responses API returns response IDs with a "resp-" prefix (e.g., `resp-1efc93a0-39fa-41ab-aa5e-f02cc64e3ab4`), but the original validation expected pure UUID format. The updated function now accepts both:
```python
# Old validation: Only UUID format
uuid.UUID(suid)  # ✅ Works: "1efc93a0-39fa-41ab-aa5e-f02cc64e3ab4"
                 # ❌ Fails: "resp-1efc93a0-39fa-41ab-aa5e-f02cc64e3ab4"

# New validation: Both UUID and response ID formats
if suid.startswith("resp-"):
    uuid.UUID(suid[5:])  # ✅ Works: "resp-1efc93a0-39fa-41ab-aa5e-f02cc64e3ab4"
else:
    uuid.UUID(suid)      # ✅ Works: "1efc93a0-39fa-41ab-aa5e-f02cc64e3ab4"
```

### 5. `lightspeed-stack/src/app/endpoints/conversations.py`

**Changes Made:**
- **Updated**: `simplify_session_data()` → `simplify_response_data()` to handle response format instead of session format
- **Updated**: Get conversation endpoint to use `client.responses.retrieve()` or `client.agents.get_openai_response()`
- **Updated**: Delete conversation endpoint to use responses API (with fallback if deletion not supported)
- **Kept**: List conversations endpoint (uses local database, no changes needed)

**Key Changes:**
```python
# Old: Agent sessions
agent_sessions = await client.agents.session.list(agent_id=agent_id)
session_response = await client.agents.session.retrieve(agent_id=agent_id, session_id=session_id)

# New: Individual responses
response_obj = await client.responses.retrieve(response_id=response_id)
# OR fallback: response_obj = await client.agents.get_openai_response(response_id=response_id)
```

**Note**: With responses API, each "conversation" is actually a single response. True conversation history would require chaining multiple responses, which is handled at the application level, not the llama-stack level.

## API Changes

### Before (Agent API)
```python
# Session-based conversation management
agent = AsyncAgent(client, model=model_id, instructions=system_prompt, ...)
await agent.initialize()
conversation_id = agent.agent_id
session_id = await agent.create_session(get_suid())

# Turn creation with toolgroups
response = await agent.create_turn(
    messages=[UserMessage(role="user", content=query)],
    session_id=session_id,
    toolgroups=toolgroups,
)
```

### After (Responses API)
```python
# Response chaining with previous_response_id
tools = get_rag_tools(vector_db_ids) + get_mcp_tools(mcp_servers)

response = await client.responses.create(
    input=query,
    model=model_id,
    instructions=system_prompt,
    previous_response_id=conversation_id,  # Chain from previous response
    tools=tools,
    stream=False,
    store=True,
)
conversation_id = response.id  # Use response ID for next request
```

## Important: Responses API Access Pattern

**Critical Fix**: The responses API has its own client interface in llama-stack and should be accessed through the responses client:

```python
# ❌ Incorrect - will cause AttributeError
response = await client.create_openai_response(...)
response = await client.agents.create_openai_response(...)

# ✅ Correct - access through responses API
response = await client.responses.create(...)
```

This follows the llama-stack client pattern where each API has its own client interface (similar to `client.models.list()`, `client.inference.chat_completion()`, etc.).

## Tool Format Changes

A critical change is how tools are formatted. The responses API uses specific tool types that match llama-stack's pydantic models:

### RAG Tools
```python
# Old: Toolgroups format
toolgroups = [
    ToolgroupAgentToolGroupWithArgs(
        name="builtin::rag/knowledge_search",
        args={"vector_db_ids": vector_db_ids}
    )
]

# New: File search tools format
tools = [{
    "type": "file_search",
    "vector_store_ids": vector_db_ids,
    "max_num_results": 10
}]
```

### MCP Tools
```python
# Old: Toolgroups with server names
toolgroups = [mcp_server.name for mcp_server in mcp_servers]

# New: MCP tools format
tools = [{
    "type": "mcp",
    "server_label": mcp_server.name,
    "server_url": mcp_server.url,
    "require_approval": "never"
}]
```

## Key Differences

| Aspect | Agent API | Responses API |
|--------|-----------|---------------|
| **State Management** | Session-based persistence | Response chaining via IDs |
| **Conversation Tracking** | Session ID | Response ID |
| **Tool Integration** | `toolgroups` parameter | `tools` array |
| **Response Format** | `Turn` object | `OpenAIResponseObject` |
| **OpenAI Compatibility** | Custom format | Direct OpenAI compatibility |
| **Session Persistence** | Automatic | Manual via chaining |

## Benefits

1. **Simplified Architecture**: No session management overhead
2. **OpenAI Compatibility**: Direct compatibility with OpenAI response patterns
3. **Flexible Conversations**: Can branch/fork from any response
4. **Dynamic Configuration**: Can change model/tools per response
5. **Reduced Complexity**: Fewer abstraction layers

## Trade-offs

1. **Manual State Management**: Must manually track conversation chains
2. **No Input/Output Shields**: Responses API doesn't support shields (yet)
3. **Tool Format Changes**: Need to convert toolgroups to tools format
4. **Breaking Changes**: Conversation IDs are now response IDs
5. **Smart Conversation Chaining**: Text-only conversations maintain continuity; tool-based queries are independent to avoid llama-stack bugs

## Remaining Work

1. **Streaming Events**: The streaming response processing may need updates to handle the different event format from responses API
2. **Error Handling**: Update error handling for responses API specific errors
3. **Testing**: Comprehensive testing of the new response chaining logic
4. **Documentation**: Update API documentation to reflect the changes

## Recent Fixes Applied

- ✅ **Tool Format Fix**: Updated RAG and MCP tool formats to match llama-stack's pydantic models (`OpenAIResponseInputToolFileSearch` and `OpenAIResponseInputToolMCP`)
- ✅ **API Access Fix**: Corrected to use `client.responses.create()` instead of `client.agents.create_openai_response()`
- ✅ **Conversation ID Validation Fix**: Updated `suid.check_suid()` to accept response IDs with "resp-" prefix (e.g., `resp-1efc93a0-39fa-41ab-aa5e-f02cc64e3ab4`)
- ✅ **Stateless Design**: Embraced responses API's stateless design - each response is independent with optional chaining
- ⚠️ **Smart Chaining Workaround**: Implemented selective conversation chaining to avoid llama-stack bug with tool outputs
- ✅ **Enhanced System Prompt**: Added tool usage instructions to encourage LLM to use available MCP tools
- ✅ **MCP Debug Logging**: Added comprehensive debug logging for MCP tool calls and model responses

## Architectural Approach

### Stateless Design

The responses API is designed to be stateless, where each response is independent. This is actually a cleaner approach than forcing conversation state:

**How it works:**
1. Each query creates a new response with its own ID
2. If a client wants to chain responses, they can provide the previous response ID as `conversation_id`
3. This gets passed as `previous_response_id` to llama-stack for optional context chaining
4. The new response ID is returned for potential future chaining

**Benefits:**
- ✅ **Simpler architecture** - no forced session management
- ✅ **Flexible chaining** - client decides what to chain and when
- ✅ **Stateless service** - aligns with responses API design
- ✅ **Avoids artificial conversation states**

**Usage Pattern:**
```
Query 1: conversation_id=null → Response ID: resp-abc-123
Query 2: conversation_id=resp-abc-123 → Response ID: resp-def-456  (chained if no tools)
Query 3: conversation_id=null → Response ID: resp-ghi-789  (new thread)
```

### Smart Chaining Workaround

Due to a llama-stack bug with tool outputs in conversation chaining, we implemented a selective chaining strategy:

```python
# Only chain when no tools are present to avoid llama-stack bug
use_chaining = query_request.conversation_id and not tools

response = await client.responses.create(
    input=query_request.query,
    model=model_id,
    instructions=system_prompt,
    previous_response_id=query_request.conversation_id if use_chaining else None,
    tools=tools if tools else None,
    stream=False,
    store=True,
)
```

**Behavior:**
- ✅ **Text-only queries**: Full conversation chaining works
- ⚠️ **Queries with tools (RAG/MCP)**: Each query is independent to avoid crashes
- 🔄 **Future**: When llama-stack fixes the bug, remove this workaround

### Enhanced System Prompt for Tool Usage

Since the responses API doesn't support `tool_choice=required` (it defaults to `auto`), we enhanced the system prompt to encourage tool usage:

```python
def get_system_prompt(query_request: QueryRequest, config: AppConfig, has_mcp_tools: bool = False) -> str:
    base_prompt = # ... get base prompt from config/request/default ...

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
```

**Benefits:**
- ✅ **Encourages tool usage**: LLM is more likely to use MCP tools when appropriate
- ✅ **Context-specific**: Only adds instructions when MCP tools are actually available
- ✅ **Backward compatible**: Doesn't affect queries without tools

### MCP Debug Logging

Added comprehensive logging to track MCP tool usage and model responses for easier debugging:

**Query Processing Logs:**
```
MCP DEBUGGING: Configured 2 MCP tools: ['assisted', 'another-server']
MCP DEBUGGING: Creating response with query: 'list my clusters' and 2 tools
MCP DEBUGGING: Received response with ID: resp-abc-123, output items: 3
```

**Response Processing Logs:**
```
MCP DEBUGGING: Processing output item 0, type: OpenAIResponseOutputMessage
MCP DEBUGGING: Model response content: 'I'll help you list your clusters using the available tools.'
MCP DEBUGGING: Found 1 tool calls in output item 1
MCP DEBUGGING: Tool call 0 - Name: assisted::list_clusters, Args: {}
MCP DEBUGGING: Response processing complete - Tool calls: 1, Response length: 245 chars
```

**Streaming Logs (additional):**
```
MCP DEBUGGING: Starting streaming response processing
MCP DEBUGGING: Processing chunk 0, event_type: step_start, step_type: tool_execution
MCP DEBUGGING: Tool execution step started - chunk 1
MCP DEBUGGING: Tool execution step complete - 1 tool calls, 1 responses
MCP DEBUGGING: Tool call - Name: assisted::list_clusters, Args: {}
MCP DEBUGGING: Tool response - Name: assisted::list_clusters, Response: '[{"name": "cluster1"...}]'
```

**Benefits:**
- ✅ **Full visibility**: See exactly when MCP tools are called and what they return
- ✅ **Performance tracking**: Monitor response times and content sizes
- ✅ **Debugging friendly**: Easy to identify when tools aren't being used
- ✅ **Structured format**: All MCP logs prefixed with "MCP DEBUGGING" for easy filtering

## Configuration

The llama-stack configuration (`run.yaml`) needs the following for responses API to work:

1. **Agents API enabled**: The `agents` API must be in the `apis` list (it contains the responses endpoints)

2. **Responses Store configured**: The agents provider must have a `responses_store` configured:

```yaml
providers:
  agents:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      persistence_store:
        db_path: .llama/distributions/ollama/agents_store.db
        namespace: null
        type: sqlite
      responses_store:  # Required for responses API
        db_path: .llama/distributions/ollama/responses_store.db
        type: sqlite
```

The existing `lightspeed-stack/run.yaml` already has this configuration, so no changes are needed.

## Migration Path

For existing conversations:
- Previous conversation IDs (agent session IDs) won't be compatible
- Need migration strategy or deprecation notice for existing conversations
- Consider implementing backward compatibility layer if needed
