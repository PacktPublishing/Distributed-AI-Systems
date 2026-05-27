"""
Agentic AI and Multi-Agent Systems

This module provides implementations for:
- Multi-agent coordination for distributed inference
- Tool-calling with distributed execution
- Agent orchestration patterns
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
from collections import defaultdict


class AgentRole(Enum):
    """Roles for agents in a multi-agent system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    receiver: str
    content: Any
    message_type: str = "request"
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None


@dataclass
class ToolCall:
    """Represents a tool call from an agent."""
    tool_name: str
    arguments: dict
    agent_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    result: Any
    success: bool
    execution_time_ms: float
    error: Optional[str] = None


class Tool:
    """Base class for tools that agents can call."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError


class DistributedToolExecutor:
    """
    Execute tool calls across distributed workers.
    
    Routes tool calls to appropriate workers based on tool type
    and worker availability.
    """
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.worker_assignments: dict[str, list[str]] = defaultdict(list)
        self.execution_stats: dict[str, list[float]] = defaultdict(list)
    
    def register_tool(self, tool: Tool, worker_ids: Optional[list[str]] = None) -> None:
        """Register a tool with optional worker assignment."""
        self.tools[tool.name] = tool
        if worker_ids:
            self.worker_assignments[tool.name] = worker_ids
    
    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        start_time = time.time()
        
        if call.tool_name not in self.tools:
            return ToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                execution_time_ms=0,
                error=f"Unknown tool: {call.tool_name}"
            )
        
        tool = self.tools[call.tool_name]
        
        try:
            result = await tool.execute(**call.arguments)
            execution_time = (time.time() - start_time) * 1000
            
            self.execution_stats[call.tool_name].append(execution_time)
            
            return ToolResult(
                tool_name=call.tool_name,
                result=result,
                success=True,
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            tool_name: {
                'count': len(times),
                'avg_ms': sum(times) / len(times) if times else 0,
                'max_ms': max(times) if times else 0,
            }
            for tool_name, times in self.execution_stats.items()
        }


class Agent:
    """
    An agent that can process requests and call tools.
    
    Supports:
    - Message passing with other agents
    - Tool calling with distributed execution
    - State management
    """
    
    def __init__(self, agent_id: str, role: AgentRole, 
                 tool_executor: Optional[DistributedToolExecutor] = None):
        self.agent_id = agent_id
        self.role = role
        self.tool_executor = tool_executor
        
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.state: dict = {}
        self.conversation_history: list[dict] = []
    
    async def send_message(self, receiver: 'Agent', content: Any, 
                          message_type: str = "request") -> None:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver.agent_id,
            content=content,
            message_type=message_type
        )
        await receiver.receive_message(message)
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        await self.message_queue.put(message)
    
    async def process_messages(self) -> None:
        """Process pending messages."""
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            await self.handle_message(message)
    
    async def handle_message(self, message: AgentMessage) -> Any:
        """Handle a received message. Override in subclasses."""
        self.conversation_history.append({
            'role': 'user',
            'content': str(message.content),
            'sender': message.sender
        })
        return None
    
    async def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Call a tool through the distributed executor."""
        if not self.tool_executor:
            return ToolResult(
                tool_name=tool_name,
                result=None,
                success=False,
                execution_time_ms=0,
                error="No tool executor configured"
            )
        
        call = ToolCall(
            tool_name=tool_name,
            arguments=kwargs,
            agent_id=self.agent_id
        )
        
        return await self.tool_executor.execute(call)


class MultiAgentOrchestrator:
    """
    Orchestrate multiple agents for complex tasks.
    
    Patterns supported:
    - Sequential: Agents process in order
    - Parallel: Agents process concurrently
    - Hierarchical: Coordinator delegates to workers
    """
    
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.coordinator: Optional[Agent] = None
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the orchestrator."""
        self.agents[agent.agent_id] = agent
        if agent.role == AgentRole.COORDINATOR:
            self.coordinator = agent
    
    async def run_sequential(self, task: Any, agent_order: list[str]) -> list[Any]:
        """Run agents sequentially, passing output to next agent."""
        results = []
        current_input = task
        
        for agent_id in agent_order:
            agent = self.agents[agent_id]
            message = AgentMessage(
                sender="orchestrator",
                receiver=agent_id,
                content=current_input
            )
            await agent.receive_message(message)
            result = await agent.handle_message(message)
            results.append(result)
            current_input = result
        
        return results
    
    async def run_parallel(self, task: Any, agent_ids: list[str]) -> list[Any]:
        """Run agents in parallel on the same task."""
        async def run_agent(agent_id: str) -> Any:
            agent = self.agents[agent_id]
            message = AgentMessage(
                sender="orchestrator",
                receiver=agent_id,
                content=task
            )
            await agent.receive_message(message)
            return await agent.handle_message(message)
        
        results = await asyncio.gather(*[run_agent(aid) for aid in agent_ids])
        return list(results)
    
    async def run_hierarchical(self, task: Any) -> Any:
        """Run with coordinator delegating to workers."""
        if not self.coordinator:
            raise ValueError("No coordinator agent configured")
        
        # Send task to coordinator
        message = AgentMessage(
            sender="orchestrator",
            receiver=self.coordinator.agent_id,
            content=task
        )
        await self.coordinator.receive_message(message)
        
        # Coordinator handles delegation internally
        return await self.coordinator.handle_message(message)


class ReasoningAgent(Agent):
    """
    Agent specialized for multi-step reasoning tasks.
    
    Implements chain-of-thought reasoning with tool use.
    """
    
    def __init__(self, agent_id: str, tool_executor: DistributedToolExecutor,
                 max_reasoning_steps: int = 10):
        super().__init__(agent_id, AgentRole.SPECIALIST, tool_executor)
        self.max_reasoning_steps = max_reasoning_steps
        self.reasoning_trace: list[dict] = []
    
    async def reason(self, query: str) -> dict:
        """Perform multi-step reasoning on a query."""
        self.reasoning_trace = []
        current_context = query
        
        for step in range(self.max_reasoning_steps):
            # Determine next action (simplified - would use LLM in practice)
            action = self._determine_action(current_context)
            
            if action['type'] == 'answer':
                self.reasoning_trace.append({
                    'step': step,
                    'action': 'final_answer',
                    'content': action['content']
                })
                return {
                    'answer': action['content'],
                    'reasoning_trace': self.reasoning_trace
                }
            
            elif action['type'] == 'tool_call':
                result = await self.call_tool(
                    action['tool_name'],
                    **action['arguments']
                )
                
                self.reasoning_trace.append({
                    'step': step,
                    'action': 'tool_call',
                    'tool': action['tool_name'],
                    'result': result.result if result.success else result.error
                })
                
                current_context = f"{current_context}\nTool result: {result.result}"
            
            elif action['type'] == 'think':
                self.reasoning_trace.append({
                    'step': step,
                    'action': 'think',
                    'thought': action['content']
                })
                current_context = f"{current_context}\nThought: {action['content']}"
        
        return {
            'answer': "Max reasoning steps reached",
            'reasoning_trace': self.reasoning_trace
        }
    
    def _determine_action(self, context: str) -> dict:
        """Determine next action based on context. Simplified placeholder."""
        # In practice, this would call an LLM to decide
        if len(self.reasoning_trace) >= self.max_reasoning_steps - 1:
            return {'type': 'answer', 'content': 'Reasoning complete'}
        return {'type': 'think', 'content': 'Analyzing the problem...'}


if __name__ == "__main__":
    async def main():
        # Create tool executor
        executor = DistributedToolExecutor()
        
        # Create a simple tool
        class CalculatorTool(Tool):
            def __init__(self):
                super().__init__("calculator", "Perform arithmetic operations")
            
            async def execute(self, operation: str, a: float, b: float) -> float:
                if operation == "add":
                    return a + b
                elif operation == "multiply":
                    return a * b
                else:
                    raise ValueError(f"Unknown operation: {operation}")
        
        executor.register_tool(CalculatorTool())
        
        # Create agents
        coordinator = Agent("coordinator", AgentRole.COORDINATOR, executor)
        worker1 = Agent("worker1", AgentRole.WORKER, executor)
        worker2 = Agent("worker2", AgentRole.WORKER, executor)
        
        # Test tool calling
        result = await coordinator.call_tool("calculator", operation="add", a=5, b=3)
        print(f"Tool result: {result.result}, success: {result.success}")
        
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()
        orchestrator.add_agent(coordinator)
        orchestrator.add_agent(worker1)
        orchestrator.add_agent(worker2)
        
        # Test parallel execution
        results = await orchestrator.run_parallel(
            "Process this task",
            ["worker1", "worker2"]
        )
        print(f"Parallel results: {results}")
        
        # Test reasoning agent
        reasoning_agent = ReasoningAgent("reasoner", executor, max_reasoning_steps=5)
        reasoning_result = await reasoning_agent.reason("What is 2 + 2?")
        print(f"Reasoning trace: {len(reasoning_result['reasoning_trace'])} steps")
    
    asyncio.run(main())
