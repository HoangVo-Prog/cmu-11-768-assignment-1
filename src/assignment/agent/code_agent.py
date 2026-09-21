"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""
        
        self.tools.extend([
            EXECUTE_TOOL,
            SEND_MESSAGE_TOOL,
        ])       
         
        self.system_prompt = """
            You are a coding agent. Your goal is to solve the software issue provided by the user.
            You can inspect and modify the repository using the available tools.
        """
        
        self.task_prompt = self.task
                
        if self.skills:   
            skill_metadata = "\n\n".join(
                skill["metadata"]
                for skill in self.skills.values()
            )   
            self.system_prompt += f"\n\nAvailable skills:\n{skill_metadata}"
            
        self.system_prompt += f"""
            <system_information>
            {{
            "machine": "{self.env.machine}",
            "release": "{self.env.release}",
            "system": "{self.env.system}",
            "version": "{self.env.version}"
            }}
            </system_information>
        """

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""
        
        observations = []

        for tool_call in tool_calls:
            call_id = tool_call.get("id", "unknown")
            name = tool_call.get("function", {}).get("name", "unknown")
            raw_arguments = tool_call.get("function", {}).get("arguments", "{}")    

            try:
                arguments = json.loads(raw_arguments)
                
                if not isinstance(arguments, dict):
                    raise ValueError("Tool call arguments must be a JSON object.")
                
                if name == "execute":
                    result = self.env.execute(**arguments)
                    content = format_tool_output(result)

                elif name == "send_message":
                    summary = arguments.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError("send_message requires a string summary.")
                    
                    content = "Summary of the message sent to the user: " + summary
                    self.finished = True
                
                elif name == "invoke_skill":
                    skill_name = arguments.get("name")
                    if not isinstance(name, str):
                        raise ValueError("invoke_skill requires a string skill_name.")
                    
                    skill = self.skills.get(skill_name)

                    if skill is None:
                        content = f"Skill not found: {skill_name}"
                    else:
                        content = skill["content"]
    
                else:
                    content = f"Unknown tool: {name}"

            except json.JSONDecodeError as e:
                content = "Malformed JSON in tool call arguments: " + str(e)
            except Exception as e:
                content = "Error executing tool call: " + str(e)
                
            observations.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })

        return observations