# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        instruction = (
            f"Modify any part of the codebase at `{repo_path}` to improve its performance. "
            f"IMPORTANT: Do NOT modify any files under `{repo_path}/analysis/`. "
            f"That directory contains instrumentation and metric-tracking scripts that are "
            f"maintained separately and must remain unchanged across all generations."
        )

        new_msg_history, usage = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')
        return usage
