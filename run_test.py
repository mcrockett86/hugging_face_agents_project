#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# standalone_test.py
# Description: This script is used to test the functionality of the agent workflow in a standalone mode.
# Author: Michael Crockett
# Date: 2025-05-20


# load environment variables from .env file (e.g., OpenAI API key)
from dotenv import load_dotenv
load_dotenv('agents_course.env')

from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from tqdm import tqdm
import pandas as pd
import asyncio
import re

from llama_index.core.memory import ChatMemoryBuffer

# load the agents
from agents.agent_basic_tools import agent_basic_tools
from agents.agent_duck_duck_go import agent_duck_duck_go
from agents.agent_wikipedia import agent_wikipedia
from agents.agent_vision import agent_vision


# --- GAIA_Agent Agent Definition ---
class GAIA_Agent:
    """
    GAIA_Agent is a composite agent that orchestrates multiple specialized agents
    to answer questions using a variety of tools and resources.
    It integrates basic tools, web search, Wikipedia search, and vision capabilities.

    Attributes:
        agent (AgentWorkflow): The main agent workflow that manages the interaction between specialized agents.
        ctx (Context): The context for the agent workflow, used to maintain state and pass information.
    """

    def __init__(self):
        print("GAIA_Agent initialized.")

        # Initialize the agents with their respective tools and settings
        self.agent = AgentWorkflow(
            agents=[agent_basic_tools, agent_wikipedia, agent_duck_duck_go, agent_vision],
            root_agent=agent_basic_tools.name,
            timeout=120.0, # seconds
            initial_state={"name": "unset"}
        )


    async def __call__(self, question: str) -> str:
        print(f"\n\nAgent received question: {question}")

        memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
        handler = self.agent.run(user_msg=question, memory=memory)

        # print progres updates in detail
        current_agent = None
        current_tool_calls = ""
        async for event in handler.stream_events():
            if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
            ):
                current_agent = event.current_agent_name
                print(f"\n{'='*50}")
                print(f"🤖 Agent: {current_agent}")
                print(f"{'='*50}\n")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                if event.tool_calls:
                    print(
                        "🛠️  Planning to use tools:",
                        [call.tool_name for call in event.tool_calls],
                    )
            elif isinstance(event, ToolCallResult):
                print(f"🔧 Tool Result ({event.tool_name}):")
                print(f"   Arguments:   {event.tool_kwargs}")
                print(f"   Output:      {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"🔨 Calling Tool:   {event.tool_name}")
                print(f"   With arguments: {event.tool_kwargs}")

        # wait for the task to be done
        _ = await asyncio.gather(handler)
        value = handler.result()
        print(f'Agent returned value.result(): {value}')

        # if the answer is super long, just return the last line of the response to force terseness
        last_line_str = str(value).split("\n")[-1]

        match = re.search(r"\{(.*?)\}", last_line_str)
        if match:
            result = match.group(1)
            return result
    
        else:
            return last_line_str


async def run_test():

    df_gaia_test = pd.read_csv('data/gaia_dataset.csv')

    df_gaia_test['prediction'] = ''
    df_gaia_test['correct'] = ''

    ga = GAIA_Agent()

    for i, row in tqdm(df_gaia_test.iterrows(), total=df_gaia_test.shape[0]):

        try:
            Q = row['question']
            A = row['answer']
            P = await ga(Q)

            print(f"question:        {Q}")
            print(f"expected answer: {A}")
            print(f"prediction:      {P}")

            df_gaia_test.loc[i, 'prediction'] = str(P).upper()
            df_gaia_test.loc[i, 'correct'] = (str(P).upper() == str(A).upper())

            # save the results incrementally
            df_gaia_test.to_csv('data/gaia_dataset_snapshot.csv', index=False)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            df_gaia_test.loc[i, 'prediction'] = ''
            df_gaia_test.loc[i, 'correct'] = False 

    pct_correct = df_gaia_test['correct'].sum() / df_gaia_test.shape[0]

    print("Final Results:")
    print(f"pct_correct: {pct_correct:.2%}")


if __name__ == "__main__":

    asyncio.run(run_test())