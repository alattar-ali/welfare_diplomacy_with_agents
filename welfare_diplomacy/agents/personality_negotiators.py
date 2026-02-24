"""
Defines three types of personality based negotiators.
- therapist: Build trust and understand the other party's needs, while maintaining a diplomatic tone.
- art-of-the-deal: Aggressive agent, pushes for maximum advantage.
- back-burner: Creative, personalized approach that doesn't fit the other tools.
"""

import json
import random
from typing import Dict, List, Optional, Literal, Any, Sequence, TypedDict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from loguru import logger

import pydantic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

from pydantic import BaseModel, Field

import diplomacy
from welfare_diplomacy.agents.base_agent import DiplomacyAgent

Powers = Literal[
    "FRANCE",
    "ITALY",
    "RUSSIA",
    "ENGLAND",
    "GERMANY",
    "AUSTRIA",
    "TURKEY"
]


class NegotiationMessage(BaseModel):
    messages_to_send: Dict[Powers, str] = Field(
        default_factory=dict,
        description="Dictionary of powers to messages to send, "
                    "where keys are power names from FRANCE, ITALY, RUSSIA, ENGLAND, GERMANY, AUSTRIA, TURKEY."
                    "Names are case-sensitive & should be used exactly as stated."
    )


class OpponentModelMessage(BaseModel):
    opponent_model: Literal["therapist", "art-of-the-deal", "back-burner"] = Field(
        default="therapist",
        description="The model of the opponent's personality, "
                    "which can be 'therapist', 'art-of-the-deal', or 'back_burner'."
    )
    response_model: Literal["therapist", "art-of-the-deal", "back-burner"] = Field(
        default="therapist",
        description="The model of the response personality, "
                    "which can be 'therapist', 'art-of-the-deal', or 'back_burner'."
    )


class AgentState(BaseModel):
    current_power: str
    phase: str
    received_messages: Dict[str, List[str]] = Field(default_factory=dict)
    messages_to_send: Dict[Powers, str] = Field(default_factory=dict)
    opponent_model: Dict[Powers, str] = Field(default_factory=dict)


class PersonalityAgent(DiplomacyAgent):

    def __init__(self, game: diplomacy.Game, pow_name: str, personality="back_burner", **params):
        assert personality in ["therapist", "art-of-the-deal", "back_burner"], \
            f"Invalid personality type: {personality}. Choose from 'therapist', 'art-of-the-deal', or 'back_burner'."

        super().__init__(game, pow_name, **params)

        self._personality = personality
        personality_to_prompt_file = {
            "therapist": "therapist_system_prompt.txt",
            "art-of-the-deal": "aggressive_system_prompt.txt",
            "back_burner": "creative_system_prompt.txt",
        }
        prompt_dir = Path(__file__).resolve().parent / "personality_agent_prompts"
        prompt_path = prompt_dir / personality_to_prompt_file[self._personality]
        self._system_prompt = prompt_path.read_text()

        # Initialize LLM model with parameters
        self.model = self._build_model(params)
        self.model_msg_generator = self.model.with_structured_output(NegotiationMessage)

        # Initialize generate-messages agent
        self.generate_messages_agent = self._create_messages_agent()

    def generate_messages(self):
        # Extract previously exchanged messages between "self.power_name" and other powers
        messages = self._get_message_history_for_power()
        state = AgentState(
            current_power=self.pow_name,
            phase=self.game.get_current_phase(),
            received_messages=messages,
        )
        msg = self.generate_messages_agent.invoke(state)
        print(state.current_power, msg["messages_to_send"])

        trial = {
            "ENGLAND": "Hi!",
            "FRANCE": "Hi!",
            "TURKEY": "Hi!",
            "GERMANY": "Hi!",
            "RUSSIA": "Hi!",
            "AUSTRIA": "Hi!",
            "ITALY": "Hi!",
        }
        del trial[self.pow_name]
        return trial

    def generate_orders(self):
        # Get all locations where this power can issue orders
        orderable_locations = self.game.get_orderable_locations(self.pow_name)
        orders = []

        for location in orderable_locations:
            # Get all possible orders for the current location
            possible_orders = self.game.get_all_possible_orders()
            if possible_orders[location]:
                # Randomly select one valid order
                orders.append(random.choice(possible_orders[location]))

        return orders

    def _node_personality_agent(self, state: AgentState):
        """
        Use when you need to build trust and understand the other party's needs, while maintaining a diplomatic tone.

        :param messages:
        :return:
        """
        user_prompt = state.model_dump_json(indent=2)

        try:
            return self.model_msg_generator.invoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except pydantic.ValidationError as e:
            logger.error(f"Validation error in {self}: {e}")
            return NegotiationMessage(messages_to_send={})

    def _build_model(self, params):
        self._base_url = params["model_provider_url"]
        self._api_key = params["api_key"]
        self._model_name = params["model"]

        if self._api_key == "docker":
            self._model_name = "ai/" + params["model_name"]

        return ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model_name
        )

    def _create_messages_agent(self):
        graph = StateGraph(state_schema=AgentState)
        graph.add_node("message_generator", self._node_personality_agent)
        graph.add_edge(START, "message_generator")
        graph.add_edge("message_generator", END)
        return graph.compile(name=f"GenMsg({self.pow_name}, personality={self._personality})")

    def _get_message_history_for_power(self):
        messages = defaultdict(list)

        # Collect messages sent to and from the power in the current phase
        for msg in self.game.get_phase_data().messages.values():
            if msg.sender == self.pow_name:
                messages[msg.recipient].append(f"{msg.sender} said to {msg.recipient} that {msg.message}")
            elif msg.recipient == self.pow_name:
                messages[msg.sender].append(f"{msg.sender} said to {msg.recipient} that {msg.message}")

        return messages


class DynamicPersonalityAgent(PersonalityAgent):
    """
    Backward-compatible placeholder for the previously experimental dynamic agent.
    """

    def __init__(self, game: diplomacy.Game, pow_name: str, **params):
        # Preserve existing config compatibility by mapping dynamic agent requests
        # to the currently maintained PersonalityAgent behavior.
        super().__init__(game=game, pow_name=pow_name, personality="back_burner", **params)


# class DynamicPersonalityAgent(DiplomacyAgent):
#
#     def __init__(self, game: diplomacy.Game, pow_name: str, **params):
#         super().__init__(game, pow_name, **params)
#
#         self._system_prompt_opponent_model = (
#                 Path().absolute() / "agents" / "personality_agent_prompts" / f"sp_opponent_model.txt"
#         ).read_text()
#         self._user_prompt_opponent_model = (
#                 Path().absolute() / "agents" / "personality_agent_prompts" / f"up_opponent_model.txt"
#         ).read_text()
#         self._system_prompt_therapist = (
#                 Path().absolute() / "agents" / "personality_agent_prompts" / f"therapist_system_prompt.txt"
#         ).read_text()
#         self._system_prompt_aggressive = (
#                 Path().absolute() / "agents" / "personality_agent_prompts" / f"aggressive_system_prompt.txt"
#         ).read_text()
#         self._system_prompt_creative = (
#                 Path().absolute() / "agents" / "personality_agent_prompts" / f"creative_system_prompt.txt"
#         ).read_text()
#
#         # Initialize LLM model with parameters
#         self.model = self._build_model(params)
#         self.model_msg_generator = self.model.with_structured_output(NegotiationMessage)
#         self.model_opponent_model = self.model.with_structured_output(OpponentModelMessage)
#
#         # Initialize generate-messages agent
#         self.generate_messages_agent = self._create_messages_agent()
#
#     def generate_messages(self):
#         # Extract previously exchanged messages between "self.power_name" and other powers
#         # messages = self._get_message_history_for_power()
#         state = AgentState(
#             current_power=self.pow_name,
#             phase=self.game.get_current_phase(),
#             received_messages=dict(),
#         )
#         msg = self.generate_messages_agent.invoke(state)
#         print(state.current_power, msg["messages_to_send"])
#
#         trial = {
#             "ENGLAND": "Hi!",
#             "FRANCE": "Hi!",
#             "TURKEY": "Hi!",
#             "GERMANY": "Hi!",
#             "RUSSIA": "Hi!",
#             "AUSTRIA": "Hi!",
#             "ITALY": "Hi!",
#         }
#         del trial[self.pow_name]
#         return trial
#
#     def generate_orders(self):
#         # Get all locations where this power can issue orders
#         orderable_locations = self.game.get_orderable_locations(self.pow_name)
#         orders = []
#
#         for location in orderable_locations:
#             # Get all possible orders for the current location
#             possible_orders = self.game.get_all_possible_orders()
#             if possible_orders[location]:
#                 # Randomly select one valid order
#                 orders.append(random.choice(possible_orders[location]))
#
#         return orders
#
#     def _build_model(self, params):
#         self._base_url = params["model_provider_url"]
#         self._api_key = params["api_key"]
#         self._model_name = params["model"]
#
#         if self._api_key == "docker":
#             self._model_name = "ai/" + params["model_name"]
#
#         return ChatOpenAI(
#             base_url=self._base_url,
#             api_key=self._api_key,
#             model=self._model_name
#         )
#
#     def _node_mapper(self, state: AgentState) -> AgentState:
#         state.messages_to_send = {key: "" for key in
#                                   {"FRANCE", "ITALY", "RUSSIA", "ENGLAND", "GERMANY", "AUSTRIA", "TURKEY"} - {self.pow_name}}
#         return state
#
#     def _node_personality(self, power: Powers) -> OpponentModelMessage:
#         # Collect messages from that power
#         power = power['power']
#         messages = self._get_message_history(power)
#
#         # Determine which personality fits the power best
#         user_prompt = self._user_prompt_opponent_model.format(
#             self_power=self.pow_name, other_power=power, messages=str(messages)
#         )
#         try:
#             return self.model_opponent_model.invoke(
#                 [SystemMessage(self._system_prompt_opponent_model)] + [HumanMessage(user_prompt)]
#             )
#         except pydantic.ValidationError as e:
#             logger.error(f"Validation error in {self}: {e}. Using defaults.")
#             return OpponentModelMessage(opponent_model="therapist", response_model="therapist")
#
#     def _node_reducer(self, state: OpponentModelMessage) -> AgentState:
#         print(state)
#         return AgentState(current_power=self.pow_name, phase=self.game.get_current_phase())
#
#     def _edge_condition(self, state: AgentState):
#         parallel_nodes = []
#         for power in state.messages_to_send.keys():
#             parallel_nodes.append(Send("personality", {"power": power}))
#         return parallel_nodes
#
#     def _create_messages_agent(self):
#         graph = StateGraph(state_schema=AgentState)
#         graph.add_node("mapper", self._node_mapper)
#         graph.add_node("personality", self._node_personality)
#         graph.add_node("reducer", self._node_reducer)
#         graph.add_edge(START, "mapper")
#         graph.add_conditional_edges("mapper", self._edge_condition, ["personality"])
#         graph.add_edge("personality", "reducer")
#         graph.add_edge("reducer", END)
#         return graph.compile(name=f"DynGenMsg({self.pow_name})")
#
#     def _get_message_history(self, power):
#         messages = defaultdict(list)
#
#         # Collect messages sent to and from the power in the current phase
#         for msg in self.game.get_phase_data().messages.values():
#             if msg.sender == self.pow_name:
#                 messages[msg.recipient].append(f"{msg.sender} said to {msg.recipient} that {msg.message}")
#             elif msg.recipient == self.pow_name:
#                 messages[msg.sender].append(f"{msg.sender} said to {msg.recipient} that {msg.message}")
#
#         return messages[power]
