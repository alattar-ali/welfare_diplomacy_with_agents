import json
import random
from typing import Dict, List, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from pydantic import BaseModel, Field, ConfigDict

import diplomacy
from welfare_diplomacy.agents.base_agent import DiplomacyAgent

Powers = Literal["FRANCE", "ITALY", "RUSSIA", "ENGLAND", "GERMANY", "AUSTRIA", "TURKEY"]


class OutgoingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipient: Powers
    message: str


class NegotiationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages_to_send: List[OutgoingMessage] = Field(default_factory=list)


class AgentState(BaseModel):
    current_power: str
    phase: str
    received_messages: Dict[str, List[str]] = Field(default_factory=dict)
    messages_to_send: Dict[Powers, str] = Field(default_factory=dict)


class WDAgent(DiplomacyAgent):
    def __init__(self, game: diplomacy.Game, pow_name: str, **params):
        super().__init__(game, pow_name, **params)

        self._base_url = params["model_provider_url"]
        self._api_key = params["api_key"]
        self._model_name = params["model"]

        if self._api_key == "docker":
            self._model_name = "ai/" + params["model_name"]

        self.model = ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model_name,
        )

        # Structured output model (must be schema-compatible)
        self.model_message = self.model.with_structured_output(NegotiationMessage)

        self.generate_messages_agent = self.create_messages_agent()

    def create_messages_agent(self):
        graph = StateGraph(state_schema=AgentState)
        graph.add_node("chatbot", self._node_chatbot)
        graph.add_edge(START, "chatbot")
        return graph.compile()

    def generate_messages(self):
        state = AgentState(
            current_power=self.pow_name,
            phase=self.game.get_current_phase(),
            received_messages={
                "FRANCE": ["Let's work together against AUS."],
                "GERMANY": ["Can I trust FRA?"],
                "AUSTRIA": ["Peace in the south?"],
            },
        )

        new_state = self.generate_messages_agent.invoke(state)
        msg_dict = new_state.get("messages_to_send", {})  # Dict[Powers, str]
        msg_dict = {
        p: m.strip()
            for p, m in msg_dict.items()
            if p != self.pow_name and isinstance(m, str) and m.strip()
        }

        print(self.pow_name, new_state["messages_to_send"])
        
        return msg_dict

    def generate_orders(self):
        orderable_locations = self.game.get_orderable_locations(self.pow_name)
        orders = []
        possible_orders = self.game.get_all_possible_orders()

        for location in orderable_locations:
            if possible_orders.get(location):
                orders.append(random.choice(possible_orders[location]))
        return orders

    def _node_chatbot(self, state: AgentState):
        system_prompt = f"""
You are a skilled agent playing the board game Diplomacy.
You control the power of {self.pow_name}.
Phase: {self.game.get_current_phase()}.

Return ONLY JSON in this exact shape:
{{
  "messages_to_send": [
    {{"recipient": "GERMANY", "message": "..." }},
    {{"recipient": "AUSTRIA", "message": "..." }}
  ]
}}

- recipient must be one of: FRANCE, ITALY, RUSSIA, ENGLAND, GERMANY, AUSTRIA, TURKEY
- Do NOT include {self.pow_name} as a recipient.
- No extra keys. No extra text.
""".strip()

        user_prompt = json.dumps(state.model_dump())

        parsed: NegotiationMessage = self.model_message.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        # LangGraph node must return state updates (dict), not the Pydantic object
        msg_dict: Dict[Powers, str] = {m.recipient: m.message for m in parsed.messages_to_send}
        return {"messages_to_send": msg_dict}