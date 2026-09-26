from welfare_diplomacy.agents.base_agent import DiplomacyAgent
from welfare_diplomacy.agents.personality_negotiators import PersonalityAgent


def get_class(class_name: str):
    if class_name.lower() == "PersonalityAgent".lower():
        return PersonalityAgent

    raise ValueError(f"Unknown class name: {class_name}")
