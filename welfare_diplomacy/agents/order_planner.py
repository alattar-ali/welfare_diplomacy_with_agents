import json
from pathlib import Path
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


class OrderPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orders: list[str] = Field(
        description="Complete orders for this power in this phase. In winter, [] retains units and skips builds when permitted."
    )


class WinterAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["build", "disband", "keep"] = Field(
        description="Choose one available_winter_actions entry before selecting individual orders."
    )


class OrderPlanningError(ValueError):
    """The model did not produce a valid complete order plan."""


class LLMOrderPlanner:
    def __init__(self, model):
        self.model = model
        self.winter_model = model.with_structured_output(WinterAction, method="function_calling")
        self.system_prompt = Path(__file__).with_name("order_system_prompt.txt").read_text()

    def generate_orders(self, game, power_name, *, personality=None, final_year=None):
        context = build_order_context(game, power_name, personality, final_year)
        if not context["legal_orders_by_location"]:
            return []

        if context["phase_type"] == "A":
            action = self._choose_winter_action(context)
            logger.info(f"{power_name} winter action ({context['phase']}): {action}")
            if action == "keep":
                return validate_orders([], context)
            context["winter_action"] = action
            suffix = " B" if action == "build" else " D"
            legal = {
                loc: [order for order in options if order.endswith(suffix)]
                for loc, options in context["legal_orders_by_location"].items()
            }
            context["legal_orders_by_location"] = {loc: options for loc, options in legal.items() if options}
            choices = tuple(sorted({order for options in legal.values() for order in options}))
            limits = context["adjustment_limits"]
            minimum = 0 if action == "build" else limits["minimum_disbands"]
            maximum = (
                limits["maximum_builds"] if action == "build"
                else len(context["legal_orders_by_location"]) if context["welfare"]
                else minimum
            )
            order_schema = create_model(
                "WinterOrderPlan", __base__=OrderPlan,
                orders=(list[Literal[choices]], Field(
                    min_length=minimum, max_length=maximum,
                    description=f"Select {minimum} to {maximum} {action} orders, at most one per province. Omit retained units.",
                )),
            )
        else:

            order_schema = create_model(
                "UnitOrderPlan", __config__=ConfigDict(extra="forbid"),
                **{
                    loc: (Literal[tuple(options)], Field(description=f"One legal order for the unit at {loc}."))
                    for loc, options in context["legal_orders_by_location"].items()
                },
            )
        order_model = self.model.with_structured_output(
            order_schema, method="function_calling", strict=True,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=json.dumps(context)),
        ]
        # one initial call plus one repair attempt
        for attempt in range(2):
            plan = None
            try:
                plan = order_schema.model_validate(order_model.invoke(messages))
                orders = plan.orders if context["phase_type"] == "A" else list(plan.model_dump().values())
                orders = validate_orders(orders, context)
                logger.info(f"{power_name} orders ({context['phase']}): {orders}")
                return orders
            except (ValidationError, OutputParserException, OrderPlanningError) as exc:
                if attempt == 1:
                    raise OrderPlanningError(
                        f"{power_name} could not produce valid orders for {context['phase']} after one repair: {exc}"
                    ) from exc
                logger.warning(f"Invalid orders for {power_name}; requesting one repair: {exc}")
                messages.append(HumanMessage(content=json.dumps({
                    "previous_orders": plan.model_dump() if plan is not None else None,
                    "validation_error": str(exc),
                    "instruction": f"Return a corrected, complete {order_schema.__name__} using the original context and legal orders.",
                })))

    def _choose_winter_action(self, context):
        limits = context["adjustment_limits"]
        if limits["minimum_disbands"] > 0:
            return "disband"

        available = ["keep"]
        options = [order for orders in context["legal_orders_by_location"].values() for order in orders]
        if limits["maximum_builds"] > 0 and any(order.endswith(" B") for order in options):
            available.append("build")
        if context["welfare"] and any(order.endswith(" D") for order in options):
            available.append("disband")
        if len(available) == 1:
            return "keep"

        messages = [
            SystemMessage(content=self.system_prompt + (
                "\nChoose only a WinterAction from available_winter_actions now. "
                "Individual orders will be selected afterward."
            )),
            HumanMessage(content=json.dumps({**context, "available_winter_actions": available})),
        ]
        for attempt in range(2):
            choice = None
            try:
                choice = WinterAction.model_validate(self.winter_model.invoke(messages))
                if choice.action not in available:
                    raise OrderPlanningError(f"Choose one of the available winter actions: {available}")
                return choice.action
            except (ValidationError, OutputParserException, OrderPlanningError) as exc:
                if attempt == 1:
                    raise OrderPlanningError(
                        f"{context['power']} could not choose a valid winter action for {context['phase']} after one repair: {exc}"
                    ) from exc
                logger.warning(f"Invalid winter action for {context['power']}; requesting one repair: {exc}")
                messages.append(HumanMessage(content=json.dumps({
                    "previous_action": choice.action if choice is not None else None,
                    "validation_error": str(exc),
                    "instruction": "Return a corrected WinterAction from available_winter_actions, without orders.",
                })))


def get_conversation_history(game, power_name):
    """Read full message history"""
    phases = list(game.message_history.items())
    phases.append((game.get_current_phase(), game.messages))
    return [
        {
            "phase": str(phase), "time_sent": msg.time_sent,
            "sender": msg.sender, "recipient": msg.recipient, "message": msg.message,
        }
        for phase, messages in phases
        for msg in messages.values()
        if msg.sender == power_name or msg.recipient in (power_name, "GLOBAL")
    ]


def build_order_context(game, power_name, personality=None, final_year=None):
    state = game.get_state()
    power = game.get_power(power_name)
    possible_orders = game.get_all_possible_orders()
    legal = {
        loc: sorted(order for order in possible_orders.get(loc, []) if order != "WAIVE")
        for loc in game.get_orderable_locations(power_name)
    }
    legal = {loc: options for loc, options in legal.items() if options}
    return {
        "power": power_name,
        "personality": personality or "standard",
        "personality_style": {
            "therapist": "Empathetic, relationship-oriented negotiation with attention to your country's interests.",
            "art-of-the-deal": "Assertive, competitive negotiation seeking favorable reciprocal commitments.",
            "back_burner": "Creative negotiation using alternative deals and phased cooperation.",
        }.get(personality, "Practical diplomacy in your country's interests."),
        "phase": game.get_current_phase(),
        "phase_type": game.phase_type,
        "map": game.map.name,
        "welfare": game.welfare,
        "final_year": final_year,
        "board": {key: state[key] for key in ("units", "centers", "homes", "welfare_points")},
        "own_retreat_options": state["retreats"][power_name],
        "adjustment_limits": {
            "maximum_builds": max(0, state["builds"][power_name]["count"]),
            "minimum_disbands": max(0, len(power.units) - len(power.centers)) if game.phase_type == "A" else 0,
        },
        "conversation": get_conversation_history(game, power_name),
        "legal_orders_by_location": legal,
    }


def validate_orders(orders, context):
    orders = [" ".join(order.upper().split()) for order in orders]
    legal = context["legal_orders_by_location"]
    selected_locations = set()
    for order in orders:
        parts = order.split()
        location = parts[1][:3] if len(parts) > 1 else None
        if location not in legal or order not in legal[location]:
            raise OrderPlanningError(f"Order is not in your legal options: {order!r}")
        if location in selected_locations:
            raise OrderPlanningError(f"Multiple orders for province {location}")
        selected_locations.add(location)

    if context["phase_type"] in ("M", "R"):
        missing = set(legal) - selected_locations
        if missing:
            raise OrderPlanningError(f"Missing orders for: {', '.join(sorted(missing))}")
    elif context["phase_type"] == "A":
        builds = sum(order.endswith(" B") for order in orders)
        disbands = sum(order.endswith(" D") for order in orders)
        limits = context["adjustment_limits"]
        if builds and disbands:
            raise OrderPlanningError("Choose either builds or disbands this winter, not both")
        if builds > limits["maximum_builds"]:
            raise OrderPlanningError(f"At most {limits['maximum_builds']} builds are allowed")
        if disbands < limits["minimum_disbands"]:
            raise OrderPlanningError(f"At least {limits['minimum_disbands']} disbands are required")
        if not context["welfare"] and disbands > limits["minimum_disbands"]:
            raise OrderPlanningError("Voluntary disbands require the Welfare rule")
    return orders
