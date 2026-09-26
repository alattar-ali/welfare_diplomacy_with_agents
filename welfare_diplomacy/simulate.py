import argparse
import os
import pprint
import re
import traceback
import webbrowser
from datetime import datetime, timezone
from typing import Dict, Optional

import wandb
import yaml
from loguru import logger
from rich.progress import Progress

import welfare_diplomacy.agents as agents
from diplomacy import Game, Message

CONFIG = "config_personality"

RESPONSE_COLUMNS = [
    "row_type",       # "message" | "orders"
    "phase",
    "time_sent",
    "sender",
    "recipient",
    "message",
    "message_round",
    "orders",
]


def log_board_to_wandb(game: Game, step: int):
    """Log the board with submitted orders before the phase is adjudicated."""
    html = game.render(incl_abbrev=True, incl_orders=True)
    wandb.log(
        {
            "board/with_orders": wandb.Html(html),
            "game/phase": str(game.get_current_phase()),
        },
        step=step,
    )


def _extract_year_from_phase(phase: str) -> Optional[int]:
    """
    Tries to extract a 4-digit year from a Diplomacy phase string.
    Examples: "S1901M", "F1902M", "W1901A", or other formats that include 1901, 1902, etc.
    """
    m = re.search(r"(18|19|20)\d{2}", str(phase))
    return int(m.group(0)) if m else None


def redact_config_secrets(value):
    """Remove API keys from configuration sent to logs and W&B."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key == "api_key" else redact_config_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config_secrets(item) for item in value]
    return value


def main(personality: Optional[str] = None):
    # Show the run link and Rich progress indicators without verbose logs.
    logger.remove()

    # Load configuration
    with open(f"run_configs/{CONFIG}.yml", "r") as file:
        game_config = yaml.safe_load(file)
    if personality is not None:
        for player in game_config["players"].values():
            if player["agent_class"].lower() == "personalityagent":
                player["agent_params"]["personality"] = personality
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set the OPENAI_API_KEY environment variable.")
    if not game_config["wandb"]["disable"] and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Set the WANDB_API_KEY environment variable, or set wandb.disable to true.")

    personalities = {
        player["agent_params"].get("personality", "back_burner")
        for player in game_config["players"].values()
    }
    personality_labels = {
        "therapist": "Therapist", "art-of-the-deal": "Aggressive",
        "back_burner": "Creative",
    }
    run_personality = personality_labels[next(iter(personalities))] if len(personalities) == 1 else "Mixed"
    # Use local time for the run's start date and time.
    game_config["wandb"]["run_name"] = f"{run_personality} Run {datetime.now():%Y-%m-%d %H-%M-%S}"
    log_config = redact_config_secrets(game_config)
    logger.info(f"Loaded game configuration: \n{pprint.pformat(log_config)}")

    # Initialize W&B
    wandb.init(
        entity=game_config["wandb"]["entity"],
        project=game_config["wandb"]["project"],
        dir=game_config["wandb"].get("wandb_dir", None),
        name=game_config["wandb"]["run_name"],
        save_code=game_config["wandb"]["save_code"],
        config=log_config,
        mode="disabled" if game_config["wandb"]["disable"] else "online",
        settings=wandb.Settings(code_dir=".", silent=True),
    )
    assert wandb.run is not None
    if not game_config["wandb"]["disable"] and wandb.run.url:
        run_url = wandb.run.url
        print(f"W&B run: {run_url}", flush=True)
        try:
            webbrowser.open_new_tab(run_url)
        except (webbrowser.Error, OSError):
            # Keep running if no browser is available; the link is printed above.
            pass

    # Initialize the shared messages and orders table.
    wandb_responses = wandb.Table(columns=RESPONSE_COLUMNS, log_mode="MUTABLE")

    # Initialize game
    game: Game = initialize_game(
        game_config["game"]["map_name"],
        game_config["game"]["max_message_rounds"],
    )
    logger.success(f"Initialized diplomacy game: {game}")

    # Initialize players
    players: Dict[str, agents.DiplomacyAgent] = initialize_players(game, game_config)
    logger.success(f"Players initialized: \n{pprint.pformat(players)}")

    phase_step = 0

    # Run main loop
    with Progress() as progress:
        max_years = game_config["game"]["max_years"]
        progress_phases = progress.add_task("[red]🔄️ Phases...", total=max_years * 3)

        while not game.is_game_done:
            current_phase = game.get_current_phase()
            progress.update(progress_phases, description=f"[red]🔄️ Phases ({current_phase})")
            logger.info(f"🕰️  Beginning phase {current_phase}")

            # Start phase
            for _, agent in players.items():
                agent.start_phase()

            # Negotiation phase (messages)
            run_negotiation_phase(
                game=game,
                players=players,
                game_config=game_config,
                progress=progress,
                wandb_log_table=wandb_responses,
                step=phase_step,
            )

            # Movement phase (orders + log orders rows)
            try:
                run_movement_phase(
                    game=game,
                    players=players,
                    wandb_log_table=wandb_responses,
                )
            except Exception as e:
                logger.error(f"💥 Error during movement phase: \n{e}\n\n{traceback.format_exc()}")
                raise

            # W&B: board with orders set
            if not game_config["wandb"]["disable"]:
                # Push updated table after orders too (otherwise you'd only see order-rows after next log)
                wandb.log({"responses": wandb_responses}, step=phase_step)
                log_board_to_wandb(game, step=phase_step)

            # End phase hooks for agents
            for _, agent in players.items():
                agent.end_phase()

            # Adjudicate
            game.process()

            # Stop condition by year (safe parse)
            year = _extract_year_from_phase(game.get_current_phase())
            if year is not None and (year - 1900) > game_config["game"]["max_years"]:
                game.finish()

            # Commit this phase's logs with scores after adjudication.
            if not game_config["wandb"]["disable"]:
                update_wandb_game_logs(game, step=phase_step, phase=current_phase)

            # Progress + step
            progress.update(progress_phases, advance=1)
            phase_step += 1


def run_negotiation_phase(game, players, game_config, progress, wandb_log_table, step: int):
    num_message_rounds = game_config["game"]["max_message_rounds"]
    progress_message_rounds = progress.add_task(
        description="[blue]🙊 Messages",
        total=num_message_rounds * 7,
    )

    def log_message_row(msg: Message, msg_round: int):
        row = [
            "message",
            str(msg.phase),
            str(msg.time_sent),
            str(msg.sender),
            str(msg.recipient),
            str(msg.message),
            str(msg_round),
            "",  # orders empty for message rows
        ]
        wandb_log_table.add_data(*row)

    try:
        for message_round in range(1, num_message_rounds + 1):
            for power_name, agent in players.items():
                # agent decides messages
                messages: dict = agent.generate_messages()

                # send + log each message
                for recipient, message in messages.items():
                    msg = Message(
                        sender=power_name,
                        recipient=recipient,
                        message=message,
                        phase=game.get_current_phase(),
                    )
                    game.add_message(msg)
                    log_message_row(msg, msg_round=message_round)

                progress.update(progress_message_rounds, advance=1)

    except Exception:
        logger.exception("Negotiation failed; stopping the run before orders.")
        raise

    finally:
        # Upload table at end of negotiation phase
        if not game_config["wandb"]["disable"]:
            wandb.log({"responses": wandb_log_table}, step=step)
        progress.remove_task(progress_message_rounds)


def run_movement_phase(game, players, wandb_log_table):
    # collect orders
    orders = {}
    for power_name, agent in players.items():
        try:
            orders[power_name] = agent.generate_orders()
        except Exception as e:
            logger.error(f"💥 Error during movement phase for {power_name}: {e}")
            raise

    # set orders in game
    for power_name, order_list in orders.items():
        game.set_orders(power_name, order_list)

    # log one "orders" row per power (same table)
    phase = str(game.get_current_phase())
    ts = datetime.now(timezone.utc).isoformat()
    for power_name, order_list in orders.items():
        row = [
            "orders",
            phase,
            ts,
            power_name,
            "ORDERS",
            "",
            "",  # message_round empty
            str(order_list),
        ]
        wandb_log_table.add_data(*row)

    return orders


def update_wandb_game_logs(game: Game, *, step: int, phase: str):
    """Log accumulated scores for the phase that just finished."""
    scores = {name: power.welfare_points for name, power in sorted(game.powers.items())}
    table = wandb.Table(columns=["country", "welfare_points"], data=[[name, score] for name, score in scores.items()])
    wandb.log(
        {
            **{f"welfare_points/{name}": score for name, score in scores.items()},
            "welfare_points/total": sum(scores.values()),
            "welfare_points/by_country": wandb.plot.bar(
                table, "country", "welfare_points", title=f"Accumulated welfare points after {phase}",
            ),
            "game/completed_phase": phase,
        },
        step=step,
        commit=True,
    )


def initialize_game(map_name: str, max_message_rounds: int) -> Game:
    game: Game = Game(map_name=map_name)
    if max_message_rounds <= 0:
        game.add_rule("NO_PRESS")
    else:
        game.remove_rule("NO_PRESS")
    return game


def initialize_players(game, game_config):
    assert game_config["players"].keys() == game.powers.keys(), \
        f"Game config has mismatched powers: {game_config['players'].keys()=} & {game.powers.keys()=}"

    power_name_to_agent = {}
    for name, params in game_config["players"].items():
        agent_cls_name = params["agent_class"]
        agent_cls = agents.get_class(agent_cls_name)
        agent_params = dict(params["agent_params"])
        agent_params["game_end_year"] = 1900 + game_config["game"]["max_years"]
        power_name_to_agent[name] = agent_cls(game=game, pow_name=name, **agent_params)

    return power_name_to_agent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a Welfare Diplomacy simulation.")
    parser.add_argument(
        "--personality",
        choices=("therapist", "art-of-the-deal", "back_burner"),
        help="Set all PersonalityAgent players to this personality for this run; defaults to the YAML settings.",
    )
    return vars(parser.parse_args(argv))


if __name__ == "__main__":
    main(**parse_args())
