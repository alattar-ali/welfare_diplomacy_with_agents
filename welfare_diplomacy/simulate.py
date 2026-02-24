"""
Language model scaffolding to play Diplomacy.
Logs:
- Every message into a W&B Table
- Every power's orders into the SAME W&B Table (one row per power per phase)
- Board renderings to W&B as HTML:
  - state_init, state_pre, with_orders, state_post
"""

import argparse
import pprint
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import wandb
import yaml
from loguru import logger
from rich.progress import Progress
from tqdm import tqdm

import welfare_diplomacy.agents as agents
from diplomacy import Game, Message

CONFIG = "config0"

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


def log_board_to_wandb(game: Game, step: int, tag: str):
    """Logs a rendered HTML board to W&B."""
    html = game.render(incl_abbrev=True)
    wandb.log(
        {
            f"board/{tag}": wandb.Html(html),
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


def main():
    # Load configuration
    with open(f"run_configs/{CONFIG}.yml", "r") as file:
        game_config = yaml.safe_load(file)
    logger.info(f"Loaded game configuration: \n{pprint.pformat(game_config)}")

    # Initialize W&B
    wandb.init(
        entity=game_config["wandb"]["entity"],
        project=game_config["wandb"]["project"],
        dir=game_config["wandb"].get("wandb_dir", None),
        name=game_config["wandb"]["run_name"],
        save_code=game_config["wandb"]["save_code"],
        config=game_config,
        mode="disabled" if game_config["wandb"]["disable"] else "online",
        settings=wandb.Settings(code_dir="."),
    )
    assert wandb.run is not None

    # Initialize data logging
    data_dir = init_data_log_directory(
        run_name=wandb.run.name,
        prefix=Path(game_config["logging"]["output_folder"]).absolute(),
    )
    data = {"responses": []}

    wandb_responses = wandb.Table(columns=RESPONSE_COLUMNS, log_mode="MUTABLE")
    logger.debug(f"Initialized data logging directory: {data_dir}")

    # Initialize game
    game: Game = initialize_game(
        game_config["game"]["map_name"],
        game_config["game"]["max_message_rounds"],
    )
    logger.success(f"Initialized diplomacy game: {game}")

    # Initialize players
    players: Dict[str, agents.DiplomacyAgent] = initialize_players(game, game_config)
    logger.success(f"Players initialized: \n{pprint.pformat(players)}")

    # W&B: initial board
    phase_step = 0
    if not game_config["wandb"]["disable"]:
        log_board_to_wandb(game, step=phase_step, tag="state_init")

    # Run main loop
    with Progress() as progress:
        max_years = game_config["game"]["max_years"]
        progress_phases = progress.add_task("[red]🔄️ Phases...", total=max_years * 3)

        while not game.is_game_done:
            current_phase = game.get_current_phase()
            logger.info(f"🕰️  Beginning phase {current_phase}")

            # Start phase
            for _, agent in players.items():
                agent.start_phase()

            # W&B: pre-phase state
            if not game_config["wandb"]["disable"]:
                log_board_to_wandb(game, step=phase_step, tag="state_pre")

            # Negotiation phase (messages)
            try:
                run_negotiation_phase(
                    game=game,
                    players=players,
                    game_config=game_config,
                    progress=progress,
                    log_dict=data,
                    wandb_log_table=wandb_responses,
                    step=phase_step,
                )
            except Exception as e:
                logger.error(f"💥 Error during negotiation phase: \n{e}\n\n{traceback.format_exc()}")
                break

            # Movement phase (orders + log orders rows)
            try:
                run_movement_phase(
                    game=game,
                    players=players,
                    log_dict=data,
                    wandb_log_table=wandb_responses,
                )
            except Exception as e:
                logger.error(f"💥 Error during movement phase: \n{e}\n\n{traceback.format_exc()}")
                break

            # W&B: board with orders set
            if not game_config["wandb"]["disable"]:
                # Push updated table after orders too (otherwise you'd only see order-rows after next log)
                wandb.log({"responses": wandb_responses}, step=phase_step)
                log_board_to_wandb(game, step=phase_step, tag="with_orders")

            # End phase hooks for agents
            for _, agent in players.items():
                agent.end_phase()

            # Adjudicate
            game.process()

            # W&B: post-adjudication state
            if not game_config["wandb"]["disable"]:
                log_board_to_wandb(game, step=phase_step, tag="state_post")

            # Stop condition by year (safe parse)
            year = _extract_year_from_phase(game.get_current_phase())
            if year is not None and (year - 1900) > game_config["game"]["max_years"]:
                game.finish()

            # Game-level logs (placeholder)
            if not game_config["wandb"]["disable"]:
                update_wandb_game_logs(game, players)

            # Progress + step
            progress.update(progress_phases, advance=1)
            phase_step += 1


def run_negotiation_phase(game, players, game_config, progress, log_dict, wandb_log_table, step: int):
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
        log_dict["responses"].append(dict(zip(RESPONSE_COLUMNS, row)))
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

    except Exception as e:
        logger.exception(f"💥 Error during negotiation phase: \n{e}\n\n{traceback.format_exc()}")

    finally:
        # Upload table at end of negotiation phase
        if not game_config["wandb"]["disable"]:
            wandb.log({"responses": wandb_log_table}, step=step)
        progress.remove_task(progress_message_rounds)


def run_movement_phase(game, players, log_dict, wandb_log_table):
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
        log_dict["responses"].append(dict(zip(RESPONSE_COLUMNS, row)))
        wandb_log_table.add_data(*row)

    return orders


def update_wandb_player_logs(game, power_name, agent, messages):
    pass


def update_internal_player_logs(data, game, power_name, agent, messages):
    pass


def update_wandb_game_logs(game, players):
    pass


def update_internal_game_logs(data, game, players):
    pass


def init_data_log_directory(run_name: str, prefix: Path = Path() / "out", overwrite: bool = False) -> Path:
    """
    :return: Path to the directory where data will be logged.
    """
    current_time = datetime.now()
    dir_name = f"{current_time.strftime('%Y_%m_%d')}_{current_time.strftime('%H_%M')}_{run_name}"
    full_path = prefix / dir_name if prefix else Path(dir_name)

    if full_path.exists():
        if not overwrite:
            raise FileExistsError(f"Directory '{full_path}' already exists and overwrite is set to False.")
        else:
            # NOTE: rmdir only removes empty directories; keep as-is to match your original behavior.
            full_path.rmdir()

    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


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
        power_name_to_agent[name] = agent_cls(game=game, pow_name=name, **params["agent_params"])

    return power_name_to_agent


def parse_args():
    """(unused in this config-driven version)"""
    parser = argparse.ArgumentParser()
    return vars(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        tqdm.write("\n\n\n")
        exception_trace = "".join(traceback.TracebackException.from_exception(exc).format())
        tqdm.write("\n\n\n")
        raise exc