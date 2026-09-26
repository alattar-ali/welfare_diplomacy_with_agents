# How to run

start in repository root.

# 1. set up Python

create  virtual environment if you do not already have it, then activate 
and install dependencies:

Linux/macOS
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows/Powershell

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. configure the simulation

Edit `welfare_diplomacy/run_configs/config_personality.yml` to set
`game.max_years` and `game.max_message_rounds` to the values you want.

Set API keys in your terminal before running (do not put them in the YAML config):

Linux/macOS
```bash
export OPENAI_API_KEY="your-openai-api-key"
export WANDB_API_KEY="your-wandb-api-key"
```

Windows/PowerShell
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:WANDB_API_KEY="your-wandb-api-key"
```

`WANDB_API_KEY` is only required when `wandb.disable` is `false`.

## 3. run

Linux/macOS
```bash
cd welfare_diplomacy
PYTHONPATH=.. python simulate.py
```

Windows/Powershell
```
$env:PYTHONPATH=".."
python simulate.py
```

choose a personality for all personality agents when starting a run:

Linux/macOS
```
PYTHONPATH=.. python simulate.py --personality therapist
PYTHONPATH=.. python simulate.py --personality art-of-the-deal
PYTHONPATH=.. python simulate.py --personality back_burner
```

Windows/Powershell
```
$env:PYTHONPATH=".."

python simulate.py --personality therapist
python simulate.py --personality art-of-the-deal
python simulate.py --personality back_burner
```
run one of these commands: `art-of-the-deal` uses an aggressive personality prompt, and
`back_burner` uses a creative personality prompt. Without `--personality`, each country
uses its YAML setting. The option applies only to that run and does not edit
the config file.

## 4. a wandb visualization of the run will open on startup and continously update as the game progresses
