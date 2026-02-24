# welfare-diplomacy-agent

## How to run

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your run

Edit one of the config files in:

`welfare_diplomacy/run_configs/`

Set model/API fields under each player's `agent_params`, for example:

- `api_key`
- `model_provider_url`
- `model`

If you do not want Weights & Biases logging, set:

`wandb.disable: true`

### 4. Choose config file

`welfare_diplomacy/simulate.py` currently loads a config by name from:

```python
CONFIG = "config0"
```

Change this value to one of:

- `config0`
- `config_personality`
- `config_dyn_personality`

### 5. Run simulation

From the repository root:

```bash
cd welfare_diplomacy
PYTHONPATH=.. python simulate.py
```