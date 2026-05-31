# solarwind_stats

Statistical analysis pipeline for in-situ solar wind data from Parker Solar
Probe (PSP), Solar Orbiter (SolO), and ACE.

## Layout

```
solarwind_stats/
├── swpipeline/    # data download, cleaning, physics, statistics
└── tests/         # pytest suite
```

`swpipeline/` is the reusable pipeline package. Analysis code that builds on
top of it lives elsewhere — this repo only ships the pipeline.

## swpipeline modules

| Module          | Purpose                                                  |
| --------------- | -------------------------------------------------------- |
| `config.py`     | CDAWeb dataset IDs, raw column names, physical constants |
| `download.py`   | Fetch PSP/SolO/ACE CDFs via SunPy/Fido                   |
| `process.py`    | Clean, resample, and merge mag + plasma datasets         |
| `physics.py`    | Plasma beta, Alfvén speed, Mach number, etc.             |
| `stats.py`      | Distance binning, power-law fits, block bootstrap        |
| `trajectory.py` | Spacecraft heliocentric positions via SPICE              |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.12.

## Run the tests

```bash
pytest tests/ -v
```
