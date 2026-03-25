<div align="center">

<!-- Logo/Banner placeholder - uncomment and add your image -->
<!-- <img src="assets/banner.png" alt="HyperAgents Banner" width="800"> -->

<h1>HyperAgents</h1>

<p>Self-referential self-improving agents that can optimize for any computable task</p>

<p>
<a href="LICENSE.md"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="License: CC BY-NC-SA 4.0"></a>
<a href="https://arxiv.org/abs/2603.19461"><img src="https://img.shields.io/badge/arXiv-2603.19461-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<a href="https://ai.meta.com/research/publications/hyperagents/"><img src="https://img.shields.io/badge/-Blog-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
<a href="https://x.com/jennyzhangzt/status/2036099935083618487"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>

---

</div>

## Setup

### API Keys

**Option A: Direct API keys** — put these into a `.env` file:
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

**Option B: Claude Code Max subscription (no Anthropic API key needed)**

If you have a Claude Max subscription via [Claude Code](https://claude.ai/code), you can use it directly instead of an Anthropic API key:

```bash
# 1. Install ccproxy
pip install ccproxy-api

# 2. Authenticate with your Claude Code subscription
ccproxy auth login claude_api

# 3. Add to your .env file (ANTHROPIC_API_KEY can be omitted)
ANTHROPIC_AUTH_MODE=oauth
```

At runtime, a local ccproxy instance starts automatically and routes all Anthropic calls through your subscription's OAuth token (stored by Claude Code in `~/.claude/.credentials.json`).

```bash
# Install things (Fedora/RHEL)
sudo dnf install -y python3.12-devel
sudo dnf install -y graphviz graphviz-devel cmake ninja-build bzip2-devel zlib-devel ncurses-devel libffi-devel
```

```bash
# Install things (macOS)
xcode-select --install  # if not already installed
brew install python@3.12 graphviz cmake ninja
```

> **macOS note:** pygraphviz requires pointing pip at the Homebrew graphviz headers:
> ```bash
> pip install pygraphviz \
>   --config-settings=--global-option=build_ext \
>   --config-settings=--global-option="-I$(brew --prefix graphviz)/include" \
>   --config-settings=--global-option="-L$(brew --prefix graphviz)/lib"
> ```
> Run this before `pip install -r requirements.txt`.

```bash
# Create virtual environment
python3.12 -m venv venv_nat
source venv_nat/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
# To build the docker container
docker build --network=host -t hyperagents .
```

```bash
# Setup initial agents
bash ./setup_initial.sh
```

## Running HyperAgents

```bash
# See the script for args, and baseline selections
python generate_loop.py --domains <domain>
```

By default, outputs will be saved in `outputs/` directory.

## File Structure
- `agent/` code for using foundation models
- `analysis/` scripts used for plotting and analysis
- `domains/` code for each domain
- `utils/` common code used in the repo
- `run_meta_agent.py` script to help run the meta agent and get the diffs
- `meta_agent.py` main implementation of the meta agent
- `task_agent.py` main implementation of the task agent
- `generate_loop.py` entry point for running the algorithm

## Logs from Experiments

The experiment logs are stored as a multi-part ZIP archive. To extract them, ensure all .z01, .z02, etc., files are in the same directory as the .zip file, then run:
```bash
zip -s 0 outputs_os_parts.zip --out unsplit_logs.zip
unzip unsplit_outputs.zip
```

## Safety Consideration
> [!WARNING]  
> This repository involves executing untrusted, model-generated code. We strongly advise users to be aware of the associated safety risks. While it is highly unlikely that such code will perform overtly malicious actions under our current settings and with the models we use, it may still behave destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.

## Citing
If you find this project useful, please consider citing:
```bibtex
@misc{zhang2026hyperagents,
      title={Hyperagents}, 
      author={Jenny Zhang and Bingchen Zhao and Wannan Yang and Jakob Foerster and Jeff Clune and Minqi Jiang and Sam Devlin and Tatiana Shavrina},
      year={2026},
      eprint={2603.19461},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.19461}, 
}
```

