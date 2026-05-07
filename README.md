# EvoSR-LLM

Official code for **EvoSR-LLM**, an LLM-assisted evolutionary symbolic regression framework.

This repository contains the public implementation of EvoSR-LLM and the datasets used in our experiments.

## Requirements

- Python 3.10+

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
.
├── main.py
├── algorithm/
├── population/
├── Problems/
├── llm/
├── utils/
├── oes_data/
└── llm_srbench_data/
```

## Datasets

This repository includes two benchmark groups:

- `oes`
- `llm_srbench`

Each dataset contains:

- `train.csv`
- `test_id.csv`
- `test_ood.csv`

## Usage

### Run with a remote LLM API

```bash
export API_KEY=YOUR_API_KEY

python main.py \
  --benchmark oes \
  --problem-name oscillator1 \
  --llm-model YOUR_MODEL_NAME \
  --llm-api-endpoint YOUR_API_ENDPOINT
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@ARTICLE{li26evosr,
  author = {Li, Yuchen and Wang, Handing and Jin, Yaochu},
  journal = {IEEE Transactions on Evolutionary Computation},
  title = {EvoSR-LLM: Evolutionary Symbolic Regression Guided by Large Language Models},
  year = {2026},
  volume = {},
  number = {},
  pages = {1-1},
  doi = {10.1109/TEVC.2026.3689815}
}
```
