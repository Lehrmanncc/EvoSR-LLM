#!/usr/bin/env python3
"""Entry point for running EvoSR-LLM experiments."""
import argparse
import os
import sys
from datetime import datetime


def _add_repo_root_to_path() -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EvoSR-LLM on a selected dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--benchmark", default="oes", help="Benchmark name, e.g. oes or llm_srbench")
    parser.add_argument("--problem-name", required=True, help="Problem name, e.g. oscillator1, bio")
    parser.add_argument("--ins-idx", type=int, default=None, help="Instance index for llm_srbench")

    spec_group = parser.add_mutually_exclusive_group()
    spec_group.add_argument("--problem-spec", dest="problem_spec", action="store_true", default=True,
                            help="Use spec files to name variables")
    spec_group.add_argument("--no-problem-spec", dest="problem_spec", action="store_false",
                            help="Ignore spec files and use generic variable names")

    parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="Remote LLM model name")
    parser.add_argument("--llm-api-endpoint", default="aihubmix.com", help="Remote LLM endpoint")
    parser.add_argument("--llm-api-key", default=None, help="Remote LLM API key")
    parser.add_argument("--llm-api-key-env", default="API_KEY", help="Env var name for API key")
    parser.add_argument("--llm-use-local", action="store_true", help="Use a local LLM server")
    parser.add_argument("--llm-local-url", default=None, help="Local LLM server URL")

    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--offspring-size", type=int, default=2)
    parser.add_argument("--max-fe", type=int, default=3000)
    parser.add_argument("--operators-gen-num", type=int, default=120)
    parser.add_argument("--n-process", type=int, default=4)
    parser.add_argument("--lamda", type=float, default=0.00001)
    parser.add_argument("--alpha", type=float, default=5.0)

    parser.add_argument("--exp-debug-mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--exp-output-path", default=None, help="Output directory root")

    return parser.parse_args()


def _resolve_api_key(args: argparse.Namespace) -> str:
    if args.llm_use_local:
        return None
    if args.llm_api_key:
        return args.llm_api_key
    if args.llm_api_key_env:
        key = os.getenv(args.llm_api_key_env)
        if key:
            return key
    raise SystemExit(
        "No API key found. Use --llm-api-key or set the env var via --llm-api-key-env."
    )


def main() -> None:
    _add_repo_root_to_path()
    args = _parse_args()

    from algorithm.sr_evol import SrEvol
    from utils.util import Paras
    from Problems.problems import ProblemSR

    ins_idx = args.ins_idx
    if args.benchmark == "llm_srbench" and ins_idx is None:
        ins_idx = 0

    api_key = _resolve_api_key(args)

    if args.exp_output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_output_path = os.path.join(".", "results", timestamp)

    paras = Paras()
    paras.set_paras(
        benchmark=args.benchmark,
        llm_use_local=args.llm_use_local,
        llm_local_url=args.llm_local_url,
        llm_api_endpoint=args.llm_api_endpoint,
        llm_api_key=api_key,
        llm_model=args.llm_model,
        pop_size=args.pop_size,
        offspring_size=args.offspring_size,
        max_fe=args.max_fe,
        operators_gen_num=args.operators_gen_num,
        n_process=args.n_process,
        exp_debug_mode=args.exp_debug_mode,
        lamda=args.lamda,
        alpha=args.alpha,
        exp_output_path=args.exp_output_path,
    )

    sr_problem = ProblemSR(args.benchmark, args.problem_name, ins_idx, problem_spec=args.problem_spec)
    evolution = SrEvol(paras, sr_problem)
    evolution.run()


if __name__ == "__main__":
    main()
