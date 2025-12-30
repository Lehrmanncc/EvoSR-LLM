"""Plot lambda & complexity evolution from evosr-llm_results samples.

This script scans all model folders under `evosr-llm_results/**` and finds
all `samples/` directories (including nested ones like `bio/ins0/samples`).

It extracts:
- `lamda` (historical spelling in this repo; also tries `lambda`)
- `complex`

If a metric is missing or None in a given sample, the previous value is kept.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


_SAMPLES_FE_RE = re.compile(r"^samples_fe=(\d+)\.json$")


def _extract_fe(path: Path) -> Optional[int]:
    match = _SAMPLES_FE_RE.match(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_lambda(payload: dict) -> Optional[float]:
    # Historical spelling in this repo appears to be 'lamda'
    value = payload.get("lamda")
    if value is None:
        value = payload.get("lambda")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_complex(payload: dict) -> Optional[float]:
    value = payload.get("complex")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(name: str) -> str:
    # Replace path separators and other problematic characters.
    safe = name.replace("/", "__").replace(os.sep, "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe.strip("._-") or "unnamed"


def _print_metric_statistics(metric_data: Dict[str, Dict[str, List[Tuple[int, float]]]], title: str, fmt: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for model_name in sorted(metric_data.keys()):
        print(f"\n{model_name}:")
        print("-" * 60)

        for problem_name in sorted(metric_data[model_name].keys()):
            series = metric_data[model_name][problem_name]
            if not series:
                continue
            values = [v for _, v in series]
            fes = [fe for fe, _ in series]

            print(f"  {problem_name}:")
            print(f"    Samples collected: {len(series)}")
            print(f"    FE range: {min(fes)} - {max(fes)}")
            print(
                f"    Value - Min: {format(min(values), fmt)}, "
                f"Max: {format(max(values), fmt)}, "
                f"Mean: {format(float(np.mean(values)), fmt)}"
            )
            print(f"    Value - Std: {format(float(np.std(values)), fmt)}")


def load_lambda_and_complex_data(
    results_dir: str,
) -> Tuple[Dict[str, Dict[str, List[Tuple[int, float]]]], Dict[str, Dict[str, List[Tuple[int, float]]]]]:
    lambda_data: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    complex_data: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        model_name = model_dir.name
        lambda_data[model_name] = {}
        complex_data[model_name] = {}

        for samples_dir in model_dir.rglob("samples"):
            if not samples_dir.is_dir():
                continue

            sample_files = [p for p in samples_dir.iterdir() if p.is_file() and _extract_fe(p) is not None]
            if not sample_files:
                continue

            problem_key = str(samples_dir.parent.relative_to(model_dir)).replace(os.sep, "/")

            fe_lambda_pairs: List[Tuple[int, float]] = []
            fe_complex_pairs: List[Tuple[int, float]] = []
            last_lambda: Optional[float] = None
            last_complex: Optional[float] = None

            for sample_file in sorted(sample_files, key=lambda p: _extract_fe(p) or -1):
                fe = _extract_fe(sample_file)
                if fe is None:
                    continue

                try:
                    with open(sample_file, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Warning: Error reading {sample_file}: {e}")
                    continue

                current_lambda = _extract_lambda(data)
                if current_lambda is None:
                    current_lambda = last_lambda
                if current_lambda is not None:
                    last_lambda = current_lambda
                    fe_lambda_pairs.append((fe, current_lambda))

                current_complex = _extract_complex(data)
                if current_complex is None:
                    current_complex = last_complex
                if current_complex is not None:
                    last_complex = current_complex
                    fe_complex_pairs.append((fe, current_complex))

            if fe_lambda_pairs:
                fe_lambda_pairs.sort(key=lambda x: x[0])
                lambda_data[model_name][problem_key] = fe_lambda_pairs

            if fe_complex_pairs:
                fe_complex_pairs.sort(key=lambda x: x[0])
                complex_data[model_name][problem_key] = fe_complex_pairs

    return lambda_data, complex_data


def load_lambda_data(results_dir: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Load lambda values from samples files organized by model type.
    
    Args:
        results_dir: Path to the evosr-llm_results directory
        
    Returns:
        Dictionary mapping model names to lists of (fe, lambda) tuples
    """
    lambda_data, _ = load_lambda_and_complex_data(results_dir)
    return lambda_data


def fill_missing_values(fe_lambda_pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Backward-compatible alias.

    The loader already forward-fills missing/None lambda values per existing samples.
    This function now just returns the input unchanged.
    """
    return fe_lambda_pairs


def plot_lambda_and_complex_together(
    lambda_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
    complex_data: Dict[str, Dict[str, List[Tuple[int, float]]]],
    output_dir: str = None,
):
    """For each (model, problem), plot lambda and complex on one figure.

    Uses a twin y-axis:
    - left: lambda
    - right: complexity
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for model_name in sorted(set(lambda_data.keys()) | set(complex_data.keys())):
        problems = set(lambda_data.get(model_name, {}).keys()) | set(complex_data.get(model_name, {}).keys())
        for problem in sorted(problems):
            lam_series = lambda_data.get(model_name, {}).get(problem, [])
            comp_series = complex_data.get(model_name, {}).get(problem, [])
            if not lam_series and not comp_series:
                continue

            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax2 = ax1.twinx()

            handles = []
            labels = []

            if lam_series:
                fes_l, vals_l = zip(*fill_missing_values(lam_series))
                (line_l,) = ax1.plot(
                    fes_l,
                    vals_l,
                    color="tab:blue",
                    linewidth=2,
                    label="lambda",
                )
                ax1.set_ylabel("Lambda (λ)", color="tab:blue", fontsize=12)
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                handles.append(line_l)
                labels.append("lambda")
            else:
                ax1.set_ylabel("Lambda (λ)", fontsize=12)

            if comp_series:
                fes_c, vals_c = zip(*fill_missing_values(comp_series))
                (line_c,) = ax2.plot(
                    fes_c,
                    vals_c,
                    color="tab:orange",
                    linewidth=2,
                    label="complex",
                )
                ax2.set_ylabel("Complexity", color="tab:orange", fontsize=12)
                ax2.tick_params(axis="y", labelcolor="tab:orange")
                handles.append(line_c)
                labels.append("complex")
            else:
                ax2.set_ylabel("Complexity", fontsize=12)

            ax1.set_xlabel("Function Evaluations (fe)", fontsize=12)
            fig.suptitle(f"Lambda & Complexity - {model_name} - {problem}", fontsize=14, fontweight="bold")
            ax1.grid(True, alpha=0.3)

            if handles:
                ax1.legend(handles, labels, fontsize=10, loc="best")

            plt.tight_layout()

            if output_dir:
                safe_model = _safe_filename(model_name)
                safe_problem = _safe_filename(problem)
                filepath = os.path.join(output_dir, f"lambda_complex_{safe_model}__{safe_problem}.png")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                print(f"Saved: {filepath}")
                plt.close(fig)
            else:
                plt.show()


def plot_lambda_evolution(lambda_data: Dict[str, Dict[str, List[Tuple[int, float]]]], 
                         output_dir: str = None):
    """
    Create plots showing lambda evolution for different models and problems.
    
    Args:
        lambda_data: Dictionary of lambda data organized by model and problem
        output_dir: Directory to save plots (if None, displays plots)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get unique problems across all models
    all_problems = set()
    for model_data in lambda_data.values():
        all_problems.update(model_data.keys())
    
    all_problems = sorted(all_problems)
    
    # Create a figure for each problem, showing all models
    for problem in all_problems:
        plt.figure(figsize=(12, 7))
        
        for model_name in sorted(lambda_data.keys()):
            if problem in lambda_data[model_name]:
                fe_lambda_pairs = lambda_data[model_name][problem]
                filled_pairs = fill_missing_values(fe_lambda_pairs)
                
                if filled_pairs:
                    fes, lambdas = zip(*filled_pairs)
                    plt.plot(fes, lambdas, label=model_name, linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Function Evaluations (fe)', fontsize=12)
        plt.ylabel('Lambda (λ)', fontsize=12)
        plt.title(f'Lambda Evolution - {problem}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_dir:
            safe_problem = _safe_filename(problem)
            filepath = os.path.join(output_dir, f'lambda_evolution_{safe_problem}.png')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()
        else:
            plt.show()
    
    # Create a combined figure showing all problems for each model
    for model_name in sorted(lambda_data.keys()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, problem in enumerate(sorted(lambda_data[model_name].keys())[:4]):
            if problem in lambda_data[model_name]:
                fe_lambda_pairs = lambda_data[model_name][problem]
                filled_pairs = fill_missing_values(fe_lambda_pairs)
                
                if filled_pairs:
                    fes, lambdas = zip(*filled_pairs)
                    axes[idx].plot(fes, lambdas, color='steelblue', linewidth=2)
                    axes[idx].fill_between(fes, lambdas, alpha=0.3, color='steelblue')
                    axes[idx].set_xlabel('Function Evaluations (fe)', fontsize=10)
                    axes[idx].set_ylabel('Lambda (λ)', fontsize=10)
                    axes[idx].set_title(problem, fontsize=11, fontweight='bold')
                    axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(lambda_data[model_name]), 4):
            axes[idx].axis('off')
        
        fig.suptitle(f'Lambda Evolution - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir:
            safe_model = _safe_filename(model_name)
            filepath = os.path.join(output_dir, f'lambda_evolution_{safe_model}.png')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()
        else:
            plt.show()


def print_statistics(lambda_data: Dict[str, Dict[str, List[Tuple[int, float]]]]):
    """
    Print statistics about lambda values.
    
    Args:
        lambda_data: Dictionary of lambda data organized by model and problem
    """
    print("\n" + "="*80)
    print("Lambda Statistics")
    print("="*80)
    
    for model_name in sorted(lambda_data.keys()):
        print(f"\n{model_name}:")
        print("-" * 60)
        
        for problem_name in sorted(lambda_data[model_name].keys()):
            fe_lambda_pairs = lambda_data[model_name][problem_name]
            
            if not fe_lambda_pairs:
                continue
            
            lambdas = [pair[1] for pair in fe_lambda_pairs]
            fes = [pair[0] for pair in fe_lambda_pairs]
            
            print(f"  {problem_name}:")
            print(f"    Samples collected: {len(fe_lambda_pairs)}")
            print(f"    FE range: {min(fes)} - {max(fes)}")
            print(f"    Lambda - Min: {min(lambdas):.6f}, Max: {max(lambdas):.6f}, Mean: {np.mean(lambdas):.6f}")
            print(f"    Lambda - Std: {np.std(lambdas):.6f}")


if __name__ == "__main__":
    # Path to the evosr-llm_results directory
    results_dir = "/Users/lyc/Work/Src/EvoSR-LLM/code/evosr-llm_results"
    
    # Load data
    print("Loading lambda/complex data from evosr-llm_results...")
    lambda_data, complex_data = load_lambda_and_complex_data(results_dir)

    # Print statistics
    _print_metric_statistics(lambda_data, title="Lambda Statistics", fmt=".6f")
    _print_metric_statistics(complex_data, title="Complexity Statistics", fmt=".2f")
    
    # Create plots
    print("\n" + "="*80)
    print("Creating plots...")
    print("="*80)
    
    output_dir = "./lambda_plots"
    plot_lambda_and_complex_together(lambda_data, complex_data, output_dir)
    
    print("\n✓ All plots have been generated!")
