import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from typing import List, Dict, Callable
import argparse

# ------------------------ Filename index extraction ------------------------ #

def extract_index(filename: str) -> int:
    """Extract numeric iteration index from various filename patterns.
    Supported patterns:
      samples_fe=123.json, samples_123.json, sample_123.json
    Returns -1 if no pattern matched (these files will be sorted last).
    """
    import re
    patterns = [
        r"samples_fe=(\d+)\.json",
        r"samples_(\d+)\.json",
        r"sample_(\d+)\.json",
    ]
    for p in patterns:
        m = re.match(p, filename)
        if m:
            return int(m.group(1))
    return -1

# ------------------------ EvoSR-LLM specific helpers ---------------------- #

def get_pop(data):
    data_array = np.array(data)
    if data_array.size == 0:
        return []
    sorted_arr = data_array[data_array[:, 0].argsort()]
    return sorted_arr[:10].tolist() if len(sorted_arr) >= 10 else sorted_arr.tolist()


def change_obj(pop, new_lambda):
    for ind in pop:
        # ind = [objective_like, mse, S, lambda, train_nmse]
        if ind[1] is not None and ind[2] is not None:
            ind[0] = ind[1] + new_lambda * ind[2]
    return pop


def get_best_train_nmse(pop):
    try:
        pop_array = np.asarray(pop)
        sorted_arr = pop_array[pop_array[:, 0].argsort()]
        train_nmse = sorted_arr[0][-1]
    except Exception as e:
        print("Error occurred while getting best train nmse:", e)
        return None
    return train_nmse

def load_evosr_curve(samples_dir: str) -> List[float]:
    """Reproduce original convergence logic for EvoSR-LLM results.
    Each JSON may have: objective, mse, S, lamda, train_nmse. We rebuild running best train_nmse.
    """
    if not os.path.isdir(samples_dir):
        return []
    files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
    files_sorted = sorted(files, key=extract_index)

    data = []  # population buffer
    nmse_running = []
    lambda_hist = []
    init_flag = None

    for i, fname in enumerate(files_sorted):
        fpath = os.path.join(samples_dir, fname)
        try:
            with open(fpath, 'r') as f:
                js = json.load(f)
        except Exception:
            # replicate previous best if read fails
            if nmse_running:
                nmse_running.append(nmse_running[-1])
            continue

        obj = js.get('objective')
        lam = js.get('lamda')
        mse = js.get('mse')
        S = js.get('S')
        train_nmse = js.get('train_nmse')

        if init_flag is None and obj is not None:
            data.append([obj, mse, S, lam, train_nmse])
            lambda_hist.append(lam)
            best_train_nmse = train_nmse
            if best_train_nmse is not None:
                nmse_running.append(best_train_nmse)
            else:
                nmse_running.append(np.inf)
            init_flag = True

        if all(v is not None for v in [obj, mse, S, lam, train_nmse]) and init_flag:
            # lambda schedule change
            if lam != lambda_hist[-1]:
                lambda_hist = [lam]
                pop = get_pop(data)
                pop = change_obj(pop, lam)
                pop.append([obj, mse, S, lam, train_nmse])
                data = pop
            else:
                data.append([obj, mse, S, lam, train_nmse])

            best_train_nmse = get_best_train_nmse(data)
            nmse_running.append(best_train_nmse if best_train_nmse is not None else nmse_running[-1])
        else:
            # fallback: repeat last best
            if nmse_running:
                nmse_running.append(nmse_running[-1])

    return nmse_running

# ------------------------ Baseline loaders (LLM-SR, FunSearch, DrSR) ------- #

def load_baseline_curve(samples_dir: str) -> List[float]:
    """Generic loader for baseline algorithms whose JSON contain train_nmse.
    We accumulate the running minimum of train_nmse (normalized MSE).
    """
    if not os.path.isdir(samples_dir):
        return []
    files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
    files_sorted = sorted(files, key=extract_index)

    best_list = []
    current_best = np.inf
    for fname in files_sorted:
        fpath = os.path.join(samples_dir, fname)
        try:
            with open(fpath, 'r') as f:
                js = json.load(f)
            train_nmse = js.get('train_nmse')
            if train_nmse is not None:
                current_best = min(current_best, train_nmse)
            best_list.append(current_best)
        except Exception:
            if best_list:
                best_list.append(best_list[-1])
    return best_list

# ------------------------ Dispatcher --------------------------------------- #

ALGO_LOADERS: Dict[str, Callable[[str], List[float]]] = {
    'evosr-llm': load_evosr_curve,
    'llm_sr': load_baseline_curve,
    'funsearch': load_baseline_curve,
    'drsr': load_baseline_curve,
}

# ------------------------ Helper function for single problem plot --------- #

def plot_single_problem(ax, problem, model, algorithms, evosr_base, baselines_base, 
                       title_map, color_cycle, style_cycle):
    """Plot convergence curves for a single problem and model."""
    y_min_global, y_max_global = np.inf, -np.inf
    
    # Algorithm name mapping for legend
    algo_labels = {
        'evosr-llm': 'EvoSR-LLM',
        'llm_sr': 'LLM-SR',
        'funsearch': 'FunSearch',
        'drsr': 'DrSR'
    }

    for algo in algorithms:
        loader = ALGO_LOADERS.get(algo)
        if loader is None:
            continue

        # Load curve for single model
        if algo == 'evosr-llm':
            samples_dir = os.path.join(evosr_base, model, problem, 'samples')
        else:
            samples_dir = os.path.join(baselines_base, algo, model, problem, 'samples')
        
        curve = loader(samples_dir)
        if not curve:
            continue

        # --- sanitize curve (remove/replace inf, nan, non-positive for log) ---
        curve_arr = np.asarray(curve, dtype=float)
        finite_pos_mask = np.isfinite(curve_arr)
        if not np.any(finite_pos_mask):
            # all invalid -> skip this algorithm for this problem
            continue
        # Replace leading invalids with first finite; others with previous finite (forward fill)
        first_finite_val = curve_arr[finite_pos_mask][0]
        for idx in range(len(curve_arr)):
            if not np.isfinite(curve_arr[idx]):
                curve_arr[idx] = first_finite_val
            else:
                first_finite_val = curve_arr[idx]

        # For log scale we need positive numbers; replace non-positive with smallest positive * 0.1
        positive_mask = curve_arr > 0
        if not np.any(positive_mask):
            # cannot plot on log scale; skip
            continue
        min_positive = np.min(curve_arr[positive_mask])
        curve_arr[~positive_mask] = min_positive * 0.1

        # update stats using sanitized curve
        curve_min = float(np.min(curve_arr))
        curve_max = float(np.max(curve_arr))
        if np.isfinite(curve_min) and curve_min > 0:
            y_min_global = min(y_min_global, curve_min)
        if np.isfinite(curve_max):
            y_max_global = max(y_max_global, curve_max)

        xs = np.arange(1, len(curve_arr) + 1)
        ax.plot(xs, curve_arr,
                label=algo_labels.get(algo, algo),
                color=color_cycle.get(algo, None),
                linestyle=style_cycle.get(algo, '-'),
                linewidth=3.5)  # 进一步增加线宽 (2.5 -> 3.5)
    
    ax.set_xlabel('Number of Evaluations', fontsize=28)  # 进一步增大 (20 -> 28)
    ax.set_ylabel('Normalized MSE', fontsize=28)        # 进一步增大 (20 -> 28)
    ax.grid(True, linestyle=':', linewidth=1.0)         # 增加网格线宽度 (0.7 -> 1.0)

    # Setup log scale only if we have valid finite positive bounds
    if np.isfinite(y_min_global) and np.isfinite(y_max_global) and y_min_global > 0 and y_max_global > 0 and y_max_global >= y_min_global:
        try:
            log_min = int(np.floor(np.log10(y_min_global)))
            log_max = int(np.ceil(np.log10(y_max_global)))
            if np.isfinite(log_min) and np.isfinite(log_max):
                ax.set_yscale('log')
                yticks = [10 ** i for i in range(log_min, log_max + 1) if (log_max - log_min) <= 6 or i % 2 != 0]
                ax.set_yticks(yticks)
                ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
        except OverflowError:
            # fallback: no log scaling if still problematic
            pass

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)  # 增加边框线宽度 (1.5 -> 2.0)

# ------------------------ Plotting ---------------------------------------- #

def plot_multi_convergence(models: List[str],
                           problems: List[str],
                           algorithms: List[str],
                           evosr_base: str = './evosr-llm_results',
                           baselines_base: str = './baselines_results',
                           out_path: str = './results_final/convergence_curve_multi.pdf',
                           title_map: Dict[str, str] = None,
                           model_map: Dict[str, str] = None,
                           cols: int = None):
    if title_map is None:
        title_map = {
            'oscillator1': 'Oscillator 1',
            'oscillator2': 'Oscillator 2',
            'bactgrow': 'E. coli growth',
            'stressstrain': 'Stress-Strain',
            # Add 8 new problem instances with new naming
            'bio/ins0': 'BPG0',
            'bio/ins1': 'BPG1',
            'chem/ins0': 'CRK0', 
            'chem/ins1': 'CRK1',
            'matsci/ins0': 'MatSci0',
            'matsci/ins1': 'MatSci1',
            'phys/ins0': 'PO0',
            'phys/ins1': 'PO1'
        }

    # Model name mapping for standardized display
    if model_map is None:
        model_map = {
            'gpt-3.5-turbo': 'GPT-3.5-Turbo',
            'gpt-4o-mini': 'GPT-4o-mini',
            'gpt-4': 'GPT-4',
            'gpt-4o': 'GPT-4o'
        }

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'axes.titlesize': 32,      # 进一步增大子图标题 (28 -> 32)
        'axes.labelsize': 28,      # 进一步增大坐标轴标签 (24 -> 28)
        'xtick.labelsize': 24,     # 进一步增大x轴刻度标签 (20 -> 24)
        'ytick.labelsize': 24,     # 进一步增大y轴刻度标签 (20 -> 24)
        'legend.fontsize': 26,     # 进一步增大图例字体 (22 -> 26)
        'lines.linewidth': 3.0,    # 增加线宽以提高可见性 (2.5 -> 3.0)
        'grid.alpha': 0.8,
        'grid.linewidth': 1.0,     # 增加网格线宽度
        'axes.linewidth': 1.5,     # 增加坐标轴边框线宽
    })

    # Multi-model layout logic
    n_models = len(models)
    n_problems = len(problems)
    
    if n_models == 1:
        # Single model: use original dynamic grid
        if cols is None:
            cols = min(4, max(2, int(math.ceil(math.sqrt(n_problems)))))
        rows = int(math.ceil(n_problems / cols))
    else:
        # Multi-model layouts
        if n_problems == 8 and n_models == 2:
            # 8 problems + 2 models: 4x4 grid
            # 4 problem domains (bio, chem, matsci, phys) as rows
            # 2 instances per domain × 2 models = 4 columns
            rows = 4  # 4 problem domains
            cols = 4  # 2 instances × 2 models
        elif n_problems == 4 and n_models == 2:
            # 4 problems + 2 models: 4x2 grid
            # Model 1: top 2x2 (rows 0-1), Model 2: bottom 2x2 (rows 2-3)
            rows = 4  # 4 rows total
            cols = 2  # 2 columns
        else:
            # Fallback to original logic for other cases
            cols = n_models
            rows = n_problems
    
    # Adjust figure size based on grid - 增加图的高度
    fig_width = 6 * cols + 1
    fig_height = 4.8 * rows + 2  # 增加每个子图高度 (3.6 -> 4.8) 和整体高度 (1 -> 2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    color_cycle = {
        'evosr-llm': 'tab:blue',
        'llm_sr': 'tab:orange',
        'funsearch': 'tab:green',
        'drsr': 'tab:red'
    }
    style_cycle = {
        'evosr-llm': '-',
        'llm_sr': '--',
        'funsearch': '-.',
        'drsr': ':'
    }

    if n_models == 1:
        # Single model: iterate over problems
        for ax, problem in zip(axes.flat, problems):
            plot_single_problem(ax, problem, models[0], algorithms, evosr_base, baselines_base, 
                               title_map, color_cycle, style_cycle)
            ax.set_title(title_map.get(problem, problem), fontsize=32, weight='bold', pad=20)  # 确保标题字体够大
    elif n_problems == 8 and n_models == 2:
        # Special case: 8 problems, 2 models -> 4x4 grid
        # Layout: 4 domains as rows, 2 models × 2 instances = 4 columns
        # Columns: ins0+m1, ins1+m1, ins0+m2, ins1+m2
        domains = [
            ('bio', [problems[0], problems[1]]),    # bio/ins0, bio/ins1
            ('chem', [problems[2], problems[3]]),   # chem/ins0, chem/ins1  
            ('matsci', [problems[4], problems[5]]), # matsci/ins0, matsci/ins1
            ('phys', [problems[6], problems[7]])    # phys/ins0, phys/ins1
        ]
        
        for row_idx, (domain_name, domain_problems) in enumerate(domains):
            # Column layout: ins0+m1, ins1+m1, ins0+m2, ins1+m2
            col_positions = [
                (domain_problems[0], models[0], 0),  # ins0 + model1 -> col 0
                (domain_problems[1], models[0], 1),  # ins1 + model1 -> col 1
                (domain_problems[0], models[1], 2),  # ins0 + model2 -> col 2
                (domain_problems[1], models[1], 3)   # ins1 + model2 -> col 3
            ]
            
            for problem, model, col_idx in col_positions:
                ax = axes[row_idx, col_idx]
                plot_single_problem(ax, problem, model, algorithms, evosr_base, baselines_base, 
                                   title_map, color_cycle, style_cycle)
                
                # Use the new title mapping instead of domain/instance format
                problem_title = title_map.get(problem, problem)
                model_display = model_map.get(model, model)
                ax.set_title(f"{problem_title} ({model_display})", fontsize=28, weight='bold', pad=15)  # 增大标题字体 (24 -> 28)
    elif n_problems == 4 and n_models == 2:
        # Special case: 4 problems, 2 models -> 4x2 grid  
        # Layout: Model 1 in top 2x2, Model 2 in bottom 2x2
        # Rows 0-1: Model 1 (problems 0-3 in 2x2 layout)
        # Rows 2-3: Model 2 (problems 0-3 in 2x2 layout)
        for model_idx, model in enumerate(models):
            for problem_idx, problem in enumerate(problems):
                # Calculate position: model determines row offset (0 or 2), problem determines specific position
                base_row = model_idx * 2  # 0 for model1, 2 for model2
                row_offset = problem_idx // 2  # 0 for problems 0,1; 1 for problems 2,3
                col_offset = problem_idx % 2   # 0 for problems 0,2; 1 for problems 1,3
                
                row_idx = base_row + row_offset
                col_idx = col_offset
                
                ax = axes[row_idx, col_idx]
                plot_single_problem(ax, problem, model, algorithms, evosr_base, baselines_base, 
                                   title_map, color_cycle, style_cycle)
                
                problem_title = title_map.get(problem, problem)
                model_display = model_map.get(model, model)
                ax.set_title(f"{problem_title} ({model_display})", fontsize=28, weight='bold', pad=15)  # 增大标题字体 (24 -> 28)
    else:
        # Multi-model: iterate over problems (rows) and models (columns)
        for row_idx, problem in enumerate(problems):
            for col_idx, model in enumerate(models):
                ax = axes[row_idx, col_idx]
                plot_single_problem(ax, problem, model, algorithms, evosr_base, baselines_base, 
                                   title_map, color_cycle, style_cycle)
                
                # Add problem title with model name
                problem_title = title_map.get(problem, problem)
                model_display = model_map.get(model, model)
                ax.set_title(f"{problem_title} ({model_display})", fontsize=28, weight='bold', pad=15)  # 增大标题字体 (24 -> 28)

        # Add column headers for models and instances
        if n_problems == 8 and n_models == 2:
            # For 4x4 layout: ins0+m1, ins1+m1, ins0+m2, ins1+m2
            model1_display = model_map.get(models[0], models[0])
            model2_display = model_map.get(models[1], models[1])
            headers = [f"ins0 ({model1_display})", f"ins1 ({model1_display})", 
                      f"ins0 ({model2_display})", f"ins1 ({model2_display})"]
            for col_idx, header in enumerate(headers):
                axes[0, col_idx].text(0.5, 1.08, header, transform=axes[0, col_idx].transAxes,
                                     ha='center', va='bottom', fontsize=24, weight='bold')  # 降低位置减少空白 (1.15 -> 1.08)，调整字体 (26 -> 24)
        else:
            # Original logic for other layouts
            for col_idx, model in enumerate(models):
                model_display = model_map.get(model, model)
                axes[0, col_idx].text(0.5, 1.05, model_display, transform=axes[0, col_idx].transAxes,
                                     ha='center', va='bottom', fontsize=28, weight='bold')  # 降低位置减少空白 (1.1 -> 1.05)，调整字体 (30 -> 28)

    # Hide unused subplots if any (only for single model case)
    if n_models == 1:
        total_subplots = rows * cols
        for idx in range(len(problems), total_subplots):
            axes.flat[idx].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(algorithms), 5), bbox_to_anchor=(0.5, 0.985), 
               frameon=False, fontsize=26)  # 稍微降低图例位置，增加到第一行子图的距离 (0.985 -> 0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)  # 减少子图垂直间距 (h_pad: 3.0 -> 2.0)，调整上边界 (0.96 -> 0.95)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved figure to {out_path}")
    plt.close(fig)

# ------------------------ CLI --------------------------------------------- #

def parse_args():
    # benchmark = 'llm_sr'
    # problems = ['oscillator1', 'oscillator2', 'bactgrow', 'stressstrain']
    benchmark = 'llm_srbench'
    problems = ['bio/ins0', 'bio/ins1', 'chem/ins0', 'chem/ins1', 
                'matsci/ins0', 'matsci/ins1', 'phys/ins0', 'phys/ins1']
    
    models = ['gpt-3.5-turbo', 'gpt-4o-mini']
    ap = argparse.ArgumentParser(description='Plot multi-algorithm convergence curves.')
    ap.add_argument('--models', nargs='+', default=models, help='Model names (subfolders).')
    ap.add_argument('--problems', nargs='+', 
                    default=problems,
                    help='Problem instance names in format domain/instance (e.g. bio/ins0).')
    ap.add_argument('--algorithms', nargs='+', default=['evosr-llm', 'llm_sr', 'funsearch', 'drsr'])
    ap.add_argument('--out', default=f'./figures/exp/convergence_curve—new-{benchmark}.pdf')
    ap.add_argument('--cols', type=int, default=None, help='Number of subplot columns (auto if not specified).')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    plot_multi_convergence(models=args.models,
                           problems=args.problems,
                           algorithms=args.algorithms,
                           out_path=args.out,
                           cols=args.cols)
