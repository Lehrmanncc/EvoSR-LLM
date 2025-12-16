"""
批量运行所有实验配置
- 4个OES数据集 (bactgrow, oscillator1, oscillator2, stressstrain)，每个2个实例
- 4个LLM_SRBENCH数据集 (bio, chem, phys, matsci)，每个2个实例
- 2个模型 (gpt-4o-mini, gpt-3.5-turbo)
- 每个配置运行5次
总计: 12个数据集 × 2个模型 × 5次 = 120个实验
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH)

from algorithm.sr_evol import SrEvol
from utils.util import Paras
from Problems.problems import ProblemSR


# ============ 配置区域 ============
def load_api_keys_from_env():
    """从.env文件加载API keys"""
    env_path = os.path.join(ABS_PATH, '.env')
    api_keys = []
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip().startswith('API_KEY_'):
                        # 移除引号
                        value = value.strip().strip('"').strip("'")
                        api_keys.append(value)
    
    if not api_keys:
        print("Warning: No API keys found in .env file. Using placeholder.")
        api_keys = ["sk-placeholder"]
    else:
        print(f"Loaded {len(api_keys)} API keys from .env")
    
    return api_keys

API_KEYS = load_api_keys_from_env()

# OES 数据集配置 (使用 benchmark="llm_sr")
OES_DATASETS = [
    {"benchmark": "llm_sr", "problem_name": "bactgrow", "idx": None},
    {"benchmark": "llm_sr", "problem_name": "oscillator1", "idx": None},
    {"benchmark": "llm_sr", "problem_name": "oscillator2", "idx": None},
    {"benchmark": "llm_sr", "problem_name": "stressstrain", "idx": None},
]

# LLM_SRBENCH 数据集配置
LLMSRBENCH_DATASETS = [
    {"benchmark": "llm_srbench", "problem_name": "bio", "idx": 0},
    {"benchmark": "llm_srbench", "problem_name": "bio", "idx": 1},
    {"benchmark": "llm_srbench", "problem_name": "chem", "idx": 0},
    {"benchmark": "llm_srbench", "problem_name": "chem", "idx": 1},
    {"benchmark": "llm_srbench", "problem_name": "phys", "idx": 0},
    {"benchmark": "llm_srbench", "problem_name": "phys", "idx": 1},
    {"benchmark": "llm_srbench", "problem_name": "matsci", "idx": 0},
    {"benchmark": "llm_srbench", "problem_name": "matsci", "idx": 1},
]

# 合并所有数据集
ALL_DATASETS = OES_DATASETS + LLMSRBENCH_DATASETS

MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
NUM_RUNS = 5  # 每个配置运行次数

# API endpoint
API_ENDPOINT = "aihubmix.com"

# 其他参数
COMMON_PARAMS = {
    "pop_size": 10,
    "offspring_size": 2,
    "max_fe": 3000,
    "n_process": 4,
    "operators_gen_num": 120,
    "alpha": 5,
    "lamda": 0.00001,
    "exp_debug_mode": False,
}

# 并行实验数量 (建议: CPU核心数 / n_process)
# 如果每个实验用4核，8核CPU可以同时跑2个实验
MAX_PARALLEL_EXPERIMENTS = 2

# ============ 实验执行函数 ============

def run_single_experiment(config):
    """运行单个实验"""
    dataset = config["dataset"]
    model = config["model"]
    run_id = config["run_id"]
    api_key = config["api_key"]
    config_id = config["config_id"]  # 实验配置ID (1-24)
    
    benchmark = dataset["benchmark"]
    problem_name = dataset["problem_name"]
    idx = dataset["idx"]
    
    # 构建实例名: 没有idx就是数据集名，有idx就在数据集后面跟上id
    if idx is None:
        instance_name = problem_name
    else:
        instance_name = f"{problem_name}{idx}"
    
    # 构建输出路径: logs_tevc_r1/run{}/实例名/模型名/
    output_path = f"./logs_tevc_r1/run{run_id}/{instance_name}/{model}"
    
    print(f"[START] Config {config_id} | {instance_name} | {model} | Run {run_id} | API_KEY_{config_id-1}")
    start_time = time.time()
    
    try:
        # 初始化参数
        paras = Paras()
        paras.set_paras(
            benchmark=benchmark,
            llm_api_endpoint=API_ENDPOINT,
            llm_api_key=api_key,
            llm_model=model,
            exp_output_path=output_path,
            **COMMON_PARAMS
        )
        
        # 创建问题实例
        sr_problem = ProblemSR(benchmark, problem_name, idx)
        
        # 运行进化算法
        evolution = SrEvol(paras, sr_problem)
        evolution.run()
        
        elapsed = time.time() - start_time
        status = "SUCCESS"
        error_msg = None
        
    except Exception as e:
        elapsed = time.time() - start_time
        status = "FAILED"
        error_msg = str(e)
        print(f"[ERROR] Config {config_id} | {instance_name} | {model} | Run {run_id}: {error_msg}")
    
    # 保存时间记录到结果目录
    try:
        os.makedirs(output_path, exist_ok=True)
        time_log_path = os.path.join(output_path, "execution_time.txt")
        with open(time_log_path, 'w') as f:
            f.write(f"Configuration ID: {config_id}\n")
            f.write(f"Instance: {instance_name}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Status: {status}\n")
            f.write(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Elapsed Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")
            if error_msg:
                f.write(f"Error: {error_msg}\n")
    except Exception as e:
        print(f"[WARNING] Failed to save time log: {e}")
    
    result = {
        "config_id": config_id,
        "instance_name": instance_name,
        "model": model,
        "run_id": run_id,
        "status": status,
        "elapsed_time_seconds": elapsed,
        "elapsed_time_minutes": elapsed / 60,
        "elapsed_time_hours": elapsed / 3600,
        "error": error_msg,
        "output_path": output_path,
        "api_key_index": config_id - 1
    }
    
    print(f"[{status}] Config {config_id} | {instance_name} | {model} | Run {run_id} | {elapsed/60:.2f} min")
    return result


def generate_experiment_configs():
    """生成所有实验配置
    
    每个实验配置(数据集×模型组合)分配一个独立的API key
    总共24个配置: 12个数据集 × 2个模型 = 24
    每个配置运行5次，共用同一个API key
    """
    configs = []
    config_id = 0
    
    # 先按配置分组: 每个数据集×模型组合是一个配置
    for dataset in ALL_DATASETS:
        for model in MODELS:
            config_id += 1
            # 为这个配置分配一个API key (config_id: 1-24)
            api_key = API_KEYS[(config_id - 1) % len(API_KEYS)]
            
            # 这个配置运行多次，共用同一个API key，run_id从2开始
            for run_id in range(2, NUM_RUNS + 2):
                configs.append({
                    "config_id": config_id,
                    "dataset": dataset,
                    "model": model,
                    "run_id": run_id,
                    "api_key": api_key
                })
    
    return configs


def run_all_experiments_parallel():
    """并行运行所有实验"""
    configs = generate_experiment_configs()
    total = len(configs)
    
    print(f"=" * 80)
    print(f"Total experiments to run: {total}")
    print(f"  - {len(ALL_DATASETS)} datasets")
    print(f"  - {len(MODELS)} models")
    print(f"  - {NUM_RUNS} runs per configuration")
    print(f"  - Max parallel experiments: {MAX_PARALLEL_EXPERIMENTS}")
    print(f"=" * 80)
    
    # 保存配置
    config_file = f"./experiment_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2, default=str)
    print(f"Experiment configs saved to: {config_file}\n")
    
    # 运行实验
    results = []
    completed = 0
    failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_EXPERIMENTS) as executor:
        futures = {executor.submit(run_single_experiment, cfg): cfg for cfg in configs}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if result["status"] == "FAILED":
                failed += 1
            
            # 进度报告
            progress = (completed / total) * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            eta = avg_time * (total - completed)
            
            print(f"\nProgress: {completed}/{total} ({progress:.1f}%) | "
                  f"Failed: {failed} | "
                  f"ETA: {eta/3600:.1f}h\n")
    
    # 保存结果
    total_time = time.time() - start_time
    results_file = f"./experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary = {
        "total_experiments": total,
        "completed": completed,
        "successful": completed - failed,
        "failed": failed,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600,
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"All experiments completed!")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    print(f"Success: {completed - failed}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"Results saved to: {results_file}")
    print(f"=" * 80)


def run_all_experiments_sequential():
    """顺序运行所有实验 (备用方案)"""
    configs = generate_experiment_configs()
    total = len(configs)
    
    print(f"Running {total} experiments sequentially...")
    
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{total}")
        result = run_single_experiment(cfg)
        results.append(result)
    
    # 保存结果
    results_file = f"./experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    # 直接运行并行实验，不需要命令行参数
    run_all_experiments_parallel()
