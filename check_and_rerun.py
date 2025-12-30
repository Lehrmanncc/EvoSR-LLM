"""
检测哪些实验需要重跑 - 简化版
直接修改下面的配置变量，然后运行: python check_and_rerun.py
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

# ============ 配置区域 - 修改这里 ============
RUN_IDS = [2]  # 要检查的run ID列表，例如 [2, 3, 4]

GENERATE_SCRIPT = True  # 是否生成重跑脚本
CLEAN_INCOMPLETE = True   # 是否清理并备份未完成的实验（谨慎使用！）
BACKUP_DIR = "./incomplete_backups"  # 备份目录

# ============================================

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH)

# 数据集配置（与run_all_experiments.py保持一致）
OES_DATASETS = [
    {"benchmark": "oes", "problem_name": "bactgrow", "idx": None},
    {"benchmark": "oes", "problem_name": "oscillator1", "idx": None},
    {"benchmark": "oes", "problem_name": "oscillator2", "idx": None},
    {"benchmark": "oes", "problem_name": "stressstrain", "idx": None},
]

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

ALL_DATASETS = OES_DATASETS + LLMSRBENCH_DATASETS
MODELS = ["gpt-4o-mini", "gpt-3.5-turbo"]
BASE_OUTPUT_DIR = "./logs_tevc_r1"


def get_instance_name(dataset):
    """构建实例名"""
    if dataset["idx"] is None:
        return dataset["problem_name"]
    else:
        return f"{dataset['problem_name']}{dataset['idx']}"


def check_experiment_complete(run_id, instance_name, model):
    """检查单个实验是否完成"""
    result_path = os.path.join(BASE_OUTPUT_DIR, f"run{run_id}", instance_name, model)
    best_equ_path = os.path.join(result_path, "best_equ.json")
    return os.path.exists(best_equ_path)


def get_experiment_progress(run_id, instance_name, model):
    """获取实验进度信息"""
    result_path = os.path.join(BASE_OUTPUT_DIR, f"run{run_id}", instance_name, model)
    
    if not os.path.exists(result_path):
        return {"status": "not_started", "fe": 0, "max_fe": None}
    
    if check_experiment_complete(run_id, instance_name, model):
        best_equ_path = os.path.join(result_path, "best_equ.json")
        try:
            with open(best_equ_path, 'r') as f:
                data = json.load(f)
                time_cost = data.get('time_cost', 0)
                return {"status": "completed", "fe": None, "time_cost": time_cost}
        except:
            return {"status": "completed", "fe": None, "time_cost": None}
    
    # 检查中断的实验
    pops_dir = os.path.join(result_path, "pops")
    if os.path.exists(pops_dir):
        pop_files = [f for f in os.listdir(pops_dir) if f.startswith("population_fe=")]
        if pop_files:
            fe_values = []
            for f in pop_files:
                try:
                    fe = int(f.replace("population_fe=", "").replace(".json", ""))
                    fe_values.append(fe)
                except:
                    pass
            if fe_values:
                max_fe = max(fe_values)
                return {"status": "incomplete", "fe": max_fe, "max_fe": 3000}
    
    return {"status": "incomplete", "fe": 0, "max_fe": 3000}


def scan_experiments(run_ids):
    """扫描指定run_id的所有实验"""
    results = {
        "completed": [],
        "incomplete": [],
        "not_started": []
    }
    
    config_id = 0
    for dataset in ALL_DATASETS:
        for model in MODELS:
            config_id += 1
            instance_name = get_instance_name(dataset)
            
            for run_id in run_ids:
                progress = get_experiment_progress(run_id, instance_name, model)
                
                experiment_info = {
                    "config_id": config_id,
                    "dataset": dataset,
                    "model": model,
                    "instance_name": instance_name,
                    "run_id": run_id,
                    **progress
                }
                
                if progress["status"] == "completed":
                    results["completed"].append(experiment_info)
                elif progress["status"] == "incomplete":
                    results["incomplete"].append(experiment_info)
                else:
                    results["not_started"].append(experiment_info)
    
    return results


def print_summary(results, run_ids):
    """打印汇总信息"""
    total = len(results["completed"]) + len(results["incomplete"]) + len(results["not_started"])
    
    print("=" * 80)
    print(f"实验完成情况汇总 (Run IDs: {run_ids})")
    print("=" * 80)
    print(f"总实验数: {total}")
    print(f"已完成: {len(results['completed'])} ({len(results['completed'])/total*100:.1f}%)")
    print(f"未完成: {len(results['incomplete'])} ({len(results['incomplete'])/total*100:.1f}%)")
    print(f"未开始: {len(results['not_started'])} ({len(results['not_started'])/total*100:.1f}%)")
    print("=" * 80)
    
    if results["incomplete"]:
        print("\n未完成的实验详情:")
        print("-" * 80)
        for exp in results["incomplete"]:
            print(f"Config {exp['config_id']:2d} | {exp['instance_name']:15s} | "
                  f"{exp['model']:15s} | Run {exp['run_id']} | FE: {exp['fe']}/{exp['max_fe']}")
    
    if results["not_started"]:
        print("\n未开始的实验详情:")
        print("-" * 80)
        for exp in results["not_started"]:
            print(f"Config {exp['config_id']:2d} | {exp['instance_name']:15s} | "
                  f"{exp['model']:15s} | Run {exp['run_id']}")


def generate_rerun_configs(incomplete_experiments):
    """生成需要重跑的实验配置列表"""
    configs = []
    for exp in incomplete_experiments:
        configs.append({
            "config_id": exp["config_id"],
            "dataset": exp["dataset"],
            "model": exp["model"],
            "run_id": exp["run_id"]
        })
    return configs


def save_rerun_script(incomplete_experiments, run_ids):
    """生成重跑脚本"""
    run_groups = {}
    for exp in incomplete_experiments:
        run_id = exp["run_id"]
        if run_id not in run_groups:
            run_groups[run_id] = []
        run_groups[run_id].append(exp)
    
    script_lines = ["#!/bin/bash", ""]
    script_lines.append("# 重跑未完成的实验")
    script_lines.append("# 生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    script_lines.append("")
    
    for run_id in sorted(run_groups.keys()):
        exps = run_groups[run_id]
        
        config_filename = f"rerun_configs_run{run_id}.json"
        configs = generate_rerun_configs(exps)
        with open(config_filename, 'w') as f:
            json.dump(configs, f, indent=2)
        
        script_lines.append(f"# Run {run_id}: {len(exps)} 个实验需要重跑")
        script_lines.append(f"echo 'Starting {len(exps)} experiments for Run {run_id}...'")
        script_lines.append(f"python run_all_experiments.py --config-file {config_filename}")
        script_lines.append("")
        
        print(f"已保存配置文件: {config_filename} ({len(exps)} 个实验)")
    
    script_path = "rerun_incomplete_experiments.sh"
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    os.chmod(script_path, 0o755)
    print(f"\n重跑脚本已保存: {script_path}")
    print(f"执行方式: bash {script_path}")


def clean_and_backup(incomplete_experiments, backup_dir):
    """清理并备份未完成的实验"""
    if not incomplete_experiments:
        print("\n没有未完成的实验需要清理。")
        return
    
    print(f"\n警告: 即将清理 {len(incomplete_experiments)} 个未完成实验的中间文件!")
    print(f"备份目录: {backup_dir}")
    confirm = input("确认清理并备份? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("已取消清理操作")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_root = os.path.join(backup_dir, f"backup_{timestamp}")
    os.makedirs(backup_root, exist_ok=True)
    
    backup_count = 0
    delete_count = 0
    
    for exp in incomplete_experiments:
        result_path = os.path.join(BASE_OUTPUT_DIR, f"run{exp['run_id']}", 
                                  exp['instance_name'], exp['model'])
        
        if os.path.exists(result_path):
            backup_path = os.path.join(backup_root, f"run{exp['run_id']}", 
                                      exp['instance_name'], exp['model'])
            
            try:
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copytree(result_path, backup_path)
                backup_count += 1
                
                shutil.rmtree(result_path)
                delete_count += 1
                
                print(f"✓ 已备份并删除: Config {exp['config_id']} | "
                      f"{exp['instance_name']} | {exp['model']} | Run {exp['run_id']}")
            except Exception as e:
                print(f"✗ 处理失败: Config {exp['config_id']} | "
                      f"{exp['instance_name']} | {exp['model']} | Run {exp['run_id']}")
                print(f"  错误: {e}")
    
    print(f"\n清理完成!")
    print(f"  备份: {backup_count} 个实验")
    print(f"  删除: {delete_count} 个实验")
    print(f"  备份位置: {backup_root}")
    
    # 保存备份信息
    backup_info = {
        "timestamp": timestamp,
        "backup_path": backup_root,
        "total_backed_up": backup_count,
        "total_deleted": delete_count,
        "experiments": [
            {
                "config_id": exp["config_id"],
                "instance_name": exp["instance_name"],
                "model": exp["model"],
                "run_id": exp["run_id"],
                "fe": exp.get("fe", 0),
                "status": exp["status"]
            }
            for exp in incomplete_experiments
        ]
    }
    
    backup_info_file = os.path.join(backup_root, "backup_info.json")
    with open(backup_info_file, 'w') as f:
        json.dump(backup_info, f, indent=2)
    print(f"  备份信息: {backup_info_file}")


if __name__ == "__main__":
    print("\n检测实验完成情况...")
    print(f"配置: RUN_IDS = {RUN_IDS}")
    print(f"      GENERATE_SCRIPT = {GENERATE_SCRIPT}")
    print(f"      CLEAN_INCOMPLETE = {CLEAN_INCOMPLETE}")
    print()
    
    # 扫描实验
    results = scan_experiments(RUN_IDS)
    
    # 打印汇总
    print_summary(results, RUN_IDS)
    
    # 合并未完成和未开始的实验
    all_incomplete = results["incomplete"] + results["not_started"]
    
    if not all_incomplete:
        print("\n所有实验都已完成！")
    else:
        # 生成重跑脚本
        if GENERATE_SCRIPT:
            print("\n" + "=" * 80)
            print("生成重跑脚本...")
            print("=" * 80)
            save_rerun_script(all_incomplete, RUN_IDS)
        
        # 清理未完成实验
        if CLEAN_INCOMPLETE:
            print("\n" + "=" * 80)
            print("清理未完成实验...")
            print("=" * 80)
            clean_and_backup(results["incomplete"], BACKUP_DIR)
    
    print("\n完成！")
