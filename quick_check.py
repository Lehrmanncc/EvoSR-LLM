"""
快速检查实验完成情况
只显示未完成和未开始的实验
"""

import os
import sys

# ============ 配置 ============
RUN_IDS = [2, 3, 4, 5, 6]  # 要检查的run ID列表
# ==============================

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH)

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
    if dataset["idx"] is None:
        return dataset["problem_name"]
    else:
        return f"{dataset['problem_name']}{dataset['idx']}"


def check_experiment(run_id, instance_name, model):
    result_path = os.path.join(BASE_OUTPUT_DIR, f"run{run_id}", instance_name, model)
    best_equ_path = os.path.join(result_path, "best_equ.json")
    
    if os.path.exists(best_equ_path):
        return "完成"
    elif os.path.exists(result_path):
        # 检查进度
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
                    return f"进行中 ({max(fe_values)}/3000)"
        return "进行中"
    else:
        return "未开始"


if __name__ == "__main__":
    print("=" * 100)
    print(f"检查 Run IDs: {RUN_IDS}")
    print("=" * 100)
    
    incomplete_list = []
    completed_count = 0
    total_count = 0
    
    config_id = 0
    for dataset in ALL_DATASETS:
        for model in MODELS:
            config_id += 1
            instance_name = get_instance_name(dataset)
            
            for run_id in RUN_IDS:
                total_count += 1
                status = check_experiment(run_id, instance_name, model)
                
                if status == "完成":
                    completed_count += 1
                else:
                    incomplete_list.append({
                        "config_id": config_id,
                        "instance": instance_name,
                        "model": model,
                        "run_id": run_id,
                        "status": status
                    })
    
    print(f"\n总实验数: {total_count}")
    print(f"已完成: {completed_count} ({completed_count/total_count*100:.1f}%)")
    print(f"未完成: {len(incomplete_list)} ({len(incomplete_list)/total_count*100:.1f}%)")
    
    if incomplete_list:
        print("\n" + "=" * 100)
        print("未完成的实验:")
        print("=" * 100)
        print(f"{'Config':<8} {'Instance':<20} {'Model':<20} {'Run':<6} {'Status':<20}")
        print("-" * 100)
        for exp in incomplete_list:
            print(f"{exp['config_id']:<8} {exp['instance']:<20} {exp['model']:<20} "
                  f"{exp['run_id']:<6} {exp['status']:<20}")
    else:
        print("\n✓ 所有实验都已完成！")
    
    print("=" * 100)
