#!/usr/bin/env python3
"""
Plot Convergence Curve for EvoSR-LLM on oscillator1 problem
Analyzes gpt-4o-mini results, identifies improvement points, and generates convergence visualization.
"""

import os
import json
import shutil
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import argparse
from pathlib import Path


def extract_index(filename: str) -> int:
    """Extract numeric iteration index from various filename patterns.
    Supported patterns:
      samples_fe=123.json, samples_123.json, sample_123.json
    Returns -1 if no pattern matched (these files will be sorted last).
    """
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


def load_evaluation_data(samples_dir: str) -> List[Dict]:
    """Load all evaluation data from JSON files in samples directory.
    
    Returns:
        List of dictionaries containing evaluation data with added metadata
    """
    if not os.path.isdir(samples_dir):
        raise ValueError(f"Samples directory not found: {samples_dir}")
    
    files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
    if not files:
        raise ValueError(f"No JSON files found in {samples_dir}")
    
    # Sort files by evaluation index
    files_sorted = sorted(files, key=extract_index)
    
    evaluations = []
    for i, fname in enumerate(files_sorted):
        fpath = os.path.join(samples_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Extract objective value (fitness)
            objective = data.get('objective')
            if objective is None:
                print(f"Warning: No objective found in {fname}, skipping")
                continue
                
            # Add metadata
            eval_data = {
                'eval_index': i + 1,  # 1-based indexing
                'filename': fname,
                'filepath': fpath,
                'objective': objective,
                'raw_data': data
            }
            evaluations.append(eval_data)
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
    
    print(f"Loaded {len(evaluations)} evaluations from {len(files)} files")
    return evaluations


def compute_best_so_far(evaluations: List[Dict], minimize: bool = True) -> List[Dict]:
    """Compute best-so-far sequence and identify improvement points.
    
    Args:
        evaluations: List of evaluation dictionaries
        minimize: True if lower is better, False if higher is better
        
    Returns:
        Updated evaluations list with best_so_far and improved flags
    """
    if not evaluations:
        return []
    
    best_so_far = None
    improved_points = []
    
    for i, eval_data in enumerate(evaluations):
        current_fitness = eval_data['objective']
        
        # Initialize or update best-so-far
        if best_so_far is None:
            best_so_far = current_fitness
            improved = True
        else:
            if minimize:
                improved = current_fitness < best_so_far
            else:
                improved = current_fitness > best_so_far
                
            if improved:
                best_so_far = current_fitness
        
        # Update evaluation data
        eval_data['best_so_far'] = best_so_far
        eval_data['improved'] = improved
        
        if improved:
            improved_points.append(i)
    
    print(f"Found {len(improved_points)} improvement points out of {len(evaluations)} evaluations")
    return evaluations


def identify_significant_improvements(evaluations: List[Dict], minimize: bool = True, max_points: int = 4) -> List[Dict]:
    """Identify the most significant improvement points including cumulative improvement nodes.
    
    Args:
        evaluations: List of evaluation dictionaries with best_so_far computed
        minimize: True if lower is better, False if higher is better
        max_points: Maximum number of significant points to identify (including initial point)
        
    Returns:
        List of evaluation dictionaries representing significant improvements
    """
    if not evaluations:
        return []
    
    # Always include the first evaluation (initial point)
    significant_points = [evaluations[0]]
    
    # Find all improvement points with their improvement magnitudes
    improvement_candidates = []
    
    for i, eval_data in enumerate(evaluations[1:], 1):  # Skip first point
        # 不将“最后一个评估点”作为关键点候选，避免终点进入关键点集合
        if i == len(evaluations) - 1:
            continue
        if eval_data['improved']:
            # Calculate improvement magnitude
            prev_best = evaluations[i-1]['best_so_far']
            current_best = eval_data['best_so_far']
            
            if minimize:
                improvement_ratio = (prev_best - current_best) / abs(prev_best) if prev_best != 0 else 0
            else:
                improvement_ratio = (current_best - prev_best) / abs(prev_best) if prev_best != 0 else 0
            
            improvement_candidates.append({
                'eval_data': eval_data,
                'improvement_ratio': improvement_ratio,
                'absolute_improvement': abs(current_best - prev_best),
                'index': i
            })
    
    # Sort by improvement magnitude (largest improvements first)
    improvement_candidates.sort(key=lambda x: x['improvement_ratio'], reverse=True)
    
    # 选择主要改进点
    selected_improvements = []
    
    print(f"Total improvement candidates: {len(improvement_candidates)}")
    print(f"Max points allowed (excluding initial): {max_points - 1}")
    
    # 首先选择最显著的单次改进点
    for candidate in improvement_candidates[:max_points-1]:
        print(f"  Checking candidate at eval {candidate['eval_data']['eval_index']}: improvement_ratio={candidate['improvement_ratio']:.4f}")
        if candidate['improvement_ratio'] >= 0.01:  # 至少1%的改进
            selected_improvements.append(candidate)
            print(f"    -> Selected (meets 1% threshold)")
        else:
            print(f"    -> Rejected (below 1% threshold)")
    
    print(f"Selected {len(selected_improvements)} single-step improvements")
    
    # 寻找累积改进后的重要节点
    print(f"Checking if need cumulative improvements: {len(selected_improvements)} < {max_points - 1}?")
    if len(selected_improvements) < max_points - 1:
        print("Looking for cumulative improvement nodes...")
        # 传入所有评估数据，而不是预筛选的候选点
        cumulative_nodes = find_cumulative_improvement_nodes(evaluations, minimize)
        
        # 添加累积改进节点，但不超过总数限制
        for node in cumulative_nodes:
            if len(selected_improvements) >= max_points - 1:
                print(f"Reached max points limit, stopping")
                break
            # 避免重复添加已选择的点
            if not any(s['index'] == node['index'] for s in selected_improvements):
                selected_improvements.append(node)
                print(f"Added cumulative improvement node at eval {node['eval_data']['eval_index']}")
            else:
                print(f"Skipped duplicate node at eval {node['eval_data']['eval_index']}")
    else:
        print("Skipping cumulative improvements (enough single-step improvements found)")
    
    # Add the selected significant improvement points
    for candidate in selected_improvements:
        significant_points.append(candidate['eval_data'])
    
    # Sort by evaluation index for proper ordering
    significant_points.sort(key=lambda x: x['eval_index'])
    
    print(f"Selected {len(significant_points)} significant improvement points:")
    for point in significant_points:
        print(f"  Eval {point['eval_index']}: best_so_far={point['best_so_far']:.6f}")
    
    return significant_points


def find_cumulative_improvement_nodes(evaluations: List[Dict], minimize: bool = True, 
                                     cumulative_threshold: float = 0.03, plateau_length: int = 400) -> List[Dict]:
    """Find nodes that represent cumulative improvements followed by long plateau periods.
    
    Args:
        evaluations: List of evaluation dictionaries with best_so_far computed
        minimize: True if lower is better
        cumulative_threshold: Minimum cumulative improvement ratio (default 3%)
        plateau_length: Minimum plateau length in number of evaluations (default 400)
        
    Returns:
        List of cumulative improvement nodes
    """
    cumulative_nodes = []
    
    print(f"  Searching for cumulative improvement patterns...")
    print(f"  Cumulative threshold: {cumulative_threshold:.1%}, Plateau length: {plateau_length}")
    
    # 找到所有评估点作为潜在的累积起点（不仅仅是改进点）
    # 我们需要检查每个点作为起点，看后续是否有累积改进
    print(f"  Found {len(evaluations)} total evaluation points to analyze as potential start points")
    
    # 遍历每个评估点作为累积起点
    for start_idx in range(len(evaluations)):
        start_eval = evaluations[start_idx]
        start_value = start_eval['best_so_far']
        
        # 只分析一部分起点以避免过多输出，或者可以设置更严格的筛选条件
        if start_idx % 50 == 0 or start_eval['improved']:  # 每50个点分析一次，或者是改进点
            print(f"    Analyzing from eval {start_eval['eval_index']} (idx={start_idx}, value={start_value:.6f})")
        
        # 从这个起点开始向后搜索累积改进
        best_cumulative_end_idx = start_idx
        best_cumulative_end_value = start_value
        best_total_improvement = 0
        
        # 在一个合理的窗口内搜索后续点（不管是否是改进点）
        search_window = min(len(evaluations) - start_idx - 1, 200)
        
        for offset in range(1, search_window + 1):
            current_idx = start_idx + offset
            current_eval = evaluations[current_idx]
            current_value = current_eval['best_so_far']
            
            # 计算从起点到当前点的累积改进（不管当前点是否是改进点）
            if minimize:
                total_improvement = (start_value - current_value) / abs(start_value) if start_value != 0 else 0
            else:
                total_improvement = (current_value - start_value) / abs(start_value) if start_value != 0 else 0
            
            # 如果累积改进达到阈值，更新最佳累积终点
            if total_improvement >= cumulative_threshold:
                best_cumulative_end_idx = current_idx
                best_cumulative_end_value = current_value
                best_total_improvement = total_improvement
                
                if start_idx % 50 == 0 or start_eval['improved']:  # 只为部分起点打印详细信息
                    print(f"      Found cumulative improvement at eval {current_eval['eval_index']}: total={total_improvement:.4f}")
        
        # 如果找到了满足阈值的累积改进，检查平台期
        if best_total_improvement >= cumulative_threshold and best_cumulative_end_idx > start_idx:
            # 检查从累积终点开始的平台期长度
            plateau_start_idx = best_cumulative_end_idx + 1
            actual_plateau_length = count_long_plateau_length(evaluations, plateau_start_idx, minimize)
            
            if start_idx % 50 == 0 or start_eval['improved']:  # 只为部分起点打印详细信息
                print(f"      Cumulative improvement found: {best_total_improvement:.4f}")
                print(f"      Checking plateau from idx {plateau_start_idx}: length={actual_plateau_length}")
            
            # 如果平台期足够长，添加这个累积改进节点
            if actual_plateau_length >= plateau_length:
                if start_idx % 50 == 0 or start_eval['improved']:  # 只为部分起点打印详细信息
                    print(f"      -> PASSED: Adding cumulative node at eval {evaluations[best_cumulative_end_idx]['eval_index']}")
                cumulative_nodes.append({
                    'eval_data': evaluations[best_cumulative_end_idx],
                    'improvement_ratio': best_total_improvement,
                    'absolute_improvement': abs(best_cumulative_end_value - start_value),
                    'index': best_cumulative_end_idx,
                    'plateau_length': actual_plateau_length,
                    'start_eval': start_eval['eval_index'],
                    'end_eval': evaluations[best_cumulative_end_idx]['eval_index'],
                    'start_idx': start_idx
                })
            else:
                if start_idx % 50 == 0 or start_eval['improved']:  # 只为部分起点打印详细信息
                    print(f"      -> FAILED: Plateau too short ({actual_plateau_length} < {plateau_length})")
        else:
            if start_idx % 50 == 0 or start_eval['improved']:  # 只为部分起点打印详细信息
                if best_total_improvement < cumulative_threshold:
                    print(f"      -> FAILED: Cumulative improvement too small ({best_total_improvement:.4f} < {cumulative_threshold})")
                else:
                    print(f"      -> FAILED: No improvements found in window")
    
    # 按累积改进幅度排序，取最好的几个，并去除重复
    cumulative_nodes.sort(key=lambda x: x['improvement_ratio'], reverse=True)
    
    # 去除重复的终点（同一个终点可能从不同起点得到）
    unique_nodes = []
    seen_end_indices = set()
    
    for node in cumulative_nodes:
        if node['index'] not in seen_end_indices:
            unique_nodes.append(node)
            seen_end_indices.add(node['index'])
    
    print(f"  Found {len(unique_nodes)} unique cumulative improvement nodes:")
    for node in unique_nodes:
        print(f"    Eval {node['end_eval']}: cumulative improvement={node['improvement_ratio']:.4f}, plateau={node['plateau_length']}")
    
    return unique_nodes


def count_long_plateau_length(evaluations: List[Dict], start_idx: int, minimize: bool = True) -> int:
    """Count the length of plateau period starting from start_idx.
    A plateau is defined as a period with no significant improvements.
    
    Args:
        evaluations: List of evaluation dictionaries
        start_idx: Starting index to count plateau
        minimize: True if lower is better
        
    Returns:
        Length of plateau period
    """
    if start_idx >= len(evaluations):
        return 0
        
    plateau_length = 0
    
    for i in range(start_idx, len(evaluations)):
        # 如果遇到改进点，平台期结束
        if evaluations[i]['improved']:
            break
        plateau_length += 1
    
    return plateau_length


def copy_key_nodes(significant_points: List[Dict], output_dir: str) -> str:
    """Copy JSON files corresponding to significant improvement points to key_nodes directory.
    
    Args:
        significant_points: List of significant improvement point dictionaries
        
    Returns:
        Path to the key_nodes directory
    """
    key_nodes_dir = os.path.join(output_dir, 'key_nodes')
    os.makedirs(key_nodes_dir, exist_ok=True)
    
    # Copy significant improvement files
    significant_files = []
    for eval_data in significant_points:
        src_path = eval_data['filepath']
        dst_path = os.path.join(key_nodes_dir, eval_data['filename'])
        shutil.copy2(src_path, dst_path)
        significant_files.append({
            'eval_index': eval_data['eval_index'],
            'filename': eval_data['filename'],
            'objective': eval_data['objective'],
            'best_so_far': eval_data['best_so_far']
        })
    
    # Export significant points index
    index_file = os.path.join(key_nodes_dir, 'significant_points.json')
    with open(index_file, 'w') as f:
        json.dump(significant_files, f, indent=2)
    
    print(f"Copied {len(significant_files)} significant improvement files to {key_nodes_dir}")
    return key_nodes_dir


def export_trace_csv(evaluations: List[Dict], output_dir: str) -> str:
    """Export detailed trace data to CSV file."""
    trace_file = os.path.join(output_dir, 'trace.csv')
    
    with open(trace_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eval', 'fitness', 'best_so_far', 'improved', 'file'])
        
        for eval_data in evaluations:
            writer.writerow([
                eval_data['eval_index'],
                eval_data['objective'],
                eval_data['best_so_far'],
                eval_data['improved'],
                eval_data['filename']
            ])
    
    print(f"Exported trace data to {trace_file}")
    return trace_file


def plot_convergence(evaluations: List[Dict], significant_points: List[Dict], output_dir: str, minimize: bool = True):
    """Generate convergence curve plots in multiple formats."""
    
    # Set up matplotlib for publication quality
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],  # 确保Times New Roman优先
        'axes.titlesize': 24,
        'axes.labelsize': 18,      # 增大坐标轴标签字体 (18 -> 24)
        'axes.labelweight': 'bold',  # 设置轴标题为加粗
        'xtick.labelsize': 18,     # 增大刻度标签字体 (16 -> 20)
        'ytick.labelsize': 18,     # 增大刻度标签字体 (16 -> 20)
        'legend.fontsize': 18,
        'lines.linewidth': 3.5,    # 增加线宽 (2.0 -> 3.5)
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Extract data for plotting
    eval_indices = [e['eval_index'] for e in evaluations]
    fitness_values = [e['objective'] for e in evaluations]
    best_so_far_values = [e['best_so_far'] for e in evaluations]
    
    # Extract significant improvement points for highlighting
    significant_indices = [p['eval_index'] for p in significant_points]
    significant_best_so_far = [p['best_so_far'] for p in significant_points]
    
    # Debug: Print significant improvement points for verification
    print("Selected significant improvement points:")
    for point in significant_points:
        print(f"  Eval {point['eval_index']}: objective={point['objective']:.6f}, best_so_far={point['best_so_far']:.6f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 移除所有边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.0)  # 增加轴线宽度
    ax.spines['left'].set_linewidth(2.0)    # 增加轴线宽度
    
    # 设置坐标轴箭头样式（使用深色，增大尺寸）
    ax.annotate('', xy=(1.02, 0), xytext=(0, 0), 
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2.0, color='#2f2f2f'))
    ax.annotate('', xy=(0, 1.02), xytext=(0, 0), 
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2.0, color='#2f2f2f'))

    # 构建要显示标记的索引列表（只包括关键点，再加上终点）
    marker_indices = []
    
    # 只添加关键改进点的标记
    for point in significant_points:
        # 找到该关键点在eval_indices中的索引位置
        try:
            improvement_idx = eval_indices.index(point['eval_index'])
            marker_indices.append(improvement_idx)
        except ValueError:
            continue
    
    # 添加终点（最后一个评估点）
    if eval_indices and len(eval_indices) - 1 not in marker_indices:
        marker_indices.append(len(eval_indices) - 1)
    
    # 排序索引
    marker_indices.sort()
    
    print(f"Marker indices: {marker_indices}")
    print(f"Corresponding eval numbers: {[eval_indices[i] for i in marker_indices]}")
    
    # 绘制带标记的完整收敛曲线
    # 使用学术风格的颜色搭配
    ax.plot(eval_indices, best_so_far_values, 
            color='#2171B5', linewidth=5,  # 使用更深的蓝色，经典学术配色
            marker='o', markersize=15, markerfacecolor='white',
            markeredgecolor='#d62728', markeredgewidth=5.0,  # 使用深红色边框
            markevery=marker_indices, zorder=2)
    
    # 特殊处理起点标记（使用深灰色，内部填充白色）
    if marker_indices and significant_points:
        initial_point = significant_points[0]
        initial_idx = marker_indices[0]  # 第一个标记点就是起点
        # 重新绘制起点，使用深灰色边框，白色填充
        ax.plot(eval_indices[initial_idx], best_so_far_values[initial_idx],
                marker='o', markersize=15, markerfacecolor='white',
                markeredgecolor='#2f2f2f', markeredgewidth=5.0, zorder=3)  # 使用更深的灰色
    
    # 特殊处理终点标记（使用黑色，内部填充白色）
    if marker_indices:
        final_idx = marker_indices[-1]  # 最后一个标记点就是终点
        # 检查终点是否是关键点之一
        final_eval_index = eval_indices[final_idx]
        is_final_in_significant = any(point['eval_index'] == final_eval_index for point in significant_points)
        
        # 如果终点不是关键点，单独标记为黑色
        if not is_final_in_significant:
            ax.plot(eval_indices[final_idx], best_so_far_values[final_idx],
                    marker='o', markersize=15, markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=5.0, zorder=3)
    
    # 添加标注 (只为中间的关键点添加fitness数值标注)
    # 跳过初始点(i=0)，只为中间的关键改进点添加标注
    for i, point in enumerate(significant_points[1:], 1):  # 从第二个点开始
        # 检查是否是终点
        is_final_point = point['eval_index'] == eval_indices[-1]
        
        # 只为非终点的关键点添加fitness数值标注
        if not is_final_point:
            fitness_text = f'{point["best_so_far"]:.2e}'  # 使用科学计数法显示fitness
            ax.annotate(fitness_text, 
                       (point['eval_index'], point['best_so_far']),
                       xytext=(42, 9), textcoords='offset points',  # 向右偏移35，向下偏移25
                       fontsize=16, ha='center', va='center', weight='bold',  # 增大字体到18
                       color='#d62728', zorder=6)  # 使用与标记边框一致的深红色
    
    # 不再为终点单独添加标注
    
    # Formatting - 移除标题和图例
    ax.set_xlabel('Number of Evaluations', fontsize=18, labelpad=15, weight='bold')  # 明确设置加粗
    fitness_label = 'Fitness Value' if minimize else 'Fitness (higher is better)'
    ax.set_ylabel(fitness_label, fontsize=18, labelpad=15, weight='bold')  # 明确设置加粗
    
    # 精简网格
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Set y-scale to log if appropriate (for minimization problems)
    if minimize and min(fitness_values) > 0:
        ax.set_yscale('log')
    
    # 设置刻度样式（使用深色）
    ax.tick_params(direction='out', length=8, width=1.5, colors='#2f2f2f')  # 使用深灰色刻度线
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in multiple formats
    base_path = os.path.join(output_dir, 'convergence')
    
    # PDF (vector)
    plt.savefig(f'{base_path}.pdf', bbox_inches='tight')
    print(f"Saved convergence plot: {base_path}.pdf")
    
    # PNG (raster, high DPI)
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot: {base_path}.png")
    
    # SVG (vector)
    plt.savefig(f'{base_path}.svg', bbox_inches='tight')
    print(f"Saved convergence plot: {base_path}.svg")
    
    plt.close()
    
    # Generate summary statistics
    stats = {
        'total_evaluations': len(evaluations),
        'significant_points': len(significant_points),
        'initial_fitness': fitness_values[0] if fitness_values else None,
        'final_fitness': best_so_far_values[-1] if best_so_far_values else None,
        'best_fitness': min(best_so_far_values) if minimize else max(best_so_far_values),
        'improvement_rate': len(significant_points) / len(evaluations) if evaluations else 0
    }
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'convergence_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved convergence statistics: {stats_file}")


def main():
    """Main function to analyze EvoSR-LLM convergence on oscillator1."""
    
    parser = argparse.ArgumentParser(description='Analyze EvoSR-LLM convergence curve')
    parser.add_argument('--samples_dir', type=str, 
                       default='./evosr-llm_results/gpt-4o-mini/oscillator1/samples',
                       help='Path to samples directory')
    parser.add_argument('--output_dir', type=str, default='./convergence_analysis',
                       help='Output directory for results')
    parser.add_argument('--minimize', action='store_true', default=True,
                       help='Whether lower fitness is better (default: True)')
    parser.add_argument('--maximize', action='store_true', default=False,
                       help='Whether higher fitness is better')
    
    args = parser.parse_args()
    
    # Determine optimization direction
    minimize = not args.maximize
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("EvoSR-LLM Convergence Analysis")
    print("="*60)
    print(f"Samples directory: {args.samples_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Optimization mode: {'Minimize' if minimize else 'Maximize'}")
    print()
    
    try:
        # Load evaluation data
        print("Loading evaluation data...")
        evaluations = load_evaluation_data(args.samples_dir)
        
        if not evaluations:
            print("Error: No valid evaluations found!")
            return
        
        # Compute best-so-far sequence
        print("Computing best-so-far sequence...")
        evaluations = compute_best_so_far(evaluations, minimize=minimize)
        
        # Identify significant improvement points
        print("Identifying significant improvement points...")
        significant_points = identify_significant_improvements(evaluations, minimize=minimize, max_points=4)  # 改回4个点
        
        # Copy key improvement nodes
        print("Copying significant improvement files...")
        key_nodes_dir = copy_key_nodes(significant_points, args.output_dir)
        
        # Export trace CSV
        print("Exporting trace data...")
        trace_file = export_trace_csv(evaluations, args.output_dir)
        
        # Generate convergence plots
        print("Generating convergence plots...")
        plot_convergence(evaluations, significant_points, args.output_dir, minimize=minimize)
        
        print()
        print("="*60)
        print("Analysis Complete!")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print(f"- Convergence plots: convergence.pdf/png/svg")
        print(f"- Trace data: trace.csv")
        print(f"- Key improvement files: key_nodes/")
        print(f"- Statistics: convergence_stats.json")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())