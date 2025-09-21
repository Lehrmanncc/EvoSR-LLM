import numpy as np
import os
import json


def get_complex_data(res_path, alg_name="evosr"):
    c_data = []
    convergence_curve = []
    if os.path.isdir(res_path):
        file_list = [file_name for file_name in os.listdir(res_path) if file_name.endswith('.json')]
        if alg_name == "evosr":
            sorted_files = sorted(file_list, key=lambda x: int(x.split('=')[-1].split('.')[0]))
        elif alg_name == "llmsr":
            sorted_files = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        else:
            print("alg_name error")

        for i, file_name in enumerate(sorted_files):

            if file_name.endswith('.json'):
                file_path = os.path.join(res_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        # train_nmse = json_data['train_nmse']
                        c = json_data["complex"]
                        # test_id_nmse = json_data['test_id_nmse']
                        # test_ood_nmse = json_data['test_ood_nmse']

                        # param_code = json_data['param_code']
                        if c is not None:
                            c_data.append(c)
                        # else:
                        #     c.append(np.inf)
                        # best_data = min(nmse_data)
                        # convergence_curve.append(best_data)
                            # valid_file.append(file_name)
                except Exception as e:
                    print("train_evaluate an error occurred:", e)
                    # nmse_data.append(np.inf)
                    # best_data = min(nmse_data)
                    # convergence_curve.append(best_data)
    return c_data



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    problem_names = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    title_names = ["Oscillator 1", "Oscillator 2", "E. coli growth", "Stress-Strain"]
    datas_res = []
    x = np.arange(1, 3002)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for i, (name, ax) in enumerate(zip(problem_names, axs.flat)):
        evosr_res_path = os.path.join("./results_final", name, "samples")
        llmsr_res_path = os.path.join("./llmsr_results", name, "samples")

        evosr_c = get_complex_data(evosr_res_path, "evosr")
        llmsr_c = get_complex_data(llmsr_res_path, "llmsr")

        bar_width = 0.4
        num_bins = 5
        counts1, bin_edges1 = np.histogram(evosr_c, bins=num_bins)
        x1_positions = bin_edges1[:-1] + np.diff(bin_edges1) / 2

        counts2, bin_edges2 = np.histogram(llmsr_c, bins=num_bins)
        x2_positions = bin_edges2[:-1] + np.diff(bin_edges2) / 2

        # x1_positions = np.array(evosr_c) - bar_width / 2
        # x2_positions = np.array(llmsr_c) + bar_width / 2

        ax.bar(x1_positions, counts1, width=np.diff(bin_edges1), color='skyblue', edgecolor='black', linewidth=0.6, label="EvoSR-LLM")
        ax.bar(x2_positions, counts2, width=np.diff(bin_edges2), color='orange', edgecolor='black', linewidth=0.6, label="LLM-SR")

        ax.legend()
        ax.set_title("Evaluation Counts Distribution")
        ax.set_xlabel("Evaluation Counts")
        ax.set_ylabel("Number of Configurations")

        # ax.set_xticks(range(min(min(evosr_c), min(llmsr_c)), max(max(evosr_c), max(llmsr_c)) + 1))
        # ax.set_yticks()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.savefig("{}/{}.pdf".format(save_path, save_name), bbox_inches='tight')
    plt.show()
