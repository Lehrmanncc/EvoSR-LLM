import numpy as np
import os
import json


def find_best_equ(problem_name):
    folder_path = f'./llmsr_results/{problem_name}/samples'
    best_equ_save_path = f'./llmsr_results/{problem_name}/'
    nmse_data = []
    valid_file = []
    if os.path.isdir(folder_path):
        file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]
        # sorted_files = sorted(file_list, key=lambda x: int(x.split('=')[-1].split('.')[0]))
        sorted_files = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for i, file_name in enumerate(sorted_files):

            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        obj = json_data["score"]
                        # train_nmse = json_data['train_nmse']
                        test_id_nmse = json_data['test_id_nmse']
                        test_ood_nmse = json_data['test_ood_nmse']

                        # param_code = json_data['param_code']
                        if obj is not None and test_id_nmse is not None and test_ood_nmse is not None:
                            nmse_data.append([-obj, test_id_nmse, test_ood_nmse])
                            valid_file.append(file_name)
                except Exception as e:
                    print("train_evaluate an error occurred:", e)

    nmse_data = np.asarray(nmse_data)
    best_idx = np.argmin(nmse_data[:, 0])
    best_equ_name = valid_file[best_idx]

    best_file_path = os.path.join(folder_path, best_equ_name)
    best_equ_save_path = os.path.join(best_equ_save_path, f'best_equ_{best_equ_name}')

    with open(best_file_path, 'r') as f:
        best_json_data = json.load(f)

    with open(best_equ_save_path, 'w', encoding='utf-8') as f:
        json.dump(best_json_data, f, indent=4)


# def get_convergence_curve(res_path, alg_name="evosr"):
#     nmse_data = []
#     convergence_curve = []
#     if os.path.isdir(res_path):
#         file_list = [file_name for file_name in os.listdir(res_path) if file_name.endswith('.json')]
#         if alg_name == "evosr":
#             sorted_files = sorted(file_list, key=lambda x: int(x.split('=')[-1].split('.')[0]))
#         elif alg_name == "llmsr":
#             sorted_files = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#         else:
#             print("alg_name error")
#
#         for i, file_name in enumerate(sorted_files):
#
#             if file_name.endswith('.json'):
#                 file_path = os.path.join(res_path, file_name)
#                 try:
#                     with open(file_path, 'r') as f:
#                         json_data = json.load(f)
#                         train_nmse = json_data['train_nmse']
#                         # test_id_nmse = json_data['test_id_nmse']
#                         # test_ood_nmse = json_data['test_ood_nmse']
#
#                         # param_code = json_data['param_code']
#                         if train_nmse is not None:
#                             nmse_data.append(train_nmse)
#                         else:
#                             nmse_data.append(np.inf)
#                         best_data = min(nmse_data)
#                         convergence_curve.append(best_data)
#                             # valid_file.append(file_name)
#                 except Exception as e:
#                     print("train_evaluate an error occurred:", e)
#                     nmse_data.append(np.inf)
#                     best_data = min(nmse_data)
#                     convergence_curve.append(best_data)
#     return convergence_curve

def get_pop(data):
    data_array = np.array(data)
    sorted_arr = data_array[data_array[:, 0].argsort()]

    # 获取前10行
    if len(sorted_arr) < 10:
        pop = sorted_arr.tolist()
    else:
        pop = sorted_arr[:10].tolist()
    return pop


def change_obj(pop, new_lamda):
    for ind in pop:
        if ind[1] is not None and ind[2] is not None:
            # ind['complex_reg'] = ind * ind['S']
            # ind['lamda'] = eval.lamda
            ind[0] = ind[1] + new_lamda * ind[2]
    return pop


def get_best_train_nmse(pop):
    try:
        pop_array = np.asarray(pop)
        sorted_arr = pop_array[pop_array[:, 0].argsort()]

        # 获取前10行
        train_nmse = sorted_arr[0][-1]
    except Exception as e:
        print("Error occurred while getting best train nmse:", e)
        return None

    return train_nmse

def get_convergence_curve(res_path, alg_name="evosr"):
    data = []
    nmse_data = []
    convergence_curve = []
    lamba_lst = []
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
                        obj = json_data['objective']
                        lamba = json_data['lamda']

                        mse = json_data['mse']
                        s = json_data['S']
                        train_nmse = json_data['train_nmse']

                        if i == 0 and obj is not None:
                            data.append([obj, mse, s, lamba, train_nmse])
                            lamba_lst.append(lamba)
                            best_train_nmse = data[0][-1]
                            nmse_data.append(best_train_nmse)
                        else:
                            if obj is not None and mse is not None and s is not None and lamba is not None and train_nmse is not  None:
                                if lamba != lamba_lst[-1]:
                                    lamba_lst = []
                                    lamba_lst.append(lamba)

                                    pop = get_pop(data)
                                    pop = change_obj(pop, lamba)
                                    pop.append([obj, mse, s, lamba, train_nmse])
                                    data = pop

                                    best_train_nmse = get_best_train_nmse(data)
                                else:
                                    data.append([obj, mse, s, lamba, train_nmse])
                                    best_train_nmse = get_best_train_nmse(data)

                                if best_train_nmse is not None:
                                    nmse_data.append(best_train_nmse)
                            else:
                                temp = nmse_data[-1]
                                nmse_data.append(temp)

                        # test_id_nmse = json_data['test_id_nmse']
                        # test_ood_nmse = json_data['test_ood_nmse']

                        # param_code = json_data['param_code']

                        # else:
                        #     nmse_data.append(np.inf)
                        # best_data = min(nmse_data)
                        # convergence_curve.append(best_data)
                            # valid_file.append(file_name)
                except Exception as e:
                    temp = nmse_data[-1]
                    nmse_data.append(temp)
                    print("train_evaluate an error occurred:", e)
                    # nmse_data.append(np.inf)
                    # best_data = min(nmse_data)
                    # convergence_curve.append(best_data)
    # return convergence_curve
    return nmse_data


def get_llmsr_convergence_curve(res_path, alg_name="llmsr"):
    data = []
    nmse_data = []
    convergence_curve = []
    lamba_lst = []
    if os.path.isdir(res_path):
        file_list = [file_name for file_name in os.listdir(res_path) if file_name.endswith('.json')]
        if alg_name == "llmsr":
            sorted_files = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        else:
            print("alg_name error")

        for i, file_name in enumerate(sorted_files):
            if file_name.endswith('.json'):
                file_path = os.path.join(res_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        obj = json_data['score']

                        train_nmse = json_data['train_nmse']
                        if obj is not None and train_nmse is not None:
                            data.append([-obj, train_nmse])
                            best_train_nmse = get_best_train_nmse(data)

                            nmse_data.append(best_train_nmse)
                        else:
                            temp = nmse_data[-1]
                            nmse_data.append(temp)

                except Exception as e:
                    temp = nmse_data[-1]
                    nmse_data.append(temp)
                    print("train_evaluate an error occurred:", e)
                    # nmse_data.append(np.inf)
                    # best_data = min(nmse_data)
                    # convergence_curve.append(best_data)
    # return convergence_curve
    return nmse_data

if __name__ == '__main__':
    # 找到最佳方程
    # problem_names = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    # for name in problem_names:
    #     # evosr_res_path = os.path.join("./results_final", name, "samples")
    #     # llmsr_res_path = os.path.join("./llmsr_results", name, "samples")
    #     find_best_equ(name)

    # 收敛图最终版
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatter
    import os

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        # 'axes.labelweight': 'bold',
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 22,
        'lines.linewidth': 2,  # 线条宽度
        'grid.alpha': 1,  # 网格线透明度
    })

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    problem_names = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    title_names = ["Oscillator 1", "Oscillator 2", "E. coli growth", "Stress-Strain"]
    datas_res = []
    for i, (name, ax) in enumerate(zip(problem_names, axs.flat)):
        evosr_res_path = os.path.join("./results_final", name, "samples")
        # evosr_code_res_path = os.path.join("./result_code", name, "samples")
        # evosr_operator_res_path = os.path.join("./result_operator", name, "samples")
        # evosr_reg_res_path = os.path.join("./result_reg", name, "samples")
        llmsr_res_path = os.path.join("./llmsr_results", name, "samples")

        # 假设get_convergence_curve是你的数据处理函数
        evosr_convergence = get_convergence_curve(evosr_res_path, "evosr")
        # evosr_code_convergence = get_convergence_curve(evosr_code_res_path, "evosr")
        # evosr_operator_convergence = get_convergence_curve(evosr_operator_res_path, "evosr")
        # evosr_reg_convergence = get_convergence_curve(evosr_reg_res_path, "evosr")
        llmsr_convergence = get_llmsr_convergence_curve(llmsr_res_path, "llmsr")

        ax.plot(np.arange(1, len(evosr_convergence) + 1), evosr_convergence, label='EvoSR-LLM', color='tab:blue',
                linewidth=3)
        ax.plot(np.arange(1, len(llmsr_convergence) + 1), llmsr_convergence, label='LLM-SR',
                color='tab:orange', linestyle='--', linewidth=3)
        # ax.plot(np.arange(1, len(evosr_code_convergence) + 1), evosr_code_convergence, label='EvoSR-LLM(code)',
        #         color='tab:orange', linestyle='--', linewidth=3)
        # ax.plot(np.arange(1, len(evosr_operator_convergence) + 1), evosr_operator_convergence,
        #         label='EvoSR-LLM(prompt)', color='tab:green', linestyle='-.', linewidth=3)
        # ax.plot(np.arange(1, len(evosr_reg_convergence) + 1), evosr_reg_convergence, label='EvoSR-LLM(complexity)',
        #         color='tab:blue', linestyle=':', linewidth=3)

        ax.set_title(title_names[i], fontsize=26, weight='bold')
        ax.set_xlabel('Number of Evaluations', fontsize=25)
        ax.set_ylabel('Normalized MSE', fontsize=25)
        ax.grid(True, linestyle=':', linewidth=0.5)

        # y_min = min(np.min(evosr_convergence), np.min(evosr_code_convergence), np.min(evosr_operator_convergence),
        #             np.min(evosr_reg_convergence))
        # y_max = max(np.max(evosr_convergence), np.max(evosr_code_convergence), np.max(evosr_operator_convergence),
        #             np.max(evosr_reg_convergence))
        y_min = min(np.min(evosr_convergence), np.min(llmsr_convergence))
        y_max = max(np.max(evosr_convergence), np.max(llmsr_convergence))

        log_min = np.floor(np.log10(y_min))
        log_max = np.ceil(np.log10(y_max))

        subs = [10 ** i for i in range(int(log_min), int(log_max) + 1) if i % 2 != 0]  # 自动生成subs

        ax.set_yscale('log')
        ax.set_yticks([10 ** i for i in range(int(log_min), int(log_max) + 1) if i % 2 != 0])

        ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))  # 显示1, 1e-1, 1e-2, ...
        ax.legend()

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            # 合并图例
    # handles_row1, labels_row1 = axs[0, 0].get_legend_handles_labels()  # 获取第一行的图例
    # fig.legend(handles_row1, labels_row1, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1), fontsize=18,
    #            handletextpad=0.4)

    # 自动调整布局
    plt.tight_layout(pad=1.5)  # 增加pad值以确保图例和子图不重叠
    # plt.subplots_adjust(top=0.88)
    # 保存图形
    plt.savefig("./results_final/convergence_curve_final.pdf")

    # 显示图形
    plt.show()




    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.ticker import LogLocator, LogFormatter
    # import os
    #
    # plt.rcParams.update({
    #     'font.family': 'serif',
    #     'font.serif': 'Times New Roman',
    #     'axes.titlesize': 18,
    #     'axes.labelsize': 16,
    #     # 'axes.labelweight': 'bold',
    #     'xtick.labelsize': 14,
    #     'ytick.labelsize': 14,
    #     'legend.fontsize': 16,
    #     'lines.linewidth': 2,  # 线条宽度
    #     'grid.alpha': 1,  # 网格线透明度
    # })
    #
    #
    #
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    #
    # problem_names = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    # title_names = ["Oscillator 1", "Oscillator 2", "E. coli growth", "Stress-Strain"]
    # datas_res = []
    # for i, (name, ax) in enumerate(zip(problem_names, axs.flat)):
    #     evosr_res_path = os.path.join("./results_final", name, "samples")
    #     evosr_code_res_path = os.path.join("./result_code", name, "samples")
    #     evosr_operator_res_path = os.path.join("./result_operator", name, "samples")
    #     evosr_reg_res_path = os.path.join("./result_reg", name, "samples")
    #     # llmsr_res_path = os.path.join("./llmsr_results", name, "samples")
    #
    #     # 假设get_convergence_curve是你的数据处理函数
    #     evosr_convergence = get_convergence_curve(evosr_res_path, "evosr")
    #     evosr_code_convergence = get_convergence_curve(evosr_code_res_path, "evosr")
    #     evosr_operator_convergence = get_convergence_curve(evosr_operator_res_path, "evosr")
    #     evosr_reg_convergence = get_convergence_curve(evosr_reg_res_path, "evosr")
    #     # llmsr_convergence = get_convergence_curve(llmsr_res_path, "llmsr")
    #
    #     ax.plot(np.arange(1, len(evosr_convergence) + 1), evosr_convergence, label='EvoSR-LLM', color='tab:red', linewidth=3)
    #     ax.plot(np.arange(1, len(evosr_code_convergence) + 1), evosr_code_convergence, label='EvoSR-LLM(code)', color='tab:orange', linestyle='--', linewidth=3)
    #     ax.plot(np.arange(1, len(evosr_operator_convergence) + 1), evosr_operator_convergence, label='EvoSR-LLM(prompt)', color='tab:green', linestyle='-.', linewidth=3)
    #     ax.plot(np.arange(1, len(evosr_reg_convergence) + 1), evosr_reg_convergence, label='EvoSR-LLM(complexity)', color='tab:blue', linestyle=':', linewidth=3)
    #
    #     ax.set_title(title_names[i], fontsize=22, weight='bold')
    #     ax.set_xlabel('Number of Evaluations', fontsize=20)
    #     ax.set_ylabel('Normalized MSE', fontsize=20)
    #     ax.grid(True, linestyle=':', linewidth=0.5)
    #
    #
    #     y_min = min(np.min(evosr_convergence), np.min(evosr_code_convergence), np.min(evosr_operator_convergence), np.min(evosr_reg_convergence))
    #     y_max = max(np.max(evosr_convergence), np.max(evosr_code_convergence), np.max(evosr_operator_convergence), np.max(evosr_reg_convergence))
    #
    #     log_min = np.floor(np.log10(y_min))
    #     log_max = np.ceil(np.log10(y_max))
    #
    #
    #     subs = [10 ** i for i in range(int(log_min), int(log_max) + 1) if i % 2 != 0]  # 自动生成subs
    #
    #     ax.set_yscale('log')
    #     ax.set_yticks([10 ** i for i in range(int(log_min), int(log_max) + 1) if i % 2 != 0])
    #
    #     ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))  # 显示1, 1e-1, 1e-2, ...
    #     # ax.legend()
    #
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(2)
    #         # 合并图例
    # handles_row1, labels_row1 = axs[0, 0].get_legend_handles_labels()  # 获取第一行的图例
    # fig.legend(handles_row1, labels_row1, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1), fontsize=18, handletextpad=0.4)
    #
    # # 自动调整布局
    # plt.tight_layout(pad=1.5)  # 增加pad值以确保图例和子图不重叠
    # plt.subplots_adjust(top=0.88)
    # # 保存图形
    # # plt.savefig("./results_final/Ablation_convergence_curve.pdf")
    #
    # # 显示图形
    # plt.show()





