# import numpy as np
#
#
# def equation(x, v):
#     z = x * v
#     log_term = 0.0 * np.log(np.abs(v))
#     exp_term = 0.0 * np.exp(0.22006 * (x + v))
#     # sin_term = -0.31063 * np.arcsin(x)
#     # cos_term = -0.0 * np.arccos(v)
#     poly_term = 0.2217 * x ** 3 + -0.49997 * v ** 3 + -1.5 * z
#     # tanh_term = 0.28705 * np.tanh(0.38574 * x)
#
#     # a = z + log_term + exp_term + sin_term + cos_term + poly_term + tanh_term
#     # a = z + log_term + exp_term + cos_term + poly_term + tanh_term
#     # a = z + log_term + exp_term + poly_term + tanh_term
#     a = z + log_term + exp_term + poly_term
#
#     return a


import numpy as np


def equation(x, v):
    a = (-0.19771 * x + -0.0037 * v - -0.15179 * x ** 3 +
         -0.19793 * v * np.sin(-0.13154 * x) +
         # 5e-05 * np.log(np.abs(x - v)) +
         -0.00906 * np.cos(1.01028 * x * v) +
         -0.00119 * np.exp(x + v) * np.tanh(np.sqrt(np.abs(x - v))) +
         -0.46839 * v ** 3 +
         -0.0 * (np.exp(v) / x) +
         0.00152 * np.exp(x ** 2) +
         # 0.00093 * v ** 2 * np.log(np.abs(x - v)) +
         0.00796 * np.exp(0.68568 * v * np.sqrt(np.abs(x - v))) +
         -0.52666 * np.tanh(np.sin(x * v))
         )

    return a


if __name__ == '__main__':
    import pandas as pd

    problem_names = ["oscillator1"]
    datas_res = []
    for name in problem_names:
        train_data = pd.read_csv(f"./Problems/llm_sr/{name}" + '/train.csv')
        var_name = train_data.columns.tolist()[:-1]
        train_data = np.array(train_data)

        valid_data = np.array(pd.read_csv(f"./Problems/llm_sr/{name}" + '/test_id.csv'))
        test_data = np.array(pd.read_csv(f"./Problems/llm_sr/{name}" + '/test_ood.csv'))

        valid_y_pred = equation(valid_data[:, 0], valid_data[:, 1])
        test_y_pred = equation(test_data[:, 0], test_data[:, 1])

        valid_nmse = np.mean((valid_data[:, -1] - valid_y_pred) ** 2) / np.var(valid_data[:, -1])
        test_nmse = np.mean((test_data[:, -1] - test_y_pred) ** 2) / np.var(test_data[:, -1])

        print(f"test_id nmse:{valid_nmse}")
        print(f"test_ood nmse:{test_nmse}")

    # EvoSR-LLM
    # o1_test_id_nmse = [5.360996340085795e-08, 3.5255362101916185e-08, 3.2778069665624944e-07]
    # o1_test_ood_nmse = [0.0001862782561421872, 0.0007191136396658971, 0.008203001607390369]
    #
    # print(np.mean(o1_test_id_nmse), np.std(o1_test_id_nmse))
    # print(np.mean(o1_test_ood_nmse), np.std(o1_test_ood_nmse))

    # stress_test_id_nmse = [0.019324594710243153, 0.008860178867223614, 0.030261719196738226]
    # stress_test_ood_nmse = [0.061296518779785196, 0.09421690163872479, 0.12007367920248951]
    #
    # print(np.mean(stress_test_id_nmse), np.std(stress_test_id_nmse))
    # print(np.mean(stress_test_ood_nmse), np.std(stress_test_ood_nmse))
    # o2_test_id_nmse = [1.700455745661623e-10, 2.3103215266956347e-09, 2.7557616773996415e-11]
    # o2_test_ood_nmse = [1.6985761467700783e-10, 1.3539663800270273e-09, 6.516646426481079e-11]
    #
    # print(np.mean(o2_test_id_nmse), np.std(o2_test_id_nmse))
    # print(np.mean(o2_test_ood_nmse), np.std(o2_test_ood_nmse))

    # LLM-SR
    # o1_test_id_nmse = [1.4673457815905286e-06, 1.1110473715698525e-06, 1.6831305251267916e-06]
    # o1_test_ood_nmse = [0.02579009067385863, 0.02052588057711773, 0.03169635436016271]
    #
    # print(np.mean(o1_test_id_nmse), np.std(o1_test_id_nmse))
    # print(np.mean(o1_test_ood_nmse), np.std(o1_test_ood_nmse))

    # o2_test_id_nmse = [1.5780911477050615e-06, 1.0016615076719885e-05, 1.7250089185083506e-06]
    # o2_test_ood_nmse = [0.1374589585387821, 0.0015980095021463127, 0.00024042256940311473]
    #
    # print(np.mean(o2_test_id_nmse), np.std(o2_test_id_nmse))
    # print(np.mean(o2_test_ood_nmse), np.std(o2_test_ood_nmse))
