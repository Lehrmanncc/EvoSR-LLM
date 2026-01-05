# EvoSR-LLM

Code and datasets used for the EvoSR-LLM experiments.

## Datasets

Exact training and evaluation data used in the paper:
- `oes_data/` (4 datasets)
- `llm_srbench_data/` (8 datasets)


<!-- DATASET_SUMMARY_START -->
## Dataset Details

Each dataset folder contains `train.csv`, `test_id.csv` (in-distribution), and `test_ood.csv` (out-of-distribution).
Ranges are computed per column for each split.

Summary format:
- `split` uses row counts and per-column `[min, max]` ranges.
- Column order matches the CSV header order.
<details>
<summary><strong>oes:bactgrow</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 7500 | `b`: [1.0002, 4.99797], `s`: [1.00034, 4.99893], `temp`: [1.00175, 39.9766], `pH`: [2.00047, 11.9981], `r`: [-0.00276957, 1.55335] |
| test_id | 2500 | `b`: [1.00286, 4.99782], `s`: [1.0009, 4.99932], `temp`: [1.00711, 39.9767], `pH`: [2.00264, 11.9983], `r`: [-0.00201942, 1.87825] |
| test_ood | 15000 | `b`: [1.00027, 9.99909], `s`: [1.00014, 9.99988], `temp`: [1.00707, 59.9996], `pH`: [1.00027, 13.9997], `r`: [-0.00643983, 4.04588] |
</details>

<details>
<summary><strong>oes:oscillator1</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 10000 | `x`: [-0.572822, 0.527513], `v`: [-0.17543, 0.282136], `a`: [-0.104675, 0.110646] |
| test_id | 10000 | `x`: [-0.572822, 0.527513], `v`: [-0.17543, 0.282136], `a`: [-0.104675, 0.110646] |
| test_ood | 10000 | `x`: [-0.732311, 1.10205], `v`: [-0.22538, 0.5], `a`: [-0.267752, 0.139038] |
</details>

<details>
<summary><strong>oes:oscillator2</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 10000 | `t`: [30.0016, 50], `x`: [-0.205633, 0.174939], `v`: [-0.339117, 0.36241], `a`: [-0.710829, 0.660384] |
| test_id | 10000 | `t`: [30.0006, 49.999], `x`: [-0.205782, 0.174939], `v`: [-0.339116, 0.36241], `a`: [-0.710829, 0.660847] |
| test_ood | 10000 | `t`: [0, 19.9984], `x`: [-0.40323, 0.535355], `v`: [-1.01595, 0.881026], `a`: [-3.54689, 1.90832] |
</details>

<details>
<summary><strong>oes:stressstrain</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 2161 | `a`: [-0.00191139, 0.963565], `temp`: [0.0666667, 1], `s`: [-0.00550688, 0.910639] |
| test_id | 1442 | `a`: [-0.00187562, 0.96356], `temp`: [0.0666667, 1], `s`: [-0.00292726, 0.910668] |
| test_ood | 738 | `a`: [0.000144, 0.779163], `temp`: [0.666667, 0.666667], `s`: [-0.000759813, 0.621414] |
</details>

<details>
<summary><strong>llm_srbench:bio0</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `t`: [1, 54.0988], `P`: [1, 101.434], `dP_dt`: [-0.0711579, 26.6044] |
| test_id | 500 | `t`: [1.03541, 54.087], `P`: [1.06871, 101.434], `dP_dt`: [-0.0711564, 26.604] |
| test_ood | 500 | `t`: [54.1106, 60], `P`: [101.266, 101.431], `dP_dt`: [-0.0679685, 0.100564] |
</details>

<details>
<summary><strong>llm_srbench:bio1</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `t`: [1, 54.0988], `P`: [1, 996.63], `dP_dt`: [0.328032, 48.7578] |
| test_id | 500 | `t`: [1.03541, 54.0398], `P`: [1.01167, 993.754], `dP_dt`: [0.331314, 48.6709] |
| test_ood | 500 | `t`: [54.1106, 60], `P`: [997.205, 1311.26], `dP_dt`: [48.7752, 58.0806] |
</details>

<details>
<summary><strong>llm_srbench:chem0</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `t`: [0, 53.9988], `A`: [1, 1.17405], `dA_dt`: [-0.00042314, 0.0728384] |
| test_id | 500 | `t`: [0.0360072, 53.9868], `A`: [1.00261, 1.17405], `dA_dt`: [-0.000423039, 0.0720384] |
| test_ood | 500 | `t`: [54.0108, 60], `A`: [1.17208, 1.17305], `dA_dt`: [9.8428e-05, 0.000606112] |
</details>

<details>
<summary><strong>llm_srbench:chem1</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `t`: [0, 53.9988], `A`: [0.555983, 1], `dA_dt`: [-0.97578, 0.000672173] |
| test_id | 500 | `t`: [0.0360072, 53.9388], `A`: [0.556016, 0.966421], `dA_dt`: [-0.890898, 0.000610964] |
| test_ood | 500 | `t`: [54.0108, 60], `A`: [0.556122, 0.556785], `dA_dt`: [-0.000800715, 0.0004167] |
</details>

<details>
<summary><strong>llm_srbench:matsci0</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `epsilon`: [0, 0.539988], `T`: [273, 542.994], `sigma`: [-4.2944, 333.409] |
| test_id | 500 | `epsilon`: [0.000360072, 0.539868], `T`: [273.18, 542.934], `sigma`: [-4.29394, 333.29] |
| test_ood | 500 | `epsilon`: [0.540108, 0.6], `T`: [543.054, 573], `sigma`: [333.527, 393.688] |
</details>

<details>
<summary><strong>llm_srbench:matsci1</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `epsilon`: [0, 0.539988], `T`: [273, 542.994], `sigma`: [0, 49.4446] |
| test_id | 500 | `epsilon`: [0.000360072, 0.539388], `T`: [273.18, 542.694], `sigma`: [6.51698, 49.4381] |
| test_ood | 500 | `epsilon`: [0.540108, 0.6], `T`: [543.054, 573], `sigma`: [45.5156, 51.4314] |
</details>

<details>
<summary><strong>llm_srbench:phys0</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `x`: [-1.28883, 1.27468], `t`: [0, 35.9992], `v`: [-1.20459, 1.21071], `dv_dt`: [-2.02194, 1.94111] |
| test_id | 500 | `x`: [-1.28813, 1.25398], `t`: [0.0240048, 35.9912], `v`: [-1.20323, 1.21045], `dv_dt`: [-1.97971, 1.93976] |
| test_ood | 500 | `x`: [-1.23785, 1.22719], `t`: [36.0072, 40], `v`: [-1.05431, 1.0849], `dv_dt`: [-1.70105, 1.70515] |
</details>

<details>
<summary><strong>llm_srbench:phys1</strong></summary>

| split | rows | column ranges |
| --- | ---: | --- |
| train | 4000 | `x`: [-2.1863, 2.22307], `t`: [0, 35.9992], `dv_dt`: [-2.42007, 2.39115] |
| test_id | 500 | `x`: [-2.17721, 2.22299], `t`: [0.0240048, 35.9592], `dv_dt`: [-2.42011, 2.38883] |
| test_ood | 500 | `x`: [-0.737127, 0.987888], `t`: [36.0072, 40], `dv_dt`: [-1.30825, 1.45279] |
</details>

<!-- DATASET_SUMMARY_END -->
