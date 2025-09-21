"""
独立的 llm_srbench 数据加载器
用于测试新 benchmark 的兼容性
"""
import numpy as np
from llm_srbench.datamodules import (
    BioPopGrowthDataModule, 
    ChemReactKineticsDataModule, 
    PhysOscilDataModule, 
    MatSciDataModule, 
    TransformedFeynmanDataModule
)


class LLMSRBenchLoader:
    def __init__(self, problem_name, idx=0):
        self.problem_name = problem_name      # 具体问题类型：phys_osc, matsci, bio_pop_growth, chem_react
        self.idx = idx                        # 该问题类型下的具体实例索引
        self.var_name = None
        self.output_name = None
    
    def get_problem(self):
        """加载 llm_srbench 数据集"""
        # 1) 根据 problem_name 选择数据模块（现在 problem_name 是具体的问题类型）
        if self.problem_name in ("bio_pop_growth", "bio", "BPG"):
            dm = BioPopGrowthDataModule(root="datasets/lsr-synth-bio")
        elif self.problem_name in ("chem_react", "chem", "CRK"):
            dm = ChemReactKineticsDataModule(root="datasets/lsr-synth-chem")
        elif self.problem_name in ("phys_osc", "phys", "PO"):
            dm = PhysOscilDataModule(root="datasets/lsr-synth-phys")
        elif self.problem_name in ("matsci", "MatSci"):
            dm = MatSciDataModule(root="datasets/lsr-synth-matsci")
        elif self.problem_name == "lsr_transform":
            dm = TransformedFeynmanDataModule()
        else:
            raise ValueError(f"Unknown problem type: {self.problem_name}")

        print(f"Setting up {self.problem_name}...")
        dm.setup()
        print(f"Loaded {len(dm.problems)} problems")

        # 2) 选择具体实例（使用 idx 参数）
        if self.idx >= len(dm.problems):
            raise IndexError(f"Index {self.idx} out of range for {len(dm.problems)} problems")
        
        p = dm.problems[self.idx]
        print(f"Selected problem instance {self.idx}: {p.equation_idx}")

        # 3) 从 symbols 获取变量名（第一个是输出，其余是输入）
        symbols = p.gt_equation.symbols
        self.output_name = [symbols[0]]
        self.var_name = symbols[1:]
        
        print(f"Output variable: {self.output_name}")
        print(f"Input variables: {self.var_name}")
        print(f"Ground truth equation: {p.gt_equation.expression}")

        # 4) 获取数据
        train = p.train_samples
        test = p.test_samples
        ood = p.ood_test_samples
        
        print(f"Train data type: {type(train)}, shape: {np.asarray(train).shape}")
        print(f"Test data type: {type(test)}, shape: {np.asarray(test).shape}")
        if ood is not None:
            print(f"OOD data type: {type(ood)}, shape: {np.asarray(ood).shape}")
        
        # 5) 转换数据格式以兼容旧接口
        def convert_data(data, name="data"):
            if data is None:
                return None
            
            # 直接转为 numpy 数组
            result = np.asarray(data)
            # print(f"Converting {name} to array: shape {result.shape}")
            
            # 重排列顺序：[output, inputs...] -> [inputs..., output] 
            # 这样兼容旧代码的 X = data[:, :-1], y = data[:, -1] 模式
            if result.ndim == 2 and result.shape[1] == len(symbols):
                reorder = list(range(1, len(symbols))) + [0]
                result = result[:, reorder]
                print(f"  Reordered to [inputs..., output]: shape {result.shape}")
            
            return result

        train_data = convert_data(train, "train")
        test_data = convert_data(test, "test")
        ood_data = convert_data(ood, "ood") if ood is not None else None
        
        # 6) 返回与旧接口一致的格式
        if ood_data is None:
            # 只有 train/test 的情况
            dataset = {'train_data': train_data, 'test_data': test_data}
            print("Dataset structure: train_data, test_data")
        else:
            # 有 train/test/ood 的情况
            dataset = {'train_data': train_data, 'valid_data': test_data, 'test_data': ood_data}
            print("Dataset structure: train_data, valid_data, test_data")
        
        return dataset

    def print_problem_info(self):
        """打印问题的基本信息"""
        print(f"Problem Type: {self.problem_name}")
        print(f"Instance Index: {self.idx}")
        print(f"Output: {self.output_name}")
        print(f"Inputs: {self.var_name}")

    def list_available_problems(self):
        """列出可用的问题实例"""
        if self.problem_name in ("bio_pop_growth", "bio", "BPG"):
            dm = BioPopGrowthDataModule(root="datasets/lsr-synth-bio")
        elif self.problem_name in ("chem_react", "chem", "CRK"):
            dm = ChemReactKineticsDataModule(root="datasets/lsr-synth-chem")
        elif self.problem_name in ("phys_osc", "phys", "PO"):
            dm = PhysOscilDataModule(root="datasets/lsr-synth-phys")
        elif self.problem_name in ("matsci", "MatSci"):
            dm = MatSciDataModule(root="datasets/lsr-synth-matsci")
        elif self.problem_name == "lsr_transform":
            dm = TransformedFeynmanDataModule()
        else:
            raise ValueError(f"Unknown problem type: {self.problem_name}")
        
        dm.setup()
        print(f"Available problem instances in {self.problem_name}:")
        for i, p in enumerate(dm.problems[:10]):  # 只显示前10个
            print(f"  {i}: {p.equation_idx} - {p.gt_equation.expression}")
        if len(dm.problems) > 10:
            print(f"  ... and {len(dm.problems) - 10} more")


def test_benchmark(problem_name, idx=0):
    """测试函数"""
    print(f"\n{'='*50}")
    print(f"Testing llm_srbench - {problem_name} - instance {idx}")
    print(f"{'='*50}")
    
    try:
        loader = LLMSRBenchLoader(problem_name, idx)
        
        # 列出可用问题实例
        loader.list_available_problems()
        print()
        
        # 加载数据
        dataset = loader.get_problem()
        
        # 打印基本信息
        loader.print_problem_info()
        
        # 打印数据形状
        print("\nDataset shapes:")
        for key, data in dataset.items():
            print(f"  {key}: {data.shape}")
            
        return True, loader, dataset
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None, None


if __name__ == "__main__":
    # 测试所有问题类型
    problem_types = ["bio_pop_growth", "chem_react", "phys_osc", "matsci"]
    
    for problem_type in problem_types:
        # 测试每个问题类型的第一个实例
        success, loader, dataset = test_benchmark(problem_type, 0)
        if success:
            print(f"✅ {problem_type} loaded successfully")
        else:
            print(f"❌ {problem_type} failed to load")
        print()
    
    # 额外测试：同一问题类型的不同实例
    print(f"\n{'='*50}")
    print("Testing multiple instances of phys_osc")
    print(f"{'='*50}")
    for i in range(3):
        success, loader, dataset = test_benchmark("phys_osc", i)
        if success:
            print(f"✅ phys_osc instance {i} loaded successfully")
        else:
            print(f"❌ phys_osc instance {i} failed to load")
