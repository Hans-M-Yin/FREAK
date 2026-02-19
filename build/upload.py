import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image
import datasets
from datasets import Dataset, DatasetDict
from dataclasses import dataclass

from tqdm import tqdm

@dataclass
class DatasetConfig:
    """数据集配置类"""
    image_dir: str  # 图片文件夹路径
    json_path: list[str]  # JSON文件路径


class CustomImageDatasetProcessor:
    """自定义图像数据集处理器"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.image_dir = config.image_dir
        self.json_path = [Path(_path) for _path in config.json_path]


    def detect_single_item_type(self, item:Dict) -> str:
        """自动检测数据集类型"""

        if "options" in item:
            return "mcq"  # 选择题
        elif "hallu_answer" in item:
            return "fib"  # 填空题

    def load_and_process_data(self) -> List[Dict]:
        """加载并处理数据"""
        # 1. 读取JSON文件
        data = []
        for _path in self.json_path:
            with open(_path, 'r', encoding='utf-8') as f:
                data += json.load(f)
        print("")
        if not isinstance(data, list):
            data = [data]

        # 2. 自动检测数据类型（如果未指定）


        # 3. 处理每条数据
        processed_data = []
        for item in tqdm(data):
            processed_item = self._process_single_item(item, self.detect_single_item_type(item))
            if processed_item:
                processed_data.append(processed_item)

        return processed_data

    def _process_single_item(self, item: Dict, dataset_type: str) -> Dict:
        """处理单个数据项"""
        try:

            image_path = Path(os.path.join(self.image_dir, item.get("image_path", "")))
            # if not image_path.exists():
            #     # 尝试不同的扩展名
            #     for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
            #         alt_path = self.image_dir / f"{Path(item['image_path']).stem}{ext}"
            #         if alt_path.exists():
            #             image_path = alt_path
            #             break

            # 读取图片
            image = Image.open(image_path).convert("RGB")


            # 2. 构建基础字段
            processed_item = {
                "id": item.get("id", 0),
                "image": image,
                "question": item.get("question", ""),
                "item": item.get("item", ""),
                "category": item.get("category", ""),
                "ground_truth": item.get("ground_truth", "")
            }
            if not isinstance(processed_item['category'], list):
                processed_item['category'] = [processed_item['category']]
            # 3. 根据数据类型添加特定字段
            if dataset_type == "mcq":
                processed_item["options"] = item.get("options", [])
                processed_item["hallu_answer"] = None
                processed_item["type"] = "mcq"
            elif dataset_type == "fib":
                processed_item["hallu_answer"] = item.get("hallu_answer", "")
                processed_item["options"] = None
                processed_item["type"] = "fib"

            return processed_item

        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            return None

    def create_huggingface_dataset(self, split_ratio: Dict[str, float] = None) -> DatasetDict:
        """创建HuggingFace数据集"""

        # 加载和处理数据
        print("你好")
        processed_data = self.load_and_process_data()
        print("结束")

        if not processed_data:
            raise ValueError("No valid data processed")

        mcq_len, fib_len = 0, 0
        for item in processed_data:
            if item["type"] == "mcq":
                mcq_len += 1
            else:
                fib_len += 1
        # 创建Dataset
        for i in processed_data:
            print(i)
        dataset = Dataset.from_list(processed_data)

        dataset = Dataset.from_list(processed_data)

        # 设置特征类型
        features = datasets.Features({
            "id": datasets.Value("int64"),
            "image": datasets.Image(),
            "options": datasets.Sequence(datasets.Value("string")),
            "hallu_answer": datasets.Value("string"),
            "question": datasets.Value("string"),
            "item": datasets.Value("string"),
            "category": datasets.Sequence(datasets.Value("string")),
            "ground_truth": datasets.Value("string"),
            "type": datasets.Value("string"),
        })
        dataset = dataset.cast(features)
        print("难过")

        # 划分数据集（如果需要）
        if split_ratio:
            splits = self._split_dataset(dataset, split_ratio)
            return splits
        else:
            return DatasetDict({"test": dataset})



def process_multiple_datasets(configs: List[DatasetConfig]) -> DatasetDict:
    """处理多个数据集并合并"""
    all_datasets = {}

    for i, config in enumerate(configs):
        processor = CustomImageDatasetProcessor(config)
        dataset_dict = processor.create_huggingface_dataset()

        # 为每个数据集添加前缀
        for split_name, dataset in dataset_dict.items():
            dataset_name = f"dataset_{i}_{split_name}"
            all_datasets[dataset_name] = dataset

    return DatasetDict(all_datasets)


# 使用示例
if __name__ == "__main__":
    # 示例1: 处理选择题数据集
    mcq_config = DatasetConfig(
        image_dir="./generated_dataset/final_part2",  # 选择题图片文件夹
        json_path=["dataset.json","dataset_qa.json"],  # 选择题JSON文件
    )

    processor = CustomImageDatasetProcessor(mcq_config)

    # 创建数据集并划分（例如：80%训练，10%验证，10%测试）
    dataset_dict = processor.create_huggingface_dataset(
        split_ratio=None
    )

    # 查看数据集信息
    print("数据集结构:", dataset_dict)
    # print("训练集样本数:", len(dataset_dict["train"]))

    # 查看第一个样本
    sample = dataset_dict["test"][0]
    print("\n样本示例:")
    for key, value in sample.items():
        if key != "image":
            print(f"{key}: {value}")
    print(f"image shape: {sample['image'].size}")

    # 保存到本地
    save_path = "./camera_ready/freak"
    dataset_dict.push_to_hub("hansQAQ/FREAK")

    dataset_dict.save_to_disk(save_path)
    print(f"\n数据集已保存到: {save_path}")

    # 从本地加载
    loaded_dataset = datasets.load_from_disk(save_path)

    # # 或者处理多个数据集
    # all_configs = [mcq_config, fib_config]
    # combined_datasets = process_multiple_datasets(all_configs)