# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Poster Layout RL dataset to VERL format
从data-RL-fixed目录读取parquet文件并转换为VERL训练格式
"""

import argparse
import os
import json

import datasets
import pandas as pd

def preprocess_poster_layout_rl_data():
    """
    将海报布局RL数据集转换为VERL格式
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/opt/liblibai-models/user-workspace/jiazhewei/data-RL-fixed")
    parser.add_argument("--output_dir", default="/opt/liblibai-models/user-workspace/jiazhewei/verl-data/poster-layout")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取训练和测试数据
    train_df = pd.read_parquet(os.path.join(args.input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(args.input_dir, "test.parquet"))
    
    def process_row(row, idx, split):
        """
        处理单行数据，转换为VERL格式
        如果图像数量超过阈值，返回None进行过滤
        """
        # 首先检查图像数量，提前过滤
        MAX_IMAGES = 8  # Qwen2.5-VL处理器限制
        images_array = row['images'] if isinstance(row['images'], list) else row['images'].tolist()
        
        if len(images_array) > MAX_IMAGES:
            print(f"过滤样本 {idx}: 图像数量 {len(images_array)} > {MAX_IMAGES}")
            return None  # 返回None，后续会被过滤掉
        
        # 解析JSON字段
        prompt_data = json.loads(row['prompt'])
        reward_model_data = json.loads(row['reward_model'])
        extra_info_data = json.loads(row['extra_info'])
        
        # 将原始多模态格式转换为Python原生格式（避免numpy数组）
        def convert_to_native_format(data):
            """递归转换numpy数组为Python原生类型"""
            if isinstance(data, dict):
                return {k: convert_to_native_format(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_to_native_format(item) for item in data]
            elif hasattr(data, 'tolist'):  # numpy数组
                return convert_to_native_format(data.tolist())
            else:
                return data
        
        # 转换为原生Python格式
        verl_prompt = convert_to_native_format(prompt_data)
        
        # 转换images格式为VERL期望的字典格式（原生Python格式）
        verl_images = []
        images_array = row['images'] if isinstance(row['images'], list) else row['images'].tolist()
        for img_path in images_array:
            verl_images.append({"image": str(img_path)})
        
        # 构建VERL格式的数据
        data = {
            "data_source": row['data_source'],  # "poster_layout_synthetic"
            "prompt": verl_prompt,  # 转换为VERL格式的对话
            "images": verl_images,  # 转换为字典格式的图像列表
            "ability": row['ability'],  # 任务类型
            "reward_model": convert_to_native_format(reward_model_data),  # 包含ground_truth的奖励模型配置
            "extra_info": {
                **convert_to_native_format(extra_info_data),  # 转换为原生格式
                "split": split,
                "verl_index": idx,
                "caption_structure": str(row['caption_structure']),  # 结构要求
                "width": int(row['width']),
                "height": int(row['height']),
                "total_layers": int(row['total_layers']),
                "original_prompt": json.dumps(prompt_data),  # 保留原始的多模态格式供reward函数使用
                "original_images": images_array  # 保留原始图像路径列表供reward函数使用
            }
        }
        return data
    
    # 转换训练集（过滤无效样本）
    train_data = []
    filtered_train_count = 0
    for idx, row in train_df.iterrows():
        processed_row = process_row(row, idx, "train")
        if processed_row is not None:
            train_data.append(processed_row)
        else:
            filtered_train_count += 1
    
    # 转换测试集（过滤无效样本）
    test_data = []
    filtered_test_count = 0
    for idx, row in test_df.iterrows():
        processed_row = process_row(row, idx, "test")
        if processed_row is not None:
            test_data.append(processed_row)
        else:
            filtered_test_count += 1
    
    # 创建datasets时确保使用原生Python格式
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)
    
    # 定义后处理函数，将numpy数组转换为原生Python格式
    def post_process_sample(sample):
        """将样本中的numpy数组转换为原生Python格式"""
        def convert_numpy_to_native(data):
            if isinstance(data, dict):
                return {k: convert_numpy_to_native(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_numpy_to_native(item) for item in data]
            elif hasattr(data, 'tolist'):  # numpy数组
                return convert_numpy_to_native(data.tolist())
            else:
                return data
        
        return convert_numpy_to_native(sample)
    
    # 应用后处理
    train_dataset = train_dataset.map(post_process_sample)
    test_dataset = test_dataset.map(post_process_sample)
    
    # 保存为parquet格式
    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))
    
    print(f"数据预处理完成！")
    print(f"训练集: {len(train_df)} -> {len(train_data)} 样本 (过滤掉 {filtered_train_count} 个)")
    print(f"测试集: {len(test_df)} -> {len(test_data)} 样本 (过滤掉 {filtered_test_count} 个)")
    print(f"过滤原因: 图像数量 > 8")
    print(f"输出目录: {args.output_dir}")
    
    # 打印一个样本示例用于验证
    print("\n=== 样本示例 ===")
    sample = train_data[0]
    print(f"data_source: {sample['data_source']}")
    print(f"ability: {sample['ability']}")
    print(f"images count: {len(sample['images'])}")
    print(f"prompt roles: {[msg['role'] for msg in sample['prompt']]}")
    print(f"ground_truth type: {type(sample['reward_model']['ground_truth'])}")
    if isinstance(sample['reward_model']['ground_truth'], dict):
        print(f"ground_truth keys: {list(sample['reward_model']['ground_truth'].keys())}")
    else:
        print(f"ground_truth preview: {str(sample['reward_model']['ground_truth'])[:200]}...")
    print(f"extra_info keys: {list(sample['extra_info'].keys())}")

if __name__ == "__main__":
    preprocess_poster_layout_rl_data()
