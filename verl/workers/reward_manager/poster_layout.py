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
PosterLayoutRewardManager - 专门用于海报布局生成任务的奖励管理器
"""

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("poster_layout")
class PosterLayoutRewardManager(AbstractRewardManager):
    """海报布局生成任务专用的奖励管理器"""

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        # 海报布局特定参数
        structure_weight=0.4,
        accuracy_weight=0.4,
        visual_weight=0.2,
        visual_quality_gpu_id=7,
        jsonl_file_path="/opt/liblibai-models/user-workspace/jiazhewei/typo_master/psd_dataset_169000_merged_with_caption.jsonl",
        **kwargs
    ) -> None:
        """
        初始化海报布局奖励管理器

        Args:
            tokenizer: 用于解码token ID的tokenizer
            num_examine: 打印到控制台的解码响应批次数量，用于调试
            compute_score: 计算奖励分数的函数，如果为None则使用default_compute_score
            reward_fn_key: 访问数据源的键，默认为"data_source"
            structure_weight: 结构匹配权重
            accuracy_weight: 文本准确度权重
            visual_weight: 视觉质量权重
            visual_quality_gpu_id: VisualQuality-R1模型使用的GPU ID
            jsonl_file_path: JSONL查找表文件路径
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # 海报布局特定配置
        self.reward_kwargs = {
            "structure_weight": structure_weight,
            "accuracy_weight": accuracy_weight,
            "visual_weight": visual_weight,
            "visual_quality_gpu_id": visual_quality_gpu_id,
            "jsonl_file_path": jsonl_file_path,
            **kwargs
        }
        
        print(f"PosterLayoutRewardManager initialized with:")
        print(f"  - Structure weight: {structure_weight}")
        print(f"  - Accuracy weight: {accuracy_weight}")
        print(f"  - Visual weight: {visual_weight}")
        print(f"  - VisualQuality-R1 GPU: {visual_quality_gpu_id}")
        print(f"  - JSONL lookup table: {jsonl_file_path}")

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """计算奖励分数"""

        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # 调用compute_score，传递海报布局特定的参数
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **self.reward_kwargs  # 传递海报布局特定配置
            )

            if isinstance(score, dict):
                reward = score["score"]
                # 存储详细信息
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # 调试输出
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[POSTER LAYOUT REWARD DEBUG]")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:300]}...")
                print(f"[ground_truth] {str(ground_truth)[:200]}...")
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}] {value}")
                else:
                    print(f"[score] {score}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
