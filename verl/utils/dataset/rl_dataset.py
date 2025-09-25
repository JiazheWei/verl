# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        import numpy as np
        
        messages: list = example.pop(self.prompt_key)
        
        # 添加numpy数组转换支持
        def convert_numpy_to_native(data):
            """递归转换numpy数组为Python原生类型"""
            if isinstance(data, np.ndarray):
                return [convert_numpy_to_native(item) for item in data.tolist()]
            elif isinstance(data, dict):
                return {k: convert_numpy_to_native(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_numpy_to_native(item) for item in data]
            else:
                return data
        
        # 转换numpy数组为原生Python格式
        messages = convert_numpy_to_native(messages)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                
                # 检查content是否已经是结构化格式
                if isinstance(content, list) and all(isinstance(item, dict) and "type" in item for item in content):
                    # content已经是结构化格式，只需要转换numpy并验证格式
                    converted_content = convert_numpy_to_native(content)
                    
                    # 过滤并重新排序content：确保text和image/video类型正确分离
                    filtered_content = []
                    for item in converted_content:
                        if isinstance(item, dict) and "type" in item:
                            if item["type"] == "image":
                                # 确保image项目有正确的结构，只包含type字段
                                filtered_content.append({"type": "image"})
                            elif item["type"] == "video":
                                # 确保video项目有正确的结构，只包含type字段  
                                filtered_content.append({"type": "video"})
                            elif item["type"] == "text" and item.get("text"):
                                # 保留text项目，只包含type和text字段，移除其他可能干扰的字段如"image": null
                                filtered_content.append({"type": "text", "text": item["text"]})
                    
                    # 如果没有image或video类型的内容，将整个content转换为纯文本
                    has_media = any(item.get("type") in ["image", "video"] for item in filtered_content)
                    if not has_media:
                        # 提取所有text内容合并
                        text_parts = [item.get("text", "") for item in filtered_content if item.get("type") == "text"]
                        combined_text = "".join(text_parts)
                        message["content"] = combined_text
                    else:
                        message["content"] = filtered_content
                        
                elif isinstance(content, str):
                    # content是字符串，需要拆分<image>标记
                    content_list = []
                    segments = re.split("(<image>|<video>)", content)
                    segments = [item for item in segments if item != ""]
                    for segment in segments:
                        if segment == "<image>":
                            content_list.append({"type": "image"})
                        elif segment == "<video>":
                            content_list.append({"type": "video"})
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list
                else:
                    # 处理其他格式（numpy数组等）
                    message["content"] = convert_numpy_to_native(content)

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                # 添加numpy数组转换支持
                import numpy as np
                if isinstance(row_dict_images, np.ndarray):
                    row_dict_images = row_dict_images.tolist()
                    
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                # 添加numpy数组转换支持
                if isinstance(row_dict_videos, np.ndarray):
                    row_dict_videos = row_dict_videos.tolist()
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            try:
                model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            except Exception as e:
                print(f"\n🚨 PROCESSOR ERROR at sample index {item} 🚨")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"\n=== SAMPLE DEBUG INFO ===")
                print(f"Dataset index: {item}")
                print(f"Images count: {len(images) if images else 0}")
                print(f"Videos count: {len(videos) if videos else 0}")
                print(f"Raw prompt length: {len(raw_prompt)}")
                print(f"Raw prompt preview: {raw_prompt[:300]}...")
                
                print(f"\n=== MESSAGES DEBUG ===")
                for i, msg in enumerate(messages):
                    print(f"Message {i} ({msg.get('role', 'unknown')}):")
                    content = msg.get('content', [])
                    if isinstance(content, list):
                        image_count = sum(1 for item in content if isinstance(item, dict) and item.get('type') == 'image')
                        text_count = sum(1 for item in content if isinstance(item, dict) and item.get('type') == 'text')
                        print(f"  Content items: {len(content)} (images: {image_count}, text: {text_count})")
                    else:
                        print(f"  Content type: {type(content)}")
                
                print(f"\n=== IMAGES DETAIL ===")
                if images:
                    for i, img in enumerate(images[:3]):  # 只显示前3个
                        print(f"Image {i}: type={type(img)}, size={img.size if hasattr(img, 'size') else 'N/A'}")
                
                print(f"\n=== RAW DATA DEBUG ===")
                original_row = self.dataframe[item]
                print(f"Original row keys: {list(original_row.keys())}")
                if 'extra_info' in original_row:
                    extra_info = original_row['extra_info']
                    if isinstance(extra_info, dict):
                        print(f"Sample ID: {extra_info.get('sample_id', 'unknown')}")
                        print(f"Canvas size: {extra_info.get('width', 'unknown')}x{extra_info.get('height', 'unknown')}")
                        print(f"Total layers: {extra_info.get('total_layers', 'unknown')}")
                
                # 保存完整的raw_prompt到错误文件
                try:
                    error_file_path = "/opt/liblibai-models/user-workspace/jiazhewei/error.txt"
                    with open(error_file_path, "w", encoding="utf-8") as f:
                        f.write("=== PROCESSOR ERROR SAMPLE ===\n")
                        f.write(f"Error type: {type(e).__name__}\n")
                        f.write(f"Error message: {str(e)}\n")
                        f.write(f"Dataset index: {item}\n")
                        f.write(f"Images count: {len(images) if images else 0}\n")
                        f.write(f"Videos count: {len(videos) if videos else 0}\n")
                        f.write(f"Raw prompt length: {len(raw_prompt)}\n")
                        f.write("=" * 50 + "\n")
                        f.write("COMPLETE RAW PROMPT:\n")
                        f.write("=" * 50 + "\n")
                        f.write(raw_prompt)
                        f.write("\n" + "=" * 50 + "\n")
                        f.write("MESSAGES STRUCTURE:\n")
                        f.write("=" * 50 + "\n")
                        import json
                        f.write(json.dumps(messages, indent=2, ensure_ascii=False))
                        f.write("\n" + "=" * 50 + "\n")
                    print(f"\n✅ 完整的raw_prompt已保存到: {error_file_path}")
                except Exception as save_error:
                    print(f"\n❌ 保存错误文件失败: {save_error}")
                
                print("=" * 60)
                raise e  # 重新抛出异常

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
