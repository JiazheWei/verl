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
Visual Quality Poster Reward Function for GRPO Training
海报生成任务的奖励函数，结合视觉质量、文本准确度和结构匹配
"""

import json
import os
import re
import ast
import tempfile
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher
import difflib

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import deepdiff

# 全局变量存储加载的模型和数据
_VISUAL_QUALITY_MODEL = None
_VISUAL_QUALITY_PROCESSOR = None
_JSONL_LOOKUP_TABLE = None


def _load_visual_quality_model(gpu_id: int = 7):
    """加载VisualQuality-R1模型"""
    global _VISUAL_QUALITY_MODEL, _VISUAL_QUALITY_PROCESSOR
    
    if _VISUAL_QUALITY_MODEL is None:
        MODEL_PATH = "/opt/liblibai-models/model-weights/VisualQuality-R1-7B"
        
        # 详细的CUDA环境调试
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
            print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
            import os
            print(f"[DEBUG] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # 尝试使用CUDA，如果不可用则使用CPU（分布式环境可能限制CUDA访问）
        if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
            device_str = f"cuda:{gpu_id}"
            device_map = {"": device_str}
            torch_dtype = torch.bfloat16
            print(f"Loading VisualQuality-R1 model on {device_str}")
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # 如果指定的GPU不可用，使用第一个可用GPU
            device_str = "cuda:0"
            device_map = {"": device_str}
            torch_dtype = torch.bfloat16
            print(f"GPU {gpu_id} not available, using {device_str} instead")
        else:
            # 分布式环境中如果CUDA不可用，回退到CPU（但会很慢）
            device_str = "cpu"
            device_map = None
            torch_dtype = torch.float32  # CPU上使用float32
            print(f"WARNING: CUDA not available in Ray worker, falling back to CPU for VisualQuality-R1")
            print("This will be much slower. Consider fixing CUDA access in distributed environment.")
        
        try:
            _VISUAL_QUALITY_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            
            # 如果没有使用device_map，手动移动到设备
            if device_map is None:
                _VISUAL_QUALITY_MODEL = _VISUAL_QUALITY_MODEL.to(device_str)
            
            _VISUAL_QUALITY_PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
            _VISUAL_QUALITY_PROCESSOR.tokenizer.padding_side = "left"
            print(f"✅ VisualQuality-R1 model loaded successfully on {device_str}")
            
        except Exception as e:
            print(f"❌ Failed to load VisualQuality-R1 model: {e}")
            # 最后的回退：CPU加载
            print("Attempting CPU fallback...")
            _VISUAL_QUALITY_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
            )
            _VISUAL_QUALITY_MODEL = _VISUAL_QUALITY_MODEL.to("cpu")
            _VISUAL_QUALITY_PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
            _VISUAL_QUALITY_PROCESSOR.tokenizer.padding_side = "left"
            print("✅ VisualQuality-R1 model loaded on CPU as final fallback")
    
    return _VISUAL_QUALITY_MODEL, _VISUAL_QUALITY_PROCESSOR


def _load_jsonl_lookup_table(jsonl_file_path: str):
    """加载JSONL文件作为查找表，建立sample_id到image_id映射的索引"""
    global _JSONL_LOOKUP_TABLE
    
    if _JSONL_LOOKUP_TABLE is None:
        print(f"Loading JSONL lookup table from {jsonl_file_path}")
        _JSONL_LOOKUP_TABLE = {}
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    sample_id = data.get('sample_id')
                    if sample_id:
                        _JSONL_LOOKUP_TABLE[sample_id] = data
        
        print(f"Loaded {len(_JSONL_LOOKUP_TABLE)} samples in lookup table")
    
    return _JSONL_LOOKUP_TABLE


def _extract_json_from_response(response_text: str) -> str:
    """
    从模型响应中提取JSON字符串
    参考single_sample.py的逻辑
    """
    try:
        # 查找第一个完整的JSON对象
        json_match = re.search(r'^(\{.*?\})\s*(?:\{|$)', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有找到，尝试整个文本
            json_str = response_text.strip()
            # 如果有重复的JSON，只取第一个完整的
            if json_str.count('{"layers"') > 1:
                # 找到第一个完整JSON的结束位置
                depth = 0
                start_idx = json_str.find('{')
                for idx, char in enumerate(json_str[start_idx:], start_idx):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = json_str[:idx+1]
                            break
        
        return json_str
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return response_text


def _inject_image_paths(parsed_json: Dict, sample_id: str, lookup_table: Dict) -> Dict:
    """
    向解析的JSON中注入真实的image_id路径映射
    参考single_sample.py第258-280行的逻辑
    """
    try:
        # 从lookup table获取样本信息
        sample_data = lookup_table.get(sample_id)
        if not sample_data:
            print(f"Warning: sample_id {sample_id} not found in lookup table")
            return parsed_json
        
        # 获取image_id映射
        image_id_data = sample_data.get('image_id', [])
        
        # 构建绝对路径
        json_path = sample_data.get('json', '')
        if json_path:
            # 从路径中提取基础目录
            base_dir = os.path.dirname(os.path.dirname(json_path))  # 上两级目录
            
            # 转换image_id中的相对路径为绝对路径
            for item in image_id_data:
                if 'image' in item:
                    item['image'] = os.path.join(base_dir, 'merged', item['image'])
        
        # 将image_id添加到解析的JSON中
        parsed_json['image_id'] = image_id_data
        return parsed_json
        
    except Exception as e:
        print(f"Error injecting image paths: {e}")
        return parsed_json


def _render_json_to_image(json_data: Dict, temp_dir: str) -> Optional[Image.Image]:
    """
    将JSON渲染为PIL图片
    基于reconstruct_with_psd_tools.py的逻辑
    """
    try:
        # 这里实现简化版的渲染逻辑
        # 为了快速实现，先创建一个占位符图片
        # TODO: 集成完整的渲染逻辑
        
        canvas_size = json_data.get('canvas_size', {'width': 1200, 'height': 800})
        width = canvas_size['width']
        height = canvas_size['height']
        
        # 创建基础画布
        canvas = Image.new('RGB', (width, height), (255, 255, 255))
        
        # 简化渲染：按layers中的位置放置图片
        layers = json_data.get('layers', [])
        image_id_mapping = {item['id']: item['image'] for item in json_data.get('image_id', [])}
        
        for layer in layers:
            try:
                image_id = layer.get('image_id', 0)
                image_path = image_id_mapping.get(image_id)
                
                if image_path and os.path.exists(image_path):
                    # 加载图层图片
                    layer_img = Image.open(image_path)
                    if layer_img.mode != 'RGBA':
                        layer_img = layer_img.convert('RGBA')
                    
                    # 获取位置信息
                    x = layer.get('x', 0)
                    y = layer.get('y', 0)
                    
                    # 简单粘贴到画布上
                    canvas.paste(layer_img, (x, y), layer_img)
                    
            except Exception as e:
                print(f"Error rendering layer {layer.get('image_id', 'unknown')}: {e}")
                continue
        
        return canvas
        
    except Exception as e:
        print(f"Error rendering JSON to image: {e}")
        return None


def _score_visual_quality(image: Image.Image, model, processor) -> float:
    """
    使用VisualQuality-R1模型对图片进行质量评分
    参考visual_quality_inference.py
    """
    try:
        PROMPT = (
            "You are doing the image quality assessment task. Here is the question: "
            "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
            "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
        )
        
        QUESTION_TEMPLATE = "{Question} Please only output the final answer with only one score in <answer> </answer> tags."
        message = [
            {
                "role": "user",
                "content": [
                    {'type': 'image', 'image': image},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=PROMPT)}
                ],
            }
        ]
        
        batch_messages = [message]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=True, top_k=50, top_p=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        try:
            model_output_matches = re.findall(r'<answer>(.*?)</answer>', batch_output_text[0], re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else batch_output_text[0].strip()
            score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
            return max(1.0, min(5.0, score))  # 确保分数在1-5范围内
        except:
            print(f"Failed to parse visual quality score from: {batch_output_text[0]}")
            return 2.5  # 默认中等分数
            
    except Exception as e:
        print(f"Error scoring visual quality: {e}")
        return 1.0  # 最低分数


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    try:
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity
    except Exception as e:
        print(f"Error calculating text similarity: {e}")
        return 0.0


def _calculate_structure_similarity(json1: Dict, json2: Dict) -> float:
    """
    计算两个JSON结构的相似度
    使用deepdiff进行结构比较
    """
    try:
        # 使用deepdiff计算差异
        diff = deepdiff.DeepDiff(json1, json2, ignore_order=True)
        
        # 计算相似度分数
        if not diff:
            return 1.0  # 完全相同
        
        # 根据差异类型计算相似度
        total_changes = 0
        if 'values_changed' in diff:
            total_changes += len(diff['values_changed'])
        if 'dictionary_item_added' in diff:
            total_changes += len(diff['dictionary_item_added'])
        if 'dictionary_item_removed' in diff:
            total_changes += len(diff['dictionary_item_removed'])
        if 'iterable_item_added' in diff:
            total_changes += len(diff['iterable_item_added'])
        if 'iterable_item_removed' in diff:
            total_changes += len(diff['iterable_item_removed'])
        
        # 简单的相似度计算：变化越少，相似度越高
        max_items = max(len(json1.get('layers', [])), len(json2.get('layers', [])))
        if max_items == 0:
            return 0.0
        
        similarity = max(0.0, 1.0 - (total_changes / (max_items * 5)))  # 假设每个layer最多5个关键字段
        return similarity
        
    except Exception as e:
        print(f"Error calculating structure similarity: {e}")
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Dict,
    extra_info: Dict,
    structure_weight: float = 0.4,
    accuracy_weight: float = 0.4,
    visual_weight: float = 0.2,
    visual_quality_gpu_id: int = 7,
    jsonl_file_path: str = "/opt/liblibai-models/user-workspace/jiazhewei/typo_master/psd_dataset_169000_merged_with_caption.jsonl"
) -> Dict[str, Any]:
    """
    海报布局生成任务的奖励函数
    
    Args:
        data_source: 数据源标识（应为"poster_layout_synthetic"）
        solution_str: 模型生成的JSON字符串
        ground_truth: 真实答案数据
        extra_info: 额外信息，包含sample_id等
        structure_weight: 结构匹配权重
        accuracy_weight: 文本准确度权重  
        visual_weight: 视觉质量权重
        visual_quality_gpu_id: VisualQuality-R1模型使用的GPU ID
        jsonl_file_path: JSONL查找表文件路径
        
    Returns:
        包含分数和详细信息的字典
    """
    try:
        # 加载必要的资源
        visual_model, visual_processor = _load_visual_quality_model(visual_quality_gpu_id)
        lookup_table = _load_jsonl_lookup_table(jsonl_file_path)
        
        sample_id = extra_info.get('sample_id', '')
        
        # 1. 提取和解析JSON
        json_str = _extract_json_from_response(solution_str)
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return {
                "score": 1.0,  # 最低正分
                "visual_score": 1.0,
                "accuracy_score": 0.0,
                "structure_score": 0.0,
                "error": "JSON parsing failed"
            }
        
        # 2. 注入image_id路径映射
        parsed_json = _inject_image_paths(parsed_json, sample_id, lookup_table)
        
        # 3. 渲染为图片
        with tempfile.TemporaryDirectory() as temp_dir:
            rendered_image = _render_json_to_image(parsed_json, temp_dir)
            
            if rendered_image is None:
                print("Image rendering failed")
                return {
                    "score": 1.0,  # 最低正分
                    "visual_score": 1.0,
                    "accuracy_score": 0.0,
                    "structure_score": 0.0,
                    "error": "Image rendering failed"
                }
            
            # 4. 计算视觉质量分数
            visual_score = _score_visual_quality(rendered_image, visual_model, visual_processor)
        
        # 5. 计算文本准确度分数
        ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)
        accuracy_score = _calculate_text_similarity(json_str, ground_truth_str)
        
        # 6. 计算结构相似度分数
        structure_score = _calculate_structure_similarity(parsed_json, ground_truth)
        
        # 7. 加权计算最终分数
        final_score = (
            visual_weight * visual_score +
            accuracy_weight * accuracy_score * 5 +  # 将0-1范围转换为1-5范围
            structure_weight * structure_score * 5
        )
        
        return {
            "score": final_score,
            "visual_score": visual_score,
            "accuracy_score": accuracy_score,
            "structure_score": structure_score,
            "weights": {
                "visual_weight": visual_weight,
                "accuracy_weight": accuracy_weight,
                "structure_weight": structure_weight
            }
        }
        
    except Exception as e:
        print(f"Error in compute_score: {e}")
        return {
            "score": 1.0,  # 最低正分
            "visual_score": 1.0,
            "accuracy_score": 0.0,
            "structure_score": 0.0,
            "error": str(e)
        }
