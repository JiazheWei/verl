#!/usr/bin/env python3
"""
VisualQuality-R1 Reward Model 独立服务
使用FastAPI部署在GPU 7上，为GRPO训练提供在线reward计算
"""

import os
import json
import tempfile
from typing import Dict, Any, Optional
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import deepdiff
from difflib import SequenceMatcher
import re

# 全局变量
MODEL = None
PROCESSOR = None
LOOKUP_TABLE = None

class RewardRequest(BaseModel):
    """Reward计算请求"""
    solution_str: str
    ground_truth: str  # 修改为字符串类型，接收JSON字符串
    extra_info: dict = {}
    structure_weight: float = 0.4
    accuracy_weight: float = 0.4
    visual_weight: float = 0.2
    jsonl_file_path: str = "/opt/liblibai-models/user-workspace/jiazhewei/typo_master/psd_dataset_169000_merged_with_caption.jsonl"

class RewardResponse(BaseModel):
    """Reward计算响应"""
    score: float  # 只返回最终分数，符合VERL框架期望
    error: Optional[str] = None

def load_model(gpu_id: int = 7):
    """加载VisualQuality-R1模型到指定GPU"""
    global MODEL, PROCESSOR
    
    if MODEL is None:
        print(f"🚀 Loading VisualQuality-R1 model on GPU {gpu_id}...")
        
        MODEL_PATH = "/opt/liblibai-models/model-weights/VisualQuality-R1-7B"
        
        # 强制设置GPU环境
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available for GPU {gpu_id}")
        
        device = torch.device("cuda:0")  # 由于CUDA_VISIBLE_DEVICES=7，cuda:0就是物理GPU 7
        
        MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        
        PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
        PROCESSOR.tokenizer.padding_side = "left"
        
        print(f"✅ VisualQuality-R1 model loaded successfully on {device}")
        print(f"📊 Model device: {next(MODEL.parameters()).device}")
        print(f"🔥 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

def load_lookup_table(jsonl_file_path: str):
    """加载JSONL查找表"""
    global LOOKUP_TABLE
    
    if LOOKUP_TABLE is None:
        print(f"📋 Loading JSONL lookup table from {jsonl_file_path}")
        LOOKUP_TABLE = {}
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    sample_id = data.get('sample_id')
                    if sample_id:
                        LOOKUP_TABLE[sample_id] = data
        
        print(f"✅ Loaded {len(LOOKUP_TABLE)} samples in lookup table")

def extract_json_from_response(response_text: str) -> str:
    """从模型响应中提取JSON字符串"""
    try:
        json_match = re.search(r'^(\{.*?\})\s*(?:\{|$)', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text.strip()
            if json_str.count('{"layers"') > 1:
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

def inject_image_paths(parsed_json: Dict, sample_id: str) -> Dict:
    """向解析的JSON中注入真实的image_id路径映射"""
    try:
        sample_data = LOOKUP_TABLE.get(sample_id)
        if not sample_data:
            print(f"Warning: sample_id {sample_id} not found in lookup table")
            return parsed_json
        
        image_id_data = sample_data.get('image_id', [])
        json_path = sample_data.get('json', '')
        
        if json_path:
            base_dir = os.path.dirname(os.path.dirname(json_path))
            for item in image_id_data:
                if 'image' in item:
                    item['image'] = os.path.join(base_dir, 'merged', item['image'])
        
        parsed_json['image_id'] = image_id_data
        return parsed_json
        
    except Exception as e:
        print(f"Error injecting image paths: {e}")
        return parsed_json

def render_json_to_image(json_data: Dict) -> Optional[Image.Image]:
    """将JSON渲染为PIL图片（简化版）"""
    try:
        canvas_size = json_data.get('canvas_size', {'width': 1200, 'height': 800})
        width = canvas_size['width']
        height = canvas_size['height']
        
        canvas = Image.new('RGB', (width, height), (255, 255, 255))
        
        layers = json_data.get('layers', [])
        image_id_mapping = {item['id']: item['image'] for item in json_data.get('image_id', [])}
        
        for layer in layers:
            try:
                image_id = layer.get('image_id', 0)
                image_path = image_id_mapping.get(image_id)
                
                if image_path and os.path.exists(image_path):
                    layer_img = Image.open(image_path)
                    if layer_img.mode != 'RGBA':
                        layer_img = layer_img.convert('RGBA')
                    
                    x = layer.get('x', 0)
                    y = layer.get('y', 0)
                    canvas.paste(layer_img, (x, y), layer_img)
                    
            except Exception as e:
                print(f"Error rendering layer {layer.get('image_id', 'unknown')}: {e}")
                continue
        
        return canvas
        
    except Exception as e:
        print(f"Error rendering JSON to image: {e}")
        return None

def score_visual_quality(image: Image.Image) -> float:
    """使用VisualQuality-R1模型对图片进行质量评分"""
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
        
        text = [PROCESSOR.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = PROCESSOR(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(MODEL.device)
        
        generated_ids = MODEL.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=True, top_k=50, top_p=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        try:
            model_output_matches = re.findall(r'<answer>(.*?)</answer>', batch_output_text[0], re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else batch_output_text[0].strip()
            score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
            return max(1.0, min(5.0, score))
        except:
            print(f"Failed to parse visual quality score from: {batch_output_text[0]}")
            return 2.5
            
    except Exception as e:
        print(f"Error scoring visual quality: {e}")
        return 1.0

def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    try:
        return SequenceMatcher(None, text1, text2).ratio()
    except Exception as e:
        print(f"Error calculating text similarity: {e}")
        return 0.0

def calculate_structure_similarity(json1: Dict, json2: Dict) -> float:
    """计算两个JSON结构的相似度"""
    try:
        diff = deepdiff.DeepDiff(json1, json2, ignore_order=True)
        
        if not diff:
            return 1.0
        
        total_changes = 0
        for key in ['values_changed', 'dictionary_item_added', 'dictionary_item_removed', 'iterable_item_added', 'iterable_item_removed']:
            if key in diff:
                total_changes += len(diff[key])
        
        max_items = max(len(json1.get('layers', [])), len(json2.get('layers', [])))
        if max_items == 0:
            return 0.0
        
        similarity = max(0.0, 1.0 - (total_changes / (max_items * 5)))
        return similarity
        
    except Exception as e:
        print(f"Error calculating structure similarity: {e}")
        return 0.0

# FastAPI应用
app = FastAPI(title="VisualQuality-R1 Reward Model Server", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "model_loaded": MODEL is not None,
        "gpu_memory": f"{torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
    }

@app.post("/compute_reward", response_model=RewardResponse)
async def compute_reward(request: RewardRequest):
    """计算reward分数"""
    try:
        # 加载查找表
        load_lookup_table(request.jsonl_file_path)
        
        sample_id = request.extra_info.get('sample_id', '')
        
        # 1. 解析ground_truth JSON字符串
        try:
            ground_truth_dict = json.loads(request.ground_truth)
        except json.JSONDecodeError as e:
            print(f"[REWARD ERROR] Ground truth JSON parsing failed: {e}")
            return RewardResponse(score=1.0, error=f"Ground truth JSON parsing failed: {e}")
        
        # 2. 提取和解析solution JSON
        json_str = extract_json_from_response(request.solution_str)
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[REWARD ERROR] Solution JSON parsing failed: {e}")
            return RewardResponse(score=1.0, error=f"Solution JSON parsing failed: {e}")
        
        # 3. 注入image_id路径映射
        parsed_json = inject_image_paths(parsed_json, sample_id)
        
        # 4. 渲染为图片
        rendered_image = render_json_to_image(parsed_json)
        
        if rendered_image is None:
            print(f"[REWARD ERROR] Image rendering failed for sample_id: {sample_id}")
            return RewardResponse(score=1.0, error="Image rendering failed")
        
        # 5. 计算视觉质量分数
        visual_score = score_visual_quality(rendered_image)
        
        # 6. 计算文本准确度分数
        accuracy_score = calculate_text_similarity(json_str, request.ground_truth)
        
        # 7. 计算结构相似度分数
        structure_score = calculate_structure_similarity(parsed_json, ground_truth_dict)
        
        # 8. 加权计算最终分数
        final_score = (
            request.visual_weight * visual_score +
            request.accuracy_weight * accuracy_score * 5 +
            request.structure_weight * structure_score * 5
        )
        
        # 输出详细信息到日志（便于调试）
        print(f"[REWARD DEBUG] sample_id={sample_id}")
        print(f"  visual_score={visual_score:.3f} (weight={request.visual_weight})")
        print(f"  accuracy_score={accuracy_score:.3f} (weight={request.accuracy_weight})")  
        print(f"  structure_score={structure_score:.3f} (weight={request.structure_weight})")
        print(f"  final_score={final_score:.3f}")
        
        # 只返回最终分数，符合VERL框架期望
        return RewardResponse(score=final_score)
        
    except Exception as e:
        print(f"[REWARD ERROR] Unexpected error in compute_reward: {e}")
        return RewardResponse(score=1.0, error=str(e))

if __name__ == "__main__":
    print("🚀 Starting VisualQuality-R1 Reward Model Server...")
    print(f"📍 Binding to 0.0.0.0:8899")
    print(f"🔧 Service endpoints:")
    print(f"   - Health Check: http://localhost:8899/health")
    print(f"   - Compute Reward: http://localhost:8899/compute_reward")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8899,
        log_level="info"
    )
