import torch
import time
import numpy as np
from PIL import Image

from src.evaluation.libero_bench.VLANeXt_utils import get_vla, get_processor

class DictConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            else:
                setattr(self, k, v)

def get_model_info(model):
    param_size = 0
    param_count = 0
    seen = set()
    
    for param in model.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            param_count += param.nelement()
            param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        if id(buffer) not in seen:
            seen.add(id(buffer))
            buffer_size += buffer.nelement() * buffer.element_size()
            
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return param_count, size_mb

def measure_inference_speed(model, processor, device, dtype, batch_size=1, input_modality="image", num_warmup=5, num_runs=50):
    print(f"Preparing dummy inputs for modality: {input_modality} (Batch Size: {batch_size})...")
    
    text = "Detailed instruction for the task."
    
    is_paligemma = "PaliGemma" in processor.__class__.__name__
    is_llama = "Llama" in processor.__class__.__name__
    is_qwen = "Qwen" in processor.__class__.__name__

    inputs = {}
    
    if input_modality == "image":
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        if is_paligemma:
            text_input = ["<image>" + text] * batch_size
            images = [img] * batch_size
            inputs = processor(text=text_input, images=images, return_tensors="pt", padding=True)
        elif is_llama:
            text_inputs = [text] * batch_size
            inputs = processor.tokenizer(text_inputs, padding=True, return_tensors="pt")
            image_inputs = processor.image_processor([img] * batch_size, return_tensors="pt")
            inputs["pixel_values"] = image_inputs["pixel_values"]
        elif is_qwen:
            messages = []
            for _ in range(batch_size):
                messages.append([
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}
                ])
            text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_inputs, images=[img]*batch_size, return_tensors="pt", padding=True)
        else:
            raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")
            
    elif input_modality == "video":
        if is_paligemma or is_llama:
             raise NotImplementedError("PaliGemma/Llama implementation in VLANeXt currently supports 'image' modality only.")
        
        num_frames = 8
        frames = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(num_frames)]
        
        messages = []
        all_videos = []
        for _ in range(batch_size):
            messages.append([
                {"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": text}]}
            ])
            all_videos.append(frames)
        
        video_metadata = [
            {"total_num_frames": num_frames, "fps": 20.0, "frames_indices": list(range(num_frames))}
            for _ in all_videos
        ]
        text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text_inputs, videos=all_videos, videos_kwargs={"fps": 20.0, "return_metadata": True, "video_metadata": video_metadata}, return_tensors="pt", padding=True)

    else:
        raise ValueError(f"Unknown input_modality: {input_modality}")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)

    history_len = getattr(model, "num_history", 1) 
    if history_len == 0: history_len = 1
    action_dim = getattr(model, "action_dim", 7)
    
    proprioception = None
    if getattr(model, "use_proprio_input_vlm", False):
        proprioception = torch.randn(batch_size, history_len, action_dim, device=device, dtype=dtype)
    
    history_actions = None
    if getattr(model, "use_action_input_policy", False):
        history_actions = torch.randn(batch_size, history_len, action_dim, device=device, dtype=dtype)

    print(f"Inputs prepared. Starting warmup ({num_warmup} runs)...")
    model.eval()
    
    valid_args = ["input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]
    fwd_inputs = {k: v for k, v in inputs.items() if k in valid_args}

    with torch.no_grad():
        for _ in range(num_warmup):
            model.predict_action(
                proprioception=proprioception,
                history_actions=history_actions,
                **fwd_inputs
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    print(f"Starting timing ({num_runs} runs)...")
    timings = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                model.predict_action(
                    proprioception=proprioception,
                    history_actions=history_actions,
                    **fwd_inputs
                )
                
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) # ms
            else:
                start_t = time.perf_counter()
                model.predict_action(
                    proprioception=proprioception,
                    history_actions=history_actions,
                    **fwd_inputs
                )
                end_t = time.perf_counter()
                elapsed = (end_t - start_t) * 1000.0 # ms

            timings.append(elapsed)

    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms * batch_size
    
    print(f"=== Results ({input_modality}) ===")
    print(f"  Average Latency: {avg_ms:.2f} ms +/- {std_ms:.2f} ms")
    print(f"  Throughput: {fps:.2f} FPS (BS={batch_size})")
    return avg_ms, fps

if __name__ == "__main__":
    CHECKPOINT_PATH = "/mnt/draven/checkpoints/VLANeXt/VLANeXt_final_libero/bs256_steps10000_spatial_qwen3vl2b_lr1e-4_dct0.1_v4/checkpoint_final.pt"
    INPUT_MODALITY = "image" # "image" or "video"
    BATCH_SIZE = 1

    cfg_dict = {
        "eval": {
            "finetuned_checkpoint": CHECKPOINT_PATH
        },
        "model": {}
    }
    cfg = DictConfig(cfg_dict)

    print(f"Initializing Model from Checkpoint: {CHECKPOINT_PATH}...")
    model = get_vla(cfg)
    
    print("\nInitializing Processor...")
    processor = get_processor(cfg)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    print(f"\nLocked Configuration:\n  Device: {device}\n  Dtype: {dtype}\n  Modality: {INPUT_MODALITY}\n  Batch Size: {BATCH_SIZE}")

    print("\n[Step 1] Model Size Evaluation")
    count, size_mb = get_model_info(model)
    print(f"  Total Parameters: {count / 1e6:.2f} M")
    print(f"  Total Memory Footprint: {size_mb:.2f} MB")
    
    print("\n[Step 2] Inference Speed Evaluation")
    
    measure_inference_speed(
        model, processor, device, dtype, 
        batch_size=BATCH_SIZE, 
        input_modality=INPUT_MODALITY, 
        num_warmup=5, 
        num_runs=50
    )
