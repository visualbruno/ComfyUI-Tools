from huggingface_hub import hf_hub_download

def download(ckpt_name: str) -> str:
    MAP = {
        'experiments/skeleton/articulation-xl_quantization_256/model.ckpt': 'skeleton/articulation-xl_quantization_256/model.ckpt',
        'experiments/skin/articulation-xl/model.ckpt': 'skin/articulation-xl/model.ckpt',
        'experiments/skin/skeleton/model.ckpt': 'skin/skeleton/model.ckpt',
    }
    
    try:
        if ckpt_name not in MAP:
            print(f"not found: {ckpt_name}")
            return ckpt_name
        return hf_hub_download(
            repo_id='VAST-AI/UniRig',
            filename=MAP[ckpt_name],
        )
    except Exception as e:
        print(f"Failed to download {ckpt_name}: {e}")
        return ckpt_name