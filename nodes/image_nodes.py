import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as F

file_directory = os.path.dirname(os.path.abspath(__file__))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
def tensor2pil(image: torch.Tensor) -> Image.Image:
    """
    Accepts either:
      - (H,W,C)
      - (1,H,W,C)
    Returns a PIL RGB/RGBA image depending on channels.
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            # Expect (B,H,W,C); allow only B==1 here
            if t.shape[0] != 1:
                raise ValueError(f"tensor2pil expects batch of 1, got batch={t.shape[0]}")
            t = t[0]
        elif t.ndim != 3:
            raise ValueError(f"tensor2pil expects (H,W,C) or (1,H,W,C), got shape={tuple(t.shape)}")

        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    raise TypeError(f"tensor2pil expected torch.Tensor, got {type(image)}")  
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array

class VisualBrunoToolsCropImageAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT",{"default":0,"min":0,"max":1024}),
                "remove_background": ("BOOLEAN",{"default":False}),
                "max_size": ("INT",{"default":2048,"min":512,"max":8192,"step":128}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/Image"

    def process(self, image, padding, remove_background, max_size):
        if image.ndim == 3:
            image = tensor2pil(image)
            
            if remove_background:
                from rembg import remove
                image = remove(image)
            
            image = self.preprocess_image(image, max_size)
            
            if padding>0:
                border = (int(padding), int(padding), int(padding), int(padding))
                fill_color = self.parse_fill_for_image("0,0,0,255", image)
                image = ImageOps.expand(image,border=border,fill=fill_color)
            
            image = pil2tensor(image)
        elif image.ndim == 4:
            images = convert_tensor_images_to_pil(image)
            tensor_list = []
            for img in images:
                if remove_background:
                    from rembg import remove
                    img = remove(img)
                
                img = self.preprocess_image(img, max_size)
                
                if padding>0:
                    border = (int(padding), int(padding), int(padding), int(padding))
                    fill_color = self.parse_fill_for_image("0,0,0,255", img)
                    img = ImageOps.expand(img,border=border,fill=fill_color)
                
                tensor_list.append(pil2tensor(img))
                
                max_h = max(t.shape[-3] for t in tensor_list)
                max_w = max(t.shape[-2] for t in tensor_list)

                resized_tensors = []

                for t in tensor_list:
                    # Ensure tensor is [C, H, W] for PyTorch's interpolate function
                    # Current shape is likely [H, W, C] or [1, H, W, C]
                    temp_t = t.squeeze() # Get to [H, W, C]
                    temp_t = temp_t.permute(2, 0, 1).unsqueeze(0) # Becomes [1, C, H, W]
                    
                    # 2. Resize to the max dimensions
                    # Using 'bicubic' or 'bilinear' for better quality than 'nearest'
                    temp_t = F.interpolate(temp_t, size=(max_h, max_w), mode='bicubic', align_corners=False)
                    
                    # 3. Convert back to ComfyUI format [H, W, C]
                    temp_t = temp_t.squeeze(0).permute(1, 2, 0)
                    resized_tensors.append(temp_t)                
                
            image = torch.stack(resized_tensors)
        
        return (image,)    

    def parse_fill_for_image(self, fill: str, img):
        values = [int(x.strip()) for x in fill.split(",")]

        if img.mode in ("L", "P"):
            return values[0]

        if img.mode == "RGB":
            return tuple(values[:3])

        if img.mode == "RGBA":
            return tuple(values[:4])

        raise ValueError(f"Unsupported image mode: {img.mode}")         


    def preprocess_image(self, input: Image.Image, max_res) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, max_res / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        # if has_alpha:
            # output = input
        # else:
            # input = input.convert('RGB')
            # if self.low_vram:
                # self.rembg_model.to(self.device)
            # output = self.rembg_model(input)
            # if self.low_vram:
                # self.rembg_model.cpu()
        output = input
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output    