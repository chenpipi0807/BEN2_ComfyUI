import sys
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
import BEN2

class BackgroundEraseNetwork:
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "BEN2"

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BEN2.BEN_Base().to(device).eval()
        self.model.loadcheckpoints(os.path.join(script_directory, "BEN2_Base.pth"))
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }

    def process_image(self, input_image):
        # 处理输入图像
        if isinstance(input_image, torch.Tensor):
            if input_image.dim() == 4:
                input_image = input_image[0]
            
            if input_image.dim() == 3:
                input_image = input_image.permute(2, 0, 1)

            input_image = self.to_pil(input_image)

        # 转换为RGBA格式
        if input_image.mode != 'RGBA':
            input_image = input_image.convert("RGBA")

        # 执行推理
        foreground = self.model.inference(input_image)

        # 提取alpha通道作为mask
        alpha = foreground.split()[-1]
        mask_np = np.array(alpha)
        mask_tensor = torch.from_numpy(mask_np).float() / 255.0  # 归一化到[0,1]
        mask_tensor = mask_tensor.unsqueeze(0)  # [B, H, W]

        # 转换前景图像为tensor
        foreground_tensor = self.to_tensor(foreground)
        foreground_tensor = foreground_tensor.permute(1, 2, 0).unsqueeze(0)  # [B, H, W, C]

        return (foreground_tensor, mask_tensor)

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "BackgroundEraseNetwork": BackgroundEraseNetwork
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundEraseNetwork": "Background Erase Network (Image+Mask)"
}
