import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ================= 配置区域 =================
# 1. 模型权重路径 (请修改为您训练后的 best.pt)
WEIGHTS_PATH = '/home/liuyadong/ultralytics-main-cascade-2/runs/train/yolo12s-cascade-EALF3-自适应动态调节s/weights/best.pt' 

# 2. 测试图片目录
IMAGE_DIR = '/home/liuyadong/ultralytics-main-cascade-2/图片素材'  # 替换为您的图片文件夹路径

# 3. 输出结果目录
OUTPUT_DIR = 'runs/vis_coarse_check'
# ===========================================

class CoarseHeadVisualizer:
    def __init__(self, weights_path):
        print(f"Loading model from {weights_path}...")
        self.model = YOLO(weights_path)
        self.activations = {}
        self.layer_found = False
        
        # 注册 Hook
        self._register_hook()

    def _register_hook(self):
        """
        寻找 CoarseDetect 层并注册 Forward Hook
        """
        target_layer_name = 'CoarseDetect'
        
        # 遍历模型所有子模块寻找目标层
        for name, module in self.model.model.named_modules():
            # 判断类名是否包含 CoarseDetect (兼容不同命名习惯)
            if 'CoarseDetect' in module.__class__.__name__:
                print(f"✅ Found CoarseDetect layer: {name} ({module.__class__.__name__})")
                module.register_forward_hook(self._get_activation(name))
                self.layer_found = True
                break
        
        if not self.layer_found:
            print("❌ Warning: CoarseDetect layer not found! Please check model architecture.")

    def _get_activation(self, name):
        def hook(model, input, output):
            # output 可能是 list (多尺度) 或 tensor
            if isinstance(output, list):
                # 如果是多尺度列表，通常取第一个 (P3/P2 高分辨率层)
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def process_heatmap(self, feature_map, img_size):
        """
        将特征图转换为可视化的热力图
        feature_map: 2D Tensor [H, W]
        img_size: (w, h)
        """
        heatmap = feature_map.cpu().numpy()
        
        # 归一化到 0-255
        min_v, max_v = heatmap.min(), heatmap.max()
        if max_v - min_v > 1e-6:
            heatmap = (heatmap - min_v) / (max_v - min_v) * 255.0
        else:
            heatmap = np.zeros_like(heatmap)
            
        heatmap = heatmap.astype(np.uint8)
        
        # 放大到原图尺寸
        heatmap = cv2.resize(heatmap, img_size, interpolation=cv2.INTER_CUBIC)
        
        # 伪彩色处理 (JET 颜色映射: 蓝=低, 红=高)
        colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return colored_map

    def run(self, img_dir, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        images = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
        print(f"Found {len(images)} images in {img_dir}")

        for img_file in images:
            # 1. 推理 (imgsz可以根据训练设定调整，默认640)
            results = self.model.predict(str(img_file), save=False, conf=0.25, verbose=False)
            
            if not self.activations:
                print("No activations captured. Check hook registration.")
                continue

            # 获取原始图像
            orig_img = results[0].orig_img
            h, w = orig_img.shape[:2]
            
            # 绘制检测框 (作为参考)
            annotated_img = results[0].plot()

            # 2. 获取 CoarseDetect 输出
            # key 是我们注册时用的 name
            layer_name = list(self.activations.keys())[0]
            coarse_out = self.activations[layer_name] # [B, 2, H_feat, W_feat]
            
            # 取第一张图 (Batch size 1)
            # Channel 0: Saliency, Channel 1: Uncertainty
            raw_saliency = coarse_out[0, 0, :, :]
            raw_uncertainty = coarse_out[0, 1, :, :]

            # === 处理 Saliency (概率) ===
            # 经过 Sigmoid 才是概率，这也是 Loss 监督的对象
            prob_map = torch.sigmoid(raw_saliency)
            vis_saliency = self.process_heatmap(prob_map, (w, h))

            # === 处理 Uncertainty (不确定性) ===
            # Uncertainty 不需要 Sigmoid，直接看相对高低
            vis_uncertainty = self.process_heatmap(raw_uncertainty, (w, h))

            # 3. 图像融合 (Overlay)
            alpha = 0.5
            overlay_saliency = cv2.addWeighted(orig_img, 1-alpha, vis_saliency, alpha, 0)
            overlay_uncertainty = cv2.addWeighted(orig_img, 1-alpha, vis_uncertainty, alpha, 0)

            # 4. 拼接显示 [原图+框 | Saliency | Uncertainty]
            # 添加文字说明
            cv2.putText(overlay_saliency, "Saliency (Prob)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(overlay_uncertainty, "Uncertainty", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            combined = np.hstack([annotated_img, overlay_saliency, overlay_uncertainty])
            
            out_file = save_path / f"{img_file.name}_vis.jpg"
            cv2.imwrite(str(out_file), combined)
            print(f"Saved visualization to {out_file}")

            # 清理 Hook 缓存，防止显存泄露
            self.activations = {}

if __name__ == "__main__":
    viz = CoarseHeadVisualizer(WEIGHTS_PATH)
    viz.run(IMAGE_DIR, OUTPUT_DIR)