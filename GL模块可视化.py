import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import torch.nn.functional as F
from pathlib import Path

# ================= 配置区域 =================
# 1. 您的模型路径 (可以是 best.pt 或 last.pt)
model_path = '/home/liuyadong/ultralytics-main-cascade-2/runs/train/yolo12s-cascade-EALF3-优化GL模块/weights/best.pt'

# 2. 测试图片文件夹路径
image_dir = '/home/liuyadong/ultralytics-main-cascade-2/图片素材'

# 3. 结果保存路径
save_dir = 'runs/vis_gl_module'

# 4. 要处理的图片数量 (避免跑太久)
num_images = 5
# ===========================================

# 容器，用于存储 Hook 抓取的数据
feature_maps = {}

def hook_fn(module, input, output):
    """
    Hook 函数：在前向传播时自动抓取输入和输出
    input[0]: x_local (细视特征)
    input[1]: x_global (粗视特征)
    output:   fused_feature (融合后特征)
    """
    # input 是一个 tuple，对应 forward 的参数
    # 根据您的定义 forward(self, x_local, x_global, *args)
    x_local = input[0]
    x_global = input[1]
    
    # 将 tensor 转为 numpy，取 batch 中的第一张图 (index 0)
    feature_maps['local'] = x_local[0].detach().cpu()
    feature_maps['global'] = x_global[0].detach().cpu()
    feature_maps['output'] = output[0].detach().cpu()

def process_feature_map(f_map, target_size=None):
    """
    将特征图转换为可视化的热力图
    1. 对通道维度求平均 (C, H, W) -> (H, W)
    2. 归一化到 0-255
    3. 应用伪彩色
    """
    # 对通道求均值，压缩为单通道热力图
    heatmap = torch.mean(f_map, dim=0).numpy()
    
    # 归一化
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    
    # 调整大小 (如果需要叠加到原图，可以在这里 resize)
    if target_size:
        heatmap = cv2.resize(heatmap, target_size)
    
    return heatmap

def visualize():
    # 1. 加载模型
    print(f"🚀 正在加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 2. 寻找 GL_ContextBlock 并注册 Hook
    # 我们遍历模型的所有层，找到名字里带 GL_ContextBlock 的层
    target_layer = None
    layer_name = ""
    
    for name, module in model.model.named_modules():
        if 'GL_ContextBlock' in module.__class__.__name__:
            target_layer = module
            layer_name = name
            print(f"✅ 找到 GL 模块: {name} ({module.__class__.__name__})")
            # 注册钩子
            module.register_forward_hook(hook_fn)
            break # 假设只有一个 GL 模块，找到就停
    
    if target_layer is None:
        print("❌ 未找到 GL_ContextBlock，请检查模型结构！")
        return

    # 3. 准备输出目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 遍历图片
    img_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    img_files = img_files[:num_images] # 限制数量
    
    print(f"📸 开始处理 {len(img_files)} 张图片...")

    for img_path in img_files:
        # 读取原图用于显示
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        H, W, _ = orig_img.shape
        
        # 5. 推理 (触发 Hook)
        # verbose=False 防止刷屏
        model(str(img_path), verbose=False)
        
        if not feature_maps:
            print("⚠️ Hook 未捕获到数据，可能模型并未走到该层 (例如图片没有触发级联？)")
            continue

        # 6. 处理特征图
        # 注意：Global 特征图通常很小 (如 8x8)，Local 较大 (如 64x64)
        local_map = process_feature_map(feature_maps['local'])
        global_map = process_feature_map(feature_maps['global'])
        output_map = process_feature_map(feature_maps['output'])
        
        # 7. 绘图
        plt.figure(figsize=(20, 5))
        
        # 子图 1: 原图
        plt.subplot(1, 4, 1)
        plt.imshow(orig_img)
        plt.title(f"Original: {img_path.name}")
        plt.axis('off')
        
        # 子图 2: Local Feature (细视输入)
        plt.subplot(1, 4, 2)
        plt.imshow(local_map, cmap='viridis') # 使用 viridis 或 jet
        plt.title(f"Local Input\n{feature_maps['local'].shape}")
        plt.axis('off')
        
        # 子图 3: Global Feature (粗视输入)
        plt.subplot(1, 4, 3)
        plt.imshow(global_map, cmap='magma')
        plt.title(f"Global Input\n{feature_maps['global'].shape}")
        plt.axis('off')
        
        # 子图 4: Fused Output (融合输出)
        plt.subplot(1, 4, 4)
        plt.imshow(output_map, cmap='inferno')
        plt.title(f"Fused Output\n{feature_maps['output'].shape}")
        plt.axis('off')
        
        # 保存
        save_name = os.path.join(save_dir, f"vis_{img_path.name}")
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()
        print(f"💾 已保存: {save_name}")
        
        # 清空数据以防下一轮污染
        feature_maps.clear()

if __name__ == "__main__":
    visualize()