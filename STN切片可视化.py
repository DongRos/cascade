# import torch
# import torch.nn.functional as F
# import cv2
# import numpy as np
# import os
# from ultralytics.nn.tasks import CascadeDetectionModel
# from ultralytics.utils.torch_utils import select_device

# def apply_stn_to_rgb(img_tensor, gaze_params):
#     """
#     手动对 RGB 原图应用 STN 变换
#     """
#     B, C, H, W = img_tensor.size()
#     tx = gaze_params[:, 0]
#     ty = gaze_params[:, 1]
#     s  = gaze_params[:, 2]

#     tx_trans = (tx - 0.5) * 2
#     ty_trans = (ty - 0.5) * 2
    
#     theta = torch.zeros(B, 2, 3, device=img_tensor.device)
#     theta[:, 0, 0] = s
#     theta[:, 1, 1] = s
#     theta[:, 0, 2] = tx_trans
#     theta[:, 1, 2] = ty_trans

#     grid = F.affine_grid(theta, img_tensor.size(), align_corners=False)
#     warped_rgb = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
#     return warped_rgb

# def visualize_stn(img_path, model_cfg, weights_path, device='0'):
#     # 1. 准备环境
#     device = select_device(device)
#     save_dir = 'runs/stn_visualization'
#     if os.path.exists(save_dir):
#         import shutil
#         shutil.rmtree(save_dir)
#     os.makedirs(save_dir)
    
#     print(f"🚀 开始 STN 可视化...")
    
#     # 2. 加载模型
#     model = CascadeDetectionModel(cfg=model_cfg, nc=10, verbose=False)
#     if weights_path and os.path.exists(weights_path):
#         ckpt = torch.load(weights_path, map_location=device)
#         model.load_state_dict(ckpt['model'].float().state_dict())
#         print("✅ 权重加载成功")
#     else:
#         print("⚠️ 未找到权重，使用随机参数")
    
#     model.to(device).eval()

#     # --- [关键修复] 自动查找层索引 ---
#     stn_layer_idx = -1
#     coarse_layer_idx = -1
    
#     print("\n🔍 正在自动定位模块层索引...")
#     for i, m in enumerate(model.model):
#         name = m.__class__.__name__
#         if name == 'DifferentiableGazeShift':
#             stn_layer_idx = i
#             print(f"  -> 找到 STN (DifferentiableGazeShift): Layer {i}")
#         elif name == 'CoarseDetect':
#             coarse_layer_idx = i
#             print(f"  -> 找到 CoarseDetect: Layer {i}")
            
#     if stn_layer_idx == -1 or coarse_layer_idx == -1:
#         print("❌ 错误：无法在模型中找到 CoarseDetect 或 STN 模块！请检查 YAML 配置。")
#         return
#     # -------------------------------

#     # 3. 注册 Hook (STN)
#     captured_data = {'params': None, 'features': []}

#     def hook_fn(module, input, output):
#         # [防御性检查] 防止 input[1] 越界
#         if len(input) > 1:
#             captured_data['params'] = input[1].detach()
#         else:
#             print(f"⚠️ Warning: STN Layer {stn_layer_idx} 仅接收到 1 个输入。tasks.py 逻辑可能未生效。")
#             # 尝试造一个假参数防止脚本崩溃
#             captured_data['params'] = torch.tensor([[0.5, 0.5, 1.0]], device=input[0].device)
            
#         captured_data['features'] = output
    
#     model.model[stn_layer_idx].register_forward_hook(hook_fn)

#     # 4. 图像预处理
#     original_cv_img = cv2.imread(img_path)
#     if original_cv_img is None:
#         raise FileNotFoundError(f"找不到图片: {img_path}")
        
#     h0, w0 = original_cv_img.shape[:2]
#     img = cv2.resize(original_cv_img, (640, 640))
#     img_in = img[:, :, ::-1].transpose(2, 0, 1)
#     img_in = np.ascontiguousarray(img_in)
#     img_tensor = torch.from_numpy(img_in).to(device).float() / 255.0
#     img_tensor = img_tensor[None]

#     # 5. 运行推理
#     with torch.no_grad():
#         model(img_tensor)

#     # 6. 可视化参数 & RGB 切片
#     params = captured_data['params']
#     if params is None: return

#     tx, ty, s = params[0].tolist()
#     print(f"\n🔍 [CoarseDetect 决策结果]")
#     print(f"   - 中心点: ({tx:.4f}, {ty:.4f})")
#     print(f"   - 缩放因子: {s:.4f}")
#     print(f"   => 视线坐标: ({tx*w0:.0f}, {ty*h0:.0f})")

#     rgb_crop_tensor = apply_stn_to_rgb(img_tensor, params)
#     rgb_crop = rgb_crop_tensor[0].cpu().numpy().transpose(1, 2, 0)
#     rgb_crop = (rgb_crop * 255).astype(np.uint8)
#     rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
    
#     cv2.imwrite(f"{save_dir}/1_RGB_Real_Crop.jpg", rgb_crop)
#     display_img = np.hstack([img, rgb_crop])
#     cv2.imwrite(f"{save_dir}/0_Comparison_RGB.jpg", display_img)
#     print(f"✅ 保存 RGB 对比图: {save_dir}/0_Comparison_RGB.jpg")

#     # 7. 可视化 Saliency Map (第二次推理 Hook CoarseDetect)
#     coarse_data = {'out': None}
#     def coarse_hook(module, input, output):
#         coarse_data['out'] = output
    
#     # 移除旧 Hook，注册新 Hook
#     # model.model[stn_layer_idx].remove_hook() # PyTorch 旧版本可能不支持，这里重新推理一次无妨
#     model.model[coarse_layer_idx].register_forward_hook(coarse_hook)
    
#     with torch.no_grad():
#         model(img_tensor)
        
#     if coarse_data['out'] is not None:
#         # coarse_outputs[0] 是分辨率最高的特征 (P2 或 P3)
#         raw_feat = coarse_data['out'][0] 
#         saliency = raw_feat[0, 0].cpu().numpy()
        
#         saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
#         heatmap = cv2.applyColorMap((saliency_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
#         heatmap = cv2.resize(heatmap, (w0, h0))
        
#         overlay = cv2.addWeighted(original_cv_img, 0.6, heatmap, 0.4, 0)
#         cv2.imwrite(f"{save_dir}/-1_Saliency_Map_Overlay.jpg", overlay)
#         print(f"🔥 保存热力图: {save_dir}/-1_Saliency_Map_Overlay.jpg")

#     # 8. 可视化特征图切片
#     feature_maps = captured_data['features']
#     if isinstance(feature_maps, list):
#         for i, feat in enumerate(feature_maps):
#             heatmap = torch.mean(feat[0], dim=0).cpu().numpy()
#             heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
#             heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
#             heatmap_view = cv2.resize(heatmap_color, (320, 320), interpolation=cv2.INTER_NEAREST)
#             cv2.imwrite(f"{save_dir}/2_Feature_Crop_Level_{i}.jpg", heatmap_view)

#     print("\n🎉 可视化完成！")

# if __name__ == '__main__':
#     # 替换路径
#     img_path = '/home/liuyadong/ultralytics-main-cascade/图片素材/0000146_01678_d_0000066.jpg'
    
#     # 请确保 yaml 文件和你训练权重是对应的 (P2版用P2 yaml)
#     visualize_stn(
#         img_path=img_path,
#         model_cfg='/home/liuyadong/ultralytics-main-cascade/yolo12s-cascade.yaml', 
#         weights_path='/home/liuyadong/ultralytics-main-cascade/runs/train/yolo12s-cascade2-多尺度加多处P2/weights/best.pt'
#     )









import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import sys

# [关键] 必须导入 tasks 中的函数来手动生成参数
from ultralytics.nn.tasks import CascadeDetectionModel, generate_gaze_params
from ultralytics.utils.torch_utils import select_device

def apply_stn_to_rgb(img_tensor, gaze_params):
    """
    手动对 RGB 原图应用 STN 变换
    Args:
        img_tensor: [B, 3, H, W] (0-1 float)
        gaze_params: [B, 3] -> (tx, ty, s) 
                     注意：必须假定 tx, ty 已经是 [-1, 1] 范围 (PyTorch grid_sample 标准)
    """
    B, C, H, W = img_tensor.size()
    
    # 提取参数
    tx = gaze_params[:, 0]
    ty = gaze_params[:, 1]
    s  = gaze_params[:, 2]

    # [关键修改] 移除之前的 (tx-0.5)*2 变换。
    # 假设 generate_gaze_params 已经修复为返回 [-1, 1] 的坐标。
    # 如果你的图还是偏的，请检查 tasks.py 是否加入了 clamp 和 range 映射。
    theta = torch.zeros(B, 2, 3, device=img_tensor.device)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, img_tensor.size(), align_corners=False)
    warped_rgb = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_rgb

def visualize_stn(img_path, model_cfg, weights_path, device='0'):
    # 1. 准备环境
    device = select_device(device)
    save_dir = 'runs/stn_visualization'
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    print(f"🚀 开始 STN 可视化...")
    
    # 2. 加载模型
    # 注意：nc=10 需要和你训练时的配置一致
    model = CascadeDetectionModel(cfg=model_cfg, nc=10, verbose=False)
    
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt['model'].float().state_dict())
        print(f"✅ 权重加载成功: {weights_path}")
    else:
        print("❌ 错误：找不到权重文件！")
        return
    
    model.to(device).eval()

    # 3. 自动定位 CoarseDetect 层
    coarse_layer_idx = -1
    for i, m in enumerate(model.model):
        if m.__class__.__name__ == 'CoarseDetect':
            coarse_layer_idx = i
            print(f"🔍 找到 CoarseDetect: Layer {i}")
            break
            
    if coarse_layer_idx == -1:
        print("❌ 错误：模型中未找到 CoarseDetect 模块！")
        return

    # 4. 注册 Hook (只抓取 CoarseDetect 的输出特征)
    # 我们不再抓取 STN 的输入，因为那可能是基于默认 Temperature 计算的
    captured_data = {'coarse_out': None}

    def coarse_hook(module, input, output):
        # output 通常是 list (多尺度)，我们取第一个(最高分辨率 P2/P3)
        if isinstance(output, list):
            captured_data['coarse_out'] = output[0].detach()
        else:
            captured_data['coarse_out'] = output.detach()
    
    model.model[coarse_layer_idx].register_forward_hook(coarse_hook)

    # 5. 图像预处理
    original_cv_img = cv2.imread(img_path)
    if original_cv_img is None:
        raise FileNotFoundError(f"找不到图片: {img_path}")
        
    h0, w0 = original_cv_img.shape[:2]
    # Resize 到 640x640 进行推理
    img_resized = cv2.resize(original_cv_img, (640, 640))
    img_in = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_in = np.ascontiguousarray(img_in)
    img_tensor = torch.from_numpy(img_in).to(device).float() / 255.0
    img_tensor = img_tensor[None] # [1, 3, 640, 640]

    # 6. 运行推理 (触发 Hook)
    print("running inference...")
    with torch.no_grad():
        model(img_tensor)

    # 7. 手动生成 STN 参数 (关键修复步骤)
    coarse_logits = captured_data['coarse_out'] # [1, 2, H_feat, W_feat]
    if coarse_logits is None:
        print("❌ Hook 未捕获到数据，请检查层索引")
        return

    # [关键] 主动调用 generate_gaze_params 并使用低温度系数
    # temperature=0.05 会让 Softmax 分布非常尖锐，强制模型选择概率最大的点
    # 注意：这里调用的是 tasks.py 里的逻辑，请确保那里已经加上了 clamp
    print("\n⚡ 正在使用 Temperature=0.05 重新生成注视参数...")
    try:
        gaze_params = generate_gaze_params(coarse_logits, temperature=0.05)
    except Exception as e:
        print(f"❌ 调用 generate_gaze_params 失败: {e}")
        print("请检查 ultralytics/nn/tasks.py 是否包含此函数")
        return

    # 8. 解析参数并打印
    tx = gaze_params[0, 0].item()
    ty = gaze_params[0, 1].item()
    s  = gaze_params[0, 2].item()

    # 将 [-1, 1] 映射回像素坐标用于显示信息
    # x_norm = (tx + 1) / 2
    center_x = ((tx + 1) / 2) * w0
    center_y = ((ty + 1) / 2) * h0
    
    print(f"\n🔍 [CoarseDetect 最终决策]")
    print(f"   - 原始 Logits Range: [{coarse_logits[0,0].min():.2f}, {coarse_logits[0,0].max():.2f}]")
    print(f"   - STN tx, ty (范围 -1~1): ({tx:.4f}, {ty:.4f})")
    print(f"   - 缩放因子 s: {s:.4f}")
    print(f"   => 映射回原图中心: (x={center_x:.0f}, y={center_y:.0f})")

    # 9. 生成可视化图 (STN 切片)
    # 使用手动生成的 gaze_params 进行切片
    rgb_crop_tensor = apply_stn_to_rgb(img_tensor, gaze_params)
    
    # 转回 OpenCV 格式
    rgb_crop = rgb_crop_tensor[0].cpu().numpy().transpose(1, 2, 0)
    rgb_crop = (rgb_crop * 255).astype(np.uint8)
    rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
    
    # 保存结果
    cv2.imwrite(f"{save_dir}/1_RGB_Real_Crop.jpg", rgb_crop)
    
    # 拼接对比图
    # 将切片放大回 640 大小方便对比
    rgb_crop_resized = cv2.resize(rgb_crop, (640, 640))
    display_img = np.hstack([img_resized, rgb_crop_resized])
    cv2.imwrite(f"{save_dir}/0_Comparison_RGB.jpg", display_img)
    print(f"✅ 保存对比图: {save_dir}/0_Comparison_RGB.jpg")

    # 10. 可视化 Saliency Map (热力图)
    # [关键修复] CoarseDetect 现在输出 Logits，必须手动 Sigmoid
    raw_saliency = coarse_logits[0, 0].cpu().numpy() # [H_feat, W_feat]
    prob_map = 1 / (1 + np.exp(-raw_saliency)) # Sigmoid function
    
    # 归一化用于显示
    heatmap_norm = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
    
    # 只有概率 > 0.3 的地方才显示颜色 (过滤背景噪声)
    # heatmap_norm[prob_map < 0.3] = 0 
    
    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_color, (w0, h0))
    
    overlay = cv2.addWeighted(original_cv_img, 0.6, heatmap_resized, 0.4, 0)
    cv2.imwrite(f"{save_dir}/-1_Saliency_Map_Overlay.jpg", overlay)
    print(f"🔥 保存热力图: {save_dir}/-1_Saliency_Map_Overlay.jpg")

    # 11. 绘制准星 (在 Saliency 图上画出模型选择的中心)
    cv2.circle(overlay, (int(center_x), int(center_y)), 10, (0, 255, 0), -1) # 绿点
    cv2.putText(overlay, "Gaze Center", (int(center_x)+15, int(center_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(f"{save_dir}/-1_Saliency_Map_Center.jpg", overlay)

    print("\n🎉 可视化完成！请查看 runs/stn_visualization 目录。")

if __name__ == '__main__':
    # 请替换为你的实际路径
    # ⚠️ 确保 weights_path 是你修复了 CoarseDetect(无Sigmoid) 后新训练的权重
    visualize_stn(
        img_path='/home/liuyadong/ultralytics-main-cascade-2/图片素材/9999953_00000_d_0000025.jpg',
        model_cfg='/home/liuyadong/ultralytics-main-cascade-2/yolo12s-cascade-EALF.yaml', 
        weights_path='/home/liuyadong/ultralytics-main-cascade-2/runs/train/yolo12s-cascade-EALF3-优化GL模块/weights/best.pt'
    )