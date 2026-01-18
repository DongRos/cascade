import torch

import torch.nn as nn
import torch.nn.functional as F
from .block import C2f, Bottleneck
import math
import ultralytics
print(f"🔥 当前加载的库路径: {ultralytics.__file__}")
# ==========================================
# 创新点一：可微注视变换模块 (Differentiable Gaze Shift)
# ==========================================
# class DifferentiableGazeShift(nn.Module):
#     def __init__(self, out_size=(640, 640)):
#         """
#         args:
#             out_size: 细视网络需要的输入尺寸 (H, W)
#         """
#         super().__init__()
#         self.out_size = out_size

#     def forward(self, x, crop_params):
#         """
#         x: 输入的全图 Tensor [B, C, H_in, W_in]
#         crop_params: 粗检测器输出的裁剪参数 [B, 3] -> (tx, ty, scale)
#                      tx, ty 范围在 [-1, 1], scale 范围 (0, 1]
#         """
#         B, C, H, W = x.shape
        
#         # 1. 构建仿射变换矩阵 theta [B, 2, 3]
#         #    [ sx, 0, tx ]
#         #    [ 0, sy, ty ]
#         theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        
#         # 缩放因子 (scale)。注意：STN中 scale越小，视野越小(放大倍数越大)
#         # 这里假设传入的 scale 是 "保留区域的比例"，例如 0.5 代表取一半长宽
#         s = crop_params[:, 2] 
#         tx = crop_params[:, 0]
#         ty = crop_params[:, 1]

#         theta[:, 0, 0] = s
#         theta[:, 1, 1] = s
#         theta[:, 0, 2] = tx
#         theta[:, 1, 2] = ty

#         # 2. 生成采样网格 (Affine Grid)
#         # 注意：size 需要是 (B, C, H_out, W_out)
#         grid = F.affine_grid(theta, torch.Size((B, C, self.out_size[0], self.out_size[1])), align_corners=False)

#         # 3. 可微采样 (Differentiable Sampling / Bilinear Interpolation)
#         # 这就是公式 V_out(x,y) 的代码实现
#         x_cropped = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

#         return x_cropped


class DifferentiableGazeShift(nn.Module):
    def __init__(self, out_size=(160, 160)):
        super().__init__()
        self.out_size = out_size

    def forward(self, x, crop_params):
        if isinstance(x, list): x = x[0]
        B, C, H, W = x.shape
        
        # crop_params: [B, 3] -> (tx, ty, s)
        tx = crop_params[:, 0]
        ty = crop_params[:, 1]
        s = crop_params[:, 2]
        
        # 构建 Affine Matrix [B, 2, 3]
        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        
        # s 控制缩放: s=1 (全图), s=0.5 (放大)
        # 对应矩阵对角线元素
        theta[:, 0, 0] = s
        theta[:, 1, 1] = s
        
        # tx, ty 控制平移
        # 在 affine_grid 中，T 是加在 (s*x) 上的
        # 我们已经把 tx, ty 处理成了 grid_sample 需要的中心坐标 [-1, 1]
        # 但 affine_grid 的公式是 x_in = theta * x_out
        # x_out 范围是 -1..1
        # 当 x_out=0 (中心) 时，我们希望 x_in = tx
        # 所以 theta[:, 0, 2] 应该直接等于 tx
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        # 生成网格
        grid = F.affine_grid(theta, torch.Size((B, C, self.out_size[0], self.out_size[1])), align_corners=False)
        
        # 采样 (padding_mode='zeros' 会产生黑边，但因为我们在 tasks.py 做了 clamp，这里应该不会触发了)
        x_cropped = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return x_cropped

# ==========================================
# 创新点二：全局-局部上下文纠缠模块 (GL-Context Entanglement)
# ==========================================
# class GL_ContextBlock(nn.Module):
#     def __init__(self, c_local, c_global, c_out, nhead=4):
#         """
#         c_local: 细视特征通道数 (Query)
#         c_global: 粗视特征通道数 (Key/Value)
#         c_out: 输出通道数
#         """
#         super().__init__()
#         self.norm_local = nn.LayerNorm(c_local)
#         self.norm_global = nn.LayerNorm(c_global)
        
#         # 这里的 Cross Attention 可以使用 PyTorch 自带的，也可以手写以更好地控制
#         # 为了方便集成，我们使用 nn.MultiheadAttention
#         # 注意：Transformer通常主要在 dim 维度操作，Conv层需要 permute
#         self.cross_attn = nn.MultiheadAttention(embed_dim=c_local, kdim=c_global, vdim=c_global, num_heads=nhead, batch_first=True)
        
#         self.proj = nn.Conv2d(c_local, c_out, 1) if c_local != c_out else nn.Identity()

#     def forward(self, x_local, x_global):
#         """
#         x_local: [B, C_l, H_l, W_l] (Fine Feature)
#         x_global: [B, C_g, H_g, W_g] (Coarse Feature)
#         """
#         B, C_l, H_l, W_l = x_local.shape
#         B, C_g, H_g, W_g = x_global.shape

#         # 1. 对齐空间特征 (Flatten)
#         # [B, H_l*W_l, C_l]
#         q = x_local.flatten(2).permute(0, 2, 1)
#         # [B, H_g*W_g, C_g]
#         k = x_global.flatten(2).permute(0, 2, 1)
#         v = x_global.flatten(2).permute(0, 2, 1)

#         # 归一化 (LayerNorm)
#         q = self.norm_local(q)
#         k = self.norm_global(k)
#         v = self.norm_global(v)

#         # 2. Cross Attention: Query=Local, Key/Value=Global
#         # 公式: Softmax(Q * K.T / sqrt(d)) * V
#         attn_out, _ = self.cross_attn(query=q, key=k, value=v)

#         # 3. 残差连接 + 形状还原
#         # Fused = LayerNorm(Q + Attention) ... 这里简单实现为直接相加后输出
#         out = q + attn_out
        
#         # [B, L, C] -> [B, C, L] -> [B, C, H, W]
#         out = out.permute(0, 2, 1).view(B, C_l, H_l, W_l)
        
#         return self.proj(out)
    







class GL_ContextBlock(nn.Module):
    def __init__(self, c_local, c_global, c_out=None, nhead=4, dropout=0.0):
        """
        优化后的全局-局部上下文纠缠模块 (Global-Local Context Entanglement)
        
        Args:
            c_local (int): 细视特征通道数 (Query)
            c_global (int): 粗视特征通道数 (Key/Value)
            c_out (int, optional): 输出通道数. 默认为 None，即等于 c_local
            nhead (int): 多头注意力的头数
            dropout (float): Dropout 比率
        """
        super().__init__()
        # 如果未指定输出通道，默认保持与 local 一致
        if c_out is None:
            c_out = c_local
            
        self.c_local = c_local
        
        # 1. 特征对齐投影: 将 Global 特征映射到与 Local 相同的维度
        # 这有助于在统一的语义空间计算相似度
        self.proj_global_k = nn.Conv2d(c_global, c_local, 1)
        self.proj_global_v = nn.Conv2d(c_global, c_local, 1)
        
        # 2. LayerNorm (Pre-Norm 结构)
        self.norm_l = nn.LayerNorm(c_local)
        self.norm_g = nn.LayerNorm(c_local)
        self.norm_ffn = nn.LayerNorm(c_local)

        # 3. Cross Attention
        # 优化: 此时 kdim, vdim 均已对齐为 c_local
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=c_local, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 4. FFN (Feed-Forward Network) - 增强非线性表达
        # 使用 ConvFFN (Conv1x1 -> DWConv3x3 -> GELU -> Conv1x1) 
        # 相比普通 MLP，DWConv 能更好地提取局部特征，防止位置信息丢失
        hidden_dim = c_local * 4
        self.ffn = nn.Sequential(
            nn.Conv2d(c_local, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim), # Depthwise
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, c_local, 1),
            nn.Dropout(dropout)
        )

        # 5. 输出投影
        self.proj_out = nn.Conv2d(c_local, c_out, 1) if c_local != c_out else nn.Identity()

    def _get_abs_pos_encoding(self, x):
        """
        动态生成 2D 正弦位置编码 (Sinusoidal Positional Encoding)
        x: [B, C, H, W]
        Return: [1, L, C] (L=H*W)
        """
        B, C, H, W = x.shape
        # 简单的实现：生成归一化的网格坐标
        # 注意：为了效率，实际部署时可以缓存，但动态生成对多尺度训练更鲁棒
        device = x.device
        
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W).view(-1)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).repeat(H, 1).view(-1)
        
        # 将坐标归一化并缩放，模拟频率
        # 这里使用简化版的 PE，将 x, y 坐标直接作为辅助特征加进去
        # 如果追求极致，可以使用标准的 sin/cos 公式，但这通常足够了
        
        # 为了不破坏维度，我们简单地用 sin/cos 处理一下坐标
        div_term = torch.exp(torch.arange(0, C, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / C))
        
        # [H*W, 1] * [C/2] -> [H*W, C/2]
        pe_x = torch.zeros(H * W, C, device=device)
        pe_y = torch.zeros(H * W, C, device=device)
        
        pe_x[:, 0::2] = torch.sin(x_embed.unsqueeze(1) * div_term)
        pe_x[:, 1::2] = torch.cos(x_embed.unsqueeze(1) * div_term)
        
        pe_y[:, 0::2] = torch.sin(y_embed.unsqueeze(1) * div_term)
        pe_y[:, 1::2] = torch.cos(y_embed.unsqueeze(1) * div_term)
        
        # 融合 X 和 Y 的位置信息 (简单平均)
        pe = (pe_x + pe_y) / 2.0
        return pe.unsqueeze(0) # [1, L, C]

    def forward(self, x_local, x_global):
        # [添加这一行]
        if not hasattr(self, 'traced'): # 只打印一次，防止刷屏
            print(f"\n✅ [验证成功] GL_ContextBlock 正在运行! 输入尺寸: Local={x_local.shape}, Global={x_global.shape}")
            self.traced = True
        """
        x_local:  [B, C_l, H_l, W_l] (Fine, Query)
        x_global: [B, C_g, H_g, W_g] (Coarse, Key/Value)
        """
        B, C_l, H_l, W_l = x_local.shape
        B, C_g, H_g, W_g = x_global.shape

        # --- 1. 预处理 Global 特征 ---
        # 投影 Key 和 Value 到 Local 维度，方便计算
        k_src = self.proj_global_k(x_global) # [B, C_l, H_g, W_g]
        v_src = self.proj_global_v(x_global) # [B, C_l, H_g, W_g]

        # --- 2. 展平 (Flatten) 并添加位置编码 ---
        # Query: Local
        q = x_local.flatten(2).permute(0, 2, 1) # [B, L_l, C]
        # 添加 PE 给 Query (可选，但推荐)
        q_pe = self._get_abs_pos_encoding(x_local)
        
        # Key/Value: Global
        k = k_src.flatten(2).permute(0, 2, 1)   # [B, L_g, C]
        v = v_src.flatten(2).permute(0, 2, 1)   # [B, L_g, C]
        # 添加 PE 给 Key (非常关键！让 Query 知道 Global 特征在哪里)
        k_pe = self._get_abs_pos_encoding(k_src)

        # --- 3. Attention Block (Pre-Norm) ---
        # Norm -> Attn -> Add
        q_norm = self.norm_l(q)
        k_norm = self.norm_g(k)
        
        # 注意: 传入 attn 的 query 和 key 加上位置编码，value 不加
        # 这是一种常见的 Transformer 优化 (如 DETR)
        attn_out, _ = self.cross_attn(
            query = q_norm + q_pe, 
            key   = k_norm + k_pe, 
            value = v
        )
        
        # 残差连接 1
        x = q + attn_out # [B, L_l, C]

        # --- 4. FFN Block (Pre-Norm) ---
        # 需要先 reshape 回 2D 进行卷积 FFN
        x_2d = x.permute(0, 2, 1).view(B, C_l, H_l, W_l)
        
        # FFN: Norm -> ConvFFN -> Add
        # 这里为了配合 LayerNorm，先 flatten 再 norm 再 reshape，或者直接用 GroupNorm
        # 为保持一致性，我们手动处理 LayerNorm
        x_norm = self.norm_ffn(x).permute(0, 2, 1).view(B, C_l, H_l, W_l)
        
        ffn_out = self.ffn(x_norm)
        
        # 残差连接 2
        out = x_2d + ffn_out # [B, C_l, H_l, W_l]

        # --- 5. 最终输出 ---
        return self.proj_out(out)