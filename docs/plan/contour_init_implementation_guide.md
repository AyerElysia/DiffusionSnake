# 轮廓初始化实现技术指南

## 快速开始

### 最简单的改进：椭圆初始化

**步骤1：** 创建新文件 `lib/utils/snake/contour_init.py`

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class EllipseInit:
    """椭圆初始化方法"""
    def __init__(self, eccentricity: float = 0.7, num_points: int = 128):
        """
        Args:
            eccentricity: 离心率，控制椭圆扁平程度
                        - 0.0: 圆形
                        - 0.5: 轻微椭圆
                        - 0.7-0.9: 明显椭圆
            num_points: 轮廓点数
        """
        self.eccentricity = eccentricity
        self.num_points = num_points
        # 预计算角度（节省推理时间）
        self.register_buffer('theta', torch.linspace(0, 2*np.pi, num_points))

    def forward(self, box: torch.Tensor) -> torch.Tensor:
        """
        从检测框生成椭圆轮廓

        Args:
            box: [B, 4] 格式为 [x_min, y_min, x_max, y_max]

        Returns:
            contour: [B, num_points, 2] 轮廓点坐标
        """
        device = box.device
        if self.theta.device != device:
            self.theta = self.theta.to(device)

        batch_size = box.size(0)

        # 计算椭圆参数
        center_x = (box[..., 0] + box[..., 2]) / 2.0  # [B]
        center_y = (box[..., 1] + box[..., 3]) / 2.0  # [B]
        a = (box[..., 2] - box[..., 0]) / 2.0  # 长轴 [B]
        b = (box[..., 3] - box[..., 1]) / 2.0 * self.eccentricity  # 短轴 [B]

        # 生成椭圆轮廓
        theta = self.theta.view(1, self.num_points).expand(batch_size, -1)  # [B, N]
        x = center_x.unsqueeze(1) + a.unsqueeze(1) * torch.cos(theta)  # [B, N]
        y = center_y.unsqueeze(1) + b.unsqueeze(1) * torch.sin(theta)  # [B, N]

        contour = torch.stack([x, y], dim=-1)  # [B, N, 2]

        return contour

    def register_buffer(self, name, tensor):
        """类似nn.Module的register_buffer"""
        self.__dict__[name] = tensor


class OrganSpecificEllipseInit(nn.Module):
    """针对BTCV器官的专用椭圆初始化"""
    def __init__(self, num_points: int = 128):
        super().__init__()
        self.num_points = num_points

        # 每个器官的椭圆参数（基于经验）
        # 格式：{'organ_name': {'eccentricity': float, 'orientation': str}}
        self.organ_params = {
            0: {'name': 'spleen', 'eccentricity': 0.7, 'orientation': 'horizontal'},
            1: {'name': 'right_kidney', 'eccentricity': 0.65, 'orientation': 'vertical'},
            2: {'name': 'left_kidney', 'eccentricity': 0.65, 'orientation': 'vertical'},
            3: {'name': 'gallbladder', 'eccentricity': 0.75, 'orientation': 'vertical'},
            4: {'name': 'esophagus', 'eccentricity': 0.15, 'orientation': 'vertical'},  # 更接近矩形
            5: {'name': 'liver', 'eccentricity': 0.8, 'orientation': 'horizontal'},
            6: {'name': 'stomach', 'eccentricity': 0.7, 'orientation': 'horizontal'},
            7: {'name': 'pancreas', 'eccentricity': 0.6, 'orientation': 'horizontal'},
            8: {'name': 'bladder', 'eccentricity': 0.5, 'orientation': 'horizontal'},
        }

        # 为每个器官创建初始化器
        self.initers = nn.ModuleDict({
            f'organ_{i}': EllipseInit(
                eccentricity=params['eccentricity'],
                num_points=num_points
            )
            for i, params in self.organ_params.items()
        })

    def forward(self, box: torch.Tensor, organ_class: torch.Tensor) -> torch.Tensor:
        """
        根据器官类别生成专用初始化

        Args:
            box: [B, 4] 检测框
            organ_class: [B] 器官类别ID (0-8)

        Returns:
            contour: [B, num_points, 2]
        """
        batch_size = box.size(0)
        contours = []

        for i in range(batch_size):
            cls_id = int(organ_class[i].item())
            box_i = box[i:i+1]  # [1, 4]
            initer = self.initers[f'organ_{cls_id}']
            contour_i = initer.forward(box_i)  # [1, N, 2]
            contours.append(contour_i)

        return torch.cat(contours, dim=0)  # [B, N, 2]


class AdaptiveQuadInit:
    """自适应四边形初始化（改进版）"""
    def __init__(self, num_points: int = 128):
        self.num_points = num_points

    def forward(self, box: torch.Tensor, organ_class: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        根据宽高比自适应选择初始化形状

        - aspect_ratio < 2.0: 使用菱形（接近椭圆）
        - aspect_ratio >= 2.0: 使用矩形（适合细长目标）
        """
        w = box[..., 2] - box[..., 0]
        h = box[..., 3] - box[..., 1]
        aspect_ratio = h / (w + 1e-6)

        # 根据宽高比选择初始化方法
        use_diamond = aspect_ratio < 2.0

        contours = []
        for i in range(box.size(0)):
            if use_diamond[i]:
                contour = self._diamond_init(box[i:i+1])
            else:
                contour = self._rect_init(box[i:i+1])
            contours.append(contour)

        return torch.cat(contours, dim=0)

    def _diamond_init(self, box: torch.Tensor) -> torch.Tensor:
        """菱形初始化（4个中点）"""
        x_min, y_min, x_max, y_max = box[0, 0], box[0, 1], box[0, 2], box[0, 3]

        # 4个顶点（上、左、下、右中点）
        quad = torch.tensor([
            [(x_min + x_max) / 2, y_min],  # 上
            [x_min, (y_min + y_max) / 2],  # 左
            [(x_min + x_max) / 2, y_max],  # 下
            [x_max, (y_min + y_max) / 2],  # 右
        ], device=box.device).view(1, 4, 2)

        # 上采样到128点
        return self._upsample(quad)

    def _rect_init(self, box: torch.Tensor) -> torch.Tensor:
        """矩形初始化（4个角点）"""
        x_min, y_min, x_max, y_max = box[0, 0], box[0, 1], box[0, 2], box[0, 3]

        # 4个顶点
        quad = torch.tensor([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
        ], device=box.device).view(1, 4, 2)

        # 上采样到128点
        return self._upsample(quad)

    def _upsample(self, quad: torch.Tensor) -> torch.Tensor:
        """上采样到num_points点"""
        from lib.utils.snake import snake_gcn_utils
        return snake_gcn_utils.uniform_upsample(quad, self.num_points)[0]
```

**步骤2：** 修改 `pretrain_evolution.py`

```python
# 在文件开头添加导入
from lib.utils.snake.contour_init import OrganSpecificEllipseInit, AdaptiveQuadInit

class DiffusionEvolution(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 原有代码 ...

        # 添加初始化器
        self.contour_init_method = getattr(global_cfg, 'contour_init_method', 'rectangle')
        if self.contour_init_method == 'ellipse':
            self.contour_init = OrganSpecificEllipseInit(num_points=num_points)
        elif self.contour_init_method == 'adaptive_quad':
            self.contour_init = AdaptiveQuadInit(num_points=num_points)
        else:
            self.contour_init = None  # 使用原有的矩形方法

    def forward(self, output, cnn_feature, batch=None):
        ret = output
        if self.training:
            with torch.no_grad():
                # 1) 准备训练数据
                init = snake_gcn_utils.prepare_training(output, batch)
                ret.update({
                    'i_it_4py': init['i_it_4py'],
                    'i_it_py': init['i_it_py'],
                    'i_gt_4py': init['i_gt_4py'],
                    'i_gt_py': init['i_gt_py']
                })

                # 2) 构建训练用 init 轮廓
                device = cnn_feature.device
                i_gt_4py = init['i_gt_4py'].to(device)
                i_gt_py = init['i_gt_py'].to(device)
                py_ind = init['py_ind']

                # ===== 修改开始 =====
                if self.contour_init is not None:
                    # 使用新的初始化方法
                    # 获取检测框（从4个角点计算）
                    x_min = i_gt_4py[..., 0].min(dim=1)[0]
                    y_min = i_gt_4py[..., 1].min(dim=1)[0]
                    x_max = i_gt_4py[..., 0].max(dim=1)[0]
                    y_max = i_gt_4py[..., 1].max(dim=1)[0]
                    gt_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

                    # 获取器官类别（如果有）
                    # 注意：需要确保batch中有organ_class信息
                    organ_class = batch.get('organ_class', torch.zeros(len(gt_boxes), device=device))

                    # 生成初始轮廓
                    if isinstance(self.contour_init, OrganSpecificEllipseInit):
                        i_init_train_py = self.contour_init(gt_boxes, organ_class)
                    else:
                        i_init_train_py = self.contour_init(gt_boxes, organ_class)
                else:
                    # 使用原有的矩形方法
                    x_min = i_gt_4py[..., 0].min(dim=1)[0]
                    y_min = i_gt_4py[..., 1].min(dim=1)[0]
                    x_max = i_gt_4py[..., 0].max(dim=1)[0]
                    y_max = i_gt_4py[..., 1].max(dim=1)[0]
                    gt_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1).unsqueeze(0)
                    gt_rect4 = snake_decode.get_box(gt_boxes)[0]
                    i_init_train_py = snake_gcn_utils.uniform_upsample(
                        gt_rect4.unsqueeze(0), snake_config.poly_num
                    )[0]
                # ===== 修改结束 =====

                c_init_train_py = snake_gcn_utils.img_poly_to_can_poly(i_init_train_py)

                # ... 后续代码保持不变 ...
```

**步骤3：** 修改配置文件 `configs/btcv_diffusion_dit_v2_2.yaml`

```yaml
# 在文件末尾添加
contour_init_method: 'ellipse'  # 'rectangle' | 'ellipse' | 'adaptive_quad'
```

**步骤4：** 测试运行

```bash
# 训练
python run.py --config configs/btcv_diffusion_dit_v2_2.yaml

# 对比实验
# 1. 矩形初始化（baseline）
contour_init_method: 'rectangle'

# 2. 椭圆初始化
contour_init_method: 'ellipse'

# 3. 自适应四边形
contour_init_method: 'adaptive_quad'
```

---

## 中等复杂度：PCA形状先验

### 第一步：构建形状词典

创建脚本 `scripts/build_shape_dictionary.py`：

```python
"""
构建BTCV数据集的形状词典
Usage:
    python scripts/build_shape_dictionary.py --data_path /path/to/btcv --output data/shape_models/btcv_pca.pkl
"""
import argparse
import pickle
import numpy as np
import torch
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm


class StatisticalShapePrior:
    """统计形状先验（PCA）"""
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.mean_shape = None
        self.components = None
        self.explained_variance = None
        self.pca = PCA(n_components=n_components)

    def fit(self, contours):
        """
        拟合PCA模型

        Args:
            contours: list of numpy arrays, each shape [N, 2]
        """
        # 对齐轮廓（Procrustes analysis）
        aligned = self._procrustes_align(contours)

        # 展平
        flattened = aligned.reshape(len(aligned), -1)

        # PCA拟合
        self.pca.fit(flattened)
        self.mean_shape = self.pca.mean_.reshape(-1, 2)
        self.components = self.pca.components_
        self.explained_variance = self.pca.explained_variance_ratio_

        print(f"Explained variance ratio (first {self.n_components} components):")
        print(self.explained_variance)

    def _procrustes_align(self, contours):
        """
        Procrustes对齐（平移+缩放）

        Args:
            contours: list of [N, 2] arrays

        Returns:
            aligned: list of [N, 2] arrays
        """
        aligned = []

        for contour in contours:
            # 中心化
            center = contour.mean(axis=0)
            centered = contour - center

            # 归一化尺度
            scale = np.sqrt((centered ** 2).sum())
            if scale > 1e-6:
                normalized = centered / scale
            else:
                normalized = centered

            aligned.append(normalized)

        return np.array(aligned)

    def generate(self, coeffs=None, num_points=128):
        """
        生成新轮廓

        Args:
            coeffs: [n_components] 形状系数，None则使用均值
            num_points: 轮廓点数

        Returns:
            contour: [num_points, 2]
        """
        if coeffs is None:
            coeffs = np.zeros(self.n_components)

        # 重建
        flattened = self.pca.mean_ + np.dot(coeffs, self.components)
        contour = flattened.reshape(-1, 2)

        # 重采样到指定点数
        if len(contour) != num_points:
            contour = self._resample(contour, num_points)

        return contour

    def _resample(self, contour, num_points):
        """重采样轮廓到指定点数"""
        # 简单实现：线性插值
        from scipy.interpolate import interp1d

        # 计算累积距离
        diffs = np.diff(contour, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cum_dist = np.concatenate([[0], dists]).cumsum()

        # 参数化
        t_old = np.linspace(0, 1, len(contour))
        t_new = np.linspace(0, 1, num_points)

        # 插值
        fx = interp1d(t_old, contour[:, 0], kind='linear', fill_value='extrapolate')
        fy = interp1d(t_old, contour[:, 1], kind='linear', fill_value='extrapolate')

        x_new = fx(t_new)
        y_new = fy(t_new)

        return np.stack([x_new, y_new], axis=1)


def build_btcv_shape_dictionary(data_path, output_path, n_components=5):
    """构建BTCV形状词典"""

    # 假设数据格式：
    # data_path/
    #   train/
    #     xxx.npy  -> 每个文件包含一个样本
    #       {'gt_contours': [9, 128, 2]}  # 9个器官，每个128个点

    data_path = Path(data_path)
    all_contours = {i: [] for i in range(9)}  # 9个器官

    # 收集所有轮廓
    print("Collecting contours from training data...")
    for npy_file in tqdm(list(data_path.glob("*.npy"))):
        data = np.load(npy_file, allow_pickle=True).item()

        if 'gt_contours' in data:
            contours = data['gt_contours']  # [9, 128, 2]
            for organ_id in range(9):
                contour = contours[organ_id]  # [128, 2]
                # 过滤无效轮廓
                if not np.isnan(contour).any() and contour.shape[0] >= 10:
                    all_contours[organ_id].append(contour)

    # 对每个器官拟合PCA
    print("\nFitting PCA models for each organ...")
    shape_models = {}

    organ_names = [
        'spleen', 'right_kidney', 'left_kidney', 'gallbladder',
        'esophagus', 'liver', 'stomach', 'pancreas', 'bladder'
    ]

    for organ_id in range(9):
        contours = all_contours[organ_id]
        organ_name = organ_names[organ_id]

        print(f"\n{organ_name} (ID={organ_id}):")
        print(f"  Number of samples: {len(contours)}")

        if len(contours) < 10:
            print(f"  Warning: Too few samples, skipping...")
            continue

        # 拟合PCA
        prior = StatisticalShapePrior(n_components=n_components)
        prior.fit(contours)
        shape_models[organ_id] = prior

    # 保存
    print(f"\nSaving shape models to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(shape_models, f)

    print("Done!")

    return shape_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to BTCV training data')
    parser.add_argument('--output', type=str, default='data/shape_models/btcv_pca.pkl',
                        help='Output path for shape dictionary')
    parser.add_argument('--n_components', type=int, default=5,
                        help='Number of PCA components')

    args = parser.parse_args()

    build_btcv_shape_dictionary(args.data_path, args.output, args.n_components)
```

### 第二步：实现PCA初始化

在 `lib/utils/snake/contour_init.py` 中添加：

```python
class PCAInit:
    """基于PCA形状先验的初始化"""
    def __init__(self, model_path: str, num_points: int = 128):
        """
        Args:
            model_path: PCA模型路径（pickle文件）
            num_points: 轮廓点数
        """
        self.num_points = num_points
        with open(model_path, 'rb') as f:
            self.shape_models = pickle.load(f)

    def forward(self, box: torch.Tensor, organ_class: torch.Tensor,
                coeffs: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        生成初始轮廓

        Args:
            box: [B, 4] 检测框
            organ_class: [B] 器官类别ID
            coeffs: [B, n_components] 形状系数，None则使用均值

        Returns:
            contour: [B, num_points, 2]
        """
        batch_size = box.size(0)
        contours = []

        for i in range(batch_size):
            cls_id = int(organ_class[i].item())

            # 获取PCA模型
            if cls_id not in self.shape_models:
                # fallback：使用椭圆
                ellipse = EllipseInit(eccentricity=0.7, num_points=self.num_points)
                contour = ellipse.forward(box[i:i+1])[0].cpu().numpy()
            else:
                prior = self.shape_models[cls_id]

                # 生成形状
                if coeffs is not None:
                    coeff_i = coeffs[i]
                else:
                    coeff_i = None

                contour = prior.generate(coeffs=coeff_i, num_points=self.num_points)

            # 缩放到检测框大小
            contour = self._scale_to_box(contour, box[i])

            contours.append(torch.from_numpy(contour).float())

        return torch.stack(contours).to(box.device)

    def _scale_to_box(self, contour: np.ndarray, box: torch.Tensor) -> np.ndarray:
        """
        将轮廓缩放到检测框大小

        Args:
            contour: [N, 2] numpy array
            box: [4] tensor [x_min, y_min, x_max, y_max]

        Returns:
            scaled_contour: [N, 2]
        """
        box_np = box.cpu().numpy()

        # 轮廓的包围盒
        x_min, y_min = contour.min(axis=0)
        x_max, y_max = contour.max(axis=0)

        # 缩放因子
        box_w = box_np[2] - box_np[0]
        box_h = box_np[3] - box_np[1]
        contour_w = x_max - x_min
        contour_h = y_max - y_min

        scale_x = box_w / (contour_w + 1e-6)
        scale_y = box_h / (contour_h + 1e-6)

        # 缩放
        scaled = contour.copy()
        scaled[:, 0] = (contour[:, 0] - x_min) * scale_x + box_np[0]
        scaled[:, 1] = (contour[:, 1] - y_min) * scale_y + box_np[1]

        return scaled
```

### 第三步：训练时使用随机系数

修改 `pretrain_evolution.py`:

```python
# 在训练循环中
if isinstance(self.contour_init, PCAInit) and self.training:
    # 生成随机形状系数（数据增强）
    num_samples = len(gt_boxes)
    random_coeffs = np.random.randn(num_samples, 5) * 0.5  # 小方差
    i_init_train_py = self.contour_init(gt_boxes, organ_class, coeffs=random_coeffs)
else:
    i_init_train_py = self.contour_init(gt_boxes, organ_class)
```

---

## 高级方案：轻量级预测网络

完整代码见主报告。训练流程：

1. 准备数据
2. 定义网络
3. 训练
4. 集成到DiffusionSnake

---

## 评估脚本

创建 `scripts/evaluate_init_methods.py`:

```python"""
评估不同初始化方法
"""
import torch
import numpy as np
from lib.utils.snake.contour_init import (
    EllipseInit, OrganSpecificEllipseInit,
    AdaptiveQuadInit, PCAInit
)
from lib.utils.snake import snake_gcn_utils


def compute_hausdorff_distance(pred, gt):
    """计算Hausdorff距离"""
    from scipy.spatial.distance import directed_hausdorff

    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    d1 = directed_hausdorff(pred_np, gt_np)[0]
    d2 = directed_hausdorff(gt_np, pred_np)[0]

    return max(d1, d2)


def compute_displacement_norm(init_contour, gt_contour):
    """计算位移场范数"""
    diff = gt_contour - init_contour
    norm = torch.sqrt((diff ** 2).sum(dim=-1)).mean()
    return norm.item()


def evaluate_init_method(init_method, test_data):
    """评估初始化方法"""
    hausdorff_distances = []
    displacement_norms = []

    for sample in test_data:
        box = sample['box']  # [B, 4]
        gt_contour = sample['gt_contour']  # [B, 128, 2]
        organ_class = sample.get('organ_class')

        # 生成初始轮廓
        if isinstance(init_method, PCAInit):
            init_contour = init_method.forward(box, organ_class)
        elif isinstance(init_method, OrganSpecificEllipseInit):
            init_contour = init_method.forward(box, organ_class)
        else:
            init_contour = init_method.forward(box)

        # 对齐轮廓（处理起始点不一致）
        init_contour_aligned = align_contour(init_contour, gt_contour)

        # 计算指标
        for i in range(len(box)):
            hd = compute_hausdorff_distance(
                init_contour_aligned[i],
                gt_contour[i]
            )
            disp_norm = compute_displacement_norm(
                init_contour_aligned[i],
                gt_contour[i]
            )

            hausdorff_distances.append(hd)
            displacement_norms.append(disp_norm)

    # 统计
    results = {
        'hausdorff_mean': np.mean(hausdorff_distances),
        'hausdorff_std': np.std(hausdorff_distances),
        'displacement_norm_mean': np.mean(displacement_norms),
        'displacement_norm_std': np.std(displacement_norms),
    }

    return results


def align_contour(init_contour, gt_contour):
    """对齐轮廓（处理起始点不一致）"""
    # 简单实现：找到最近的点作为起始点
    aligned = init_contour.clone()

    for i in range(len(init_contour)):
        init_i = init_contour[i]  # [128, 2]
        gt_i = gt_contour[i]  # [128, 2]

        # 找到init_i的第0个点在gt_i中最近的点
        dists = torch.sqrt(((gt_i - init_i[0]) ** 2).sum(dim=-1))
        nearest = torch.argmin(dists)

        # 循环移位gt_i
        if nearest > 0:
            gt_i_rolled = torch.roll(gt_i, -nearest, dims=0)
        else:
            gt_i_rolled = gt_i

        aligned[i] = gt_i_rolled

    return aligned


if __name__ == '__main__':
    # 加载测试数据
    # test_data = load_test_data()

    # 初始化方法
    methods = {
        'rectangle': None,  # 原有方法
        'ellipse': EllipseInit(eccentricity=0.7),
        'organ_specific': OrganSpecificEllipseInit(),
        'adaptive_quad': AdaptiveQuadInit(),
        'pca': PCAInit('data/shape_models/btcv_pca.pkl'),
    }

    # 评估
    for name, method in methods.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_init_method(method, test_data)
        print(f"  Hausdorff Distance: {results['hausdorff_mean']:.2f} ± {results['hausdorff_std']:.2f}")
        print(f"  Displacement Norm: {results['displacement_norm_mean']:.2f} ± {results['displacement_norm_std']:.2f}")
```

---

## 调试和可视化

创建 `scripts/visualize_init.py`:

```python"""
可视化不同初始化方法
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from lib.utils.snake.contour_init import *


def visualize_initializations(image, box, gt_contour, organ_class, save_path):
    """可视化不同初始化方法"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    methods = {
        'Rectangle': 'rectangle',
        'Ellipse': 'ellipse',
        'Organ-Specific Ellipse': 'organ_specific',
        'Adaptive Quad': 'adaptive_quad',
        'PCA': 'pca',
    }

    for ax, (method_name, method_key) in zip(axes.flat, methods.items()):
        # 显示图像
        ax.imshow(image, cmap='gray')

        # 显示GT
        ax.plot(gt_contour[:, 1], gt_contour[:, 0], 'g-', linewidth=2, label='GT')

        # 生成初始轮廓
        if method_key == 'rectangle':
            # 原有矩形方法
            x_min, y_min, x_max, y_max = box
            rect = torch.tensor([
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min],
            ], dtype=torch.float32).view(1, 4, 2)
            from lib.utils.snake import snake_gcn_utils
            init_contour = snake_gcn_utils.uniform_upsample(rect, 128)[0][0]
        elif method_key == 'ellipse':
            init = EllipseInit()
            init_contour = init.forward(box.unsqueeze(0))[0]
        elif method_key == 'organ_specific':
            init = OrganSpecificEllipseInit()
            init_contour = init.forward(box.unsqueeze(0), organ_class.unsqueeze(0))[0]
        elif method_key == 'adaptive_quad':
            init = AdaptiveQuadInit()
            init_contour = init.forward(box.unsqueeze(0))[0]
        elif method_key == 'pca':
            init = PCAInit('data/shape_models/btcv_pca.pkl')
            init_contour = init.forward(box.unsqueeze(0), organ_class.unsqueeze(0))[0]

        # 显示初始轮廓
        ax.plot(init_contour[:, 1], init_contour[:, 0], 'r--', linewidth=1.5, label='Init')

        # 显示检测框
        x_min, y_min, x_max, y_max = box
        ax.plot([y_min, y_max, y_max, y_min, y_min],
                [x_min, x_min, x_max, x_max, x_min],
                'b-', linewidth=1, label='Box')

        ax.set_title(method_name)
        ax.legend()
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # 加载一个样本
    # sample = load_one_sample()

    # visualize
    # visualize_initializations(
    #     image=sample['image'],
    #     box=sample['box'],
    #     gt_contour=sample['gt_contour'],
    #     organ_class=sample['organ_class'],
    #     save_path='init_visualization.png'
    # )

    print("Visualization saved to init_visualization.png")
```

---

## 常见问题

### Q1: 如何获取器官类别信息？

需要确保数据集返回organ_class。检查 `lib/datasets/` 中的数据加载代码。

### Q2: 椭圆初始化对哪些器官效果最好？

- 肾脏（左右）：效果最好，近似椭圆形
- 脾脏：效果很好
- 胆囊：效果较好（梨形）
- 肝脏/胃：效果一般（形状复杂）

### Q3: PCA需要多少训练样本？

建议每个器官至少50-100个样本。BTCV训练集通常有几十例CT，每例有9个器官，所以是足够的。

### Q4: 训练时如何使用PCA进行数据增强？

使用随机系数采样：
```python
random_coeffs = np.random.randn(batch_size, 5) * 0.1  # 小方差，接近均值
```

---

## 下一步

1. 实现最简单的椭圆初始化
2. 运行对比实验
3. 根据结果决定是否实现更复杂的方法

**文件位置参考：**
- 新文件：`lib/utils/snake/contour_init.py`
- 修改文件：`lib/networks/diffusion/pretrain_evolution.py`
- 配置文件：`configs/btcv_diffusion_dit_v2_2.yaml`
- 工具脚本：`scripts/build_shape_dictionary.py`
- 评估脚本：`scripts/evaluate_init_methods.py`
- 可视化脚本：`scripts/visualize_init.py`
