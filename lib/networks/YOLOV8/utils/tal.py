# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args（输入）:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
                预测类别分数，shape[16(批大小), 8400, 28(类别数)]，其中8400是三个层级的特征图总点数，也就是锚的数量。见model_zoo/YOLOV8/utils/loss.py里的class v8PoseLoss
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
                解码后的预测边界框，shape[16(批大小), 8400, 4]
            anc_points (Tensor): shape(num_total_anchors, 2)
                锚点坐标，shape[8400, 2]
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
                金标准类别，shape[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 1]
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
                金标准外接矩形，shape[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 4]
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
                金标准掩膜，shape[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 1]，表示哪些目标是真实目标，哪些是补零。

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]  # 批大小16
        self.n_max_boxes = gt_bboxes.shape[1]  # 当前批次数据中、含有金标准个数最多的图像中的金标准个数（例如，9）
        if self.n_max_boxes == 0:  # 如果当前批次中没有金标准，直接返回空目标和背景标注。
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )  # 计算每个锚点与金标准之间的匹配分数（align_metric）和 IoU（overlaps），并生成正例锚掩码（mask_pos）。具体在get_pos_mask函数里。
        # 上句，三个输出的shape都是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]。
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        # 上句，target_gt_idx和fg_mask的shape是[16(批大小), 8400]。mask_pos的shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]。
        #  target_gt_idx中的第i,j个数就表示，批次里第i图像上的第j个锚点，分别对应着第几个金标准。
        #  fg_mask的shape是[16(批大小), 8400]，其中第i,j个数表示第i张图像上的第j个锚框匹配上了多少个金标准（不过经过select_highest_overlaps之后，要么是0个要么是1个，验证了一下，确实如此）。
        #     感觉很像是NMS，因为他说如果一个锚框被分配给了多个gt，就选最高的。其实，反过来也是一样的。
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # 上句，给正例锚框找到对应的标签、外接矩形和一热标签（跟前面那个标签是一样的信息，只不过表示方式不同，后面会处理这个，使其不再是1而与匹配分数有关）。
        #     target_labels是[16(批大小), 8400]的，里面第i,j个元素就是第i张图像中第j个锚框对应的金标准标签，如果是正例就是有标签，否则是0；
        #     剩下俩原理相似，shape分别是[16(批大小), 8400, 4]和[16(批大小), 8400, 28(类别数)]
        align_metric *= mask_pos  # 再次将无效匹配的分数置零（因为后续用select_highest_overlaps更新过mask_pos，删除了一部分正例锚）
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        # shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数)]。
        # 这样，每一张图像对应的pos_align_metrics都是一个9维的向量，其中对应真的金标准的那些元素，就是与这个金标准匹配度最大的锚，与他的align_metric（匹配分数）。
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        # shape同上。先把未匹配的部分清零，然后找最大的重合度。
        # 类似于上面，每一张图像对应的pos_overlaps都是一个9维的向量，其中对应真的金标准的那些元素，就是与这个金标准匹配度最大的锚，与他的IoU。
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # 可以理解为，对匹配分数align_metric做了个归一化。输出shape是[16(批大小), 8400, 1]。分子是将匹配分数align_metric与对应的最大 IoU 相乘，进一步加权。
        #     分母是将匹配分数归一化，避免数值过大。amax(-2)是要对9(当前批次数据中、含有金标准个数最多的图像中的金标准个数)那个维度取最大值，
        #     也就是说，是要取与这个锚匹配最好的金标准，他俩之间的归一化匹配分数作为norm_align_metric。
        target_scores = target_scores * norm_align_metric  # 归一化值乘到一热标签之上，做个加权。
        # 这样，target_scores里的那些非零值，就从1变成了跟匹配分数align_metric相关的数了，也就是说匹配度越好，这个分数就会越高。。
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
        # 返回的是正例锚对应的金标准标签、金标准外接矩形、匹配分数，匹配的金标准数（非0即1），匹配的金标准索引。

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # 计算每个锚点与金标准之间的匹配分数（align_metric）和 IoU（overlaps），并生成正样本掩码（mask_pos）。
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  # 筛选出位于金标准目标内部的锚点的中心点，用于后续的正负样本分配。
        # 上句，输出的mask_in_gts的shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]，
        #     其中如果值为True就表示该锚点的中心落在目标框内。即，确保只有当锚点的中心落在目标边界框内，才有可能被分配为正样本。
        #     然后我转成np的看了一下，应该是金标准附近的若干个格点，都被认为是正例了。
        #     那，其实可以考虑用nms的思路再去掉一部分True，只让真的真正例结果才是True的，
        #         这样，应该至少能把最终的真正例筛选出来，且保持梯度的传递。
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # 上句，计算锚点与金标准目标之间的 对齐度量 和 IoU，进一步确定哪些锚点是有效的正样本。mask_in_gts * mask_gt是要把金标准中的补零给去掉。
        #     输出的align_metric的shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]。其中只有正例锚（即上述有效匹配）的位置，才是有数的，否则就是0。
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())  # self.topk是10
        # 上句，根据给定的准则（就是align_metric）选择前topk个候选框，并生成一个 count_tensor，这个张量指示哪些候选框被选中。具体见函数里。
        #     输出mask_topk的shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]。
        # 【重要理解】里面也是一些非0即1的值，表示每张图像中align_metric排名前10的像素点所在的位置。
        #     然后变成np后，用_mask_topk1=np.sum(_mask_topk[0],axis=1)检查了一下，这个计算的是第0张图片上，
        #     每个金标准对应了多少个入选者，结果发现得到一个9维的向量（9个金标准），
        #     其中前5个数都是10（这5个金标准是真金，每个都对应10个align_metric最大的锚框），而后面都是0，对应金标准补零的部分。
        # 也就是说，这一步可以理解为，保留了更少的正例锚。有点像是rcnn里的套路了。
        mask_pos = mask_topk * mask_in_gts * mask_gt
        # 输出mask_pos是经过了若干次筛选之后，留下的正样本掩码，表示哪些锚点可能是正样本，
        #     shape是[16(批大小), 8400, 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数)]。
        # 所以说这个应该是提示我们在训练的时候遴选预测值，又不导致无法训练的方法。
        return mask_pos, align_metric, overlaps  # mask_pos是最终入选的、与每个金标准的align_metric足够大的、self.topk（10）个锚框。

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]  # 锚数量，8400
        mask_gt = mask_gt.bool()  # shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        # 初始化张量，用于储存每个金标准目标和锚点的 IoU，shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        # 初始化张量，用于储存金标准类别的、预测的类别概率（看后面）。
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # [2, 16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数)]
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        # ind[0][i, j] 表示的是 当前批次中的第 i 张图片中的第 j 个金标准目标所属的图片索引号（也就是i，会发现第0行就全都是0，第1行就全都是1）
        ind[1] = gt_labels.squeeze(-1)
        # ind[1][i, j] 表示第 i 张图片中的第 j 个金标准目标的类别标签。
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
        # 提取每个锚点对应预测结果中，金标准类别的类别概率。shape还是[16, 9, 8400]，其中非零值就是正例锚的、金标准类别的、预测分数。
        # 应该是和mask_in_gts * mask_gt（也就是这个函数里的mask_gt）的非零值位置一样，只不过现在不再是true false的，而是类别概率了。
        # 不过，也发现这里面的类别概率，有的是很低的（可能是因为刚开始训练？）。
        # 【重要】但是这种方法，我感觉是可以用于提取我们想要的距离啊、角度啊之类的。。。
        #     应该也可以两两计算距离，然后用类似于mask_gt的方法把不想要的假正例距离给置零什么的。到时候提示孙老师一下。。。
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        # torch.Size([9479, 4])，其中9479是mask_gt中为true的元素个数，也就是有效匹配数（可以理解为，有多少个锚框被认为是正例了）。
        # 【基础】那个[mask_gt]是一种筛选操作，选择出与有效匹配（即 mask_gt 为 True 的位置）对应的 pd_bboxes，
        #     而将不符合条件的位置（即 mask_gt 为 False 的地方）排除掉。shape的大概变化情况是：
        #     shape[16(批大小), 8400, 4]先unsqeeze变成[16(批大小), 1, 8400, 4]，
        #     然后expand就复制成[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400, 4]的，然后用mask_gt这个布尔掩码来筛选出有效的候选框。
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # 【重要】这一步确实是截断了梯度，但是并不会影响训练过程。也就是说，看来用这种掩码的方式去选择一部分锚框（也就是有效匹配的锚框）是没什么问题的。。
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        # shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]。
        #     其中iou_calculation输出的是一个长度为9479（有效匹配数）的向量，即这些有效匹配的iou，然后把这些iou回填到overlaps里去。
        #     然后验证了一下，这个overlaps和bbox_scores，确实是同样的位置取了非零值。
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        # shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]。其中只有正例锚（即上述有效匹配）的位置，才是有数的，否则就是0。
        #     将预测类别概率和IoU按照各自的权重组合，用于衡量预测与金标准的匹配程度。
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
                    就是外面的align_metric，衡量金标准和预测框匹配度的分数。比如说，如果有9479个正例锚，那么这里面就有9479个非零值。
                    可以用np.count_nonzero(_align_metric)数np矩阵里的非零值个数。
            largest (bool): If True, select the largest values; otherwise, select the smallest values. 如果是True，就选择k个最大值，否则选择k个最小值。
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.
                    一个布尔张量，规定了范围，范围之外的全部清零。在这里应该是对应着，把对应于金标准补零区域的结果全部清零，即使它偶然是最大的k个候选者，仍然清零掉。
        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        # 前topk个候选者的值 和 索引位置，shape都是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 10(要选出来最大的10个)]。
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # topk_mask的shape也和上面一样。
        topk_idxs.masked_fill_(~topk_mask, 0)  # 把那些不符合topk_mask的地方，对应的索引位置清零。

        # 以下，计算 count_tensor，统计哪些候选框被选择
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)  # 一个形状与metrics相同的张量，用于记录哪些锚点被选择。
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # 上面for循环中，topk_idxs[:, :, k : k + 1] 提取了第 k 个候选框的索引，
        #     然后通过 scatter_add_ 将 ones 加到 count_tensor 的对应位置，表示这个候选框被选择。
        count_tensor.masked_fill_(count_tensor > 1, 0)  # 过滤重复的框
        # count_tensor 中可能会有值为 2 或更大的位置，因为同一个位置可能会被多个 topk 值选中。现在要把这些重复选中的框的计数归零（为啥不是置1？不过先不管了），确保每个锚点只被选中一次。
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.
        根据前面计算的匹配关系，为每个正例锚分配目标标签、边界框以及目标分数。
        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects. 金标准标签，shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 1]
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).金标准外接矩形，shape是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 4]
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.  每个正样本锚所匹配的金标准的索引号。
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.
                              fg_mask的shape是[16(批大小), 8400]，其中第i,j个数表示第i张图像上的第j个锚框匹配上了多少个金标准
                              （不过经过select_highest_overlaps之后，要么是0个要么是1个，验证了一下，确实如此）

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                                          每个正样本锚对应的金标准标签，里面第i,j个元素就是第i张图像中第j个锚框对应的金标准标签，如果是正例就是有标签，否则是0
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points. 每个正样本锚对应的金标准外接矩形
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.每个正样本锚对应的一热向量（其实就是上面那个标签得到的，信息是一样的）
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]  # 形状为(16(批大小), 1)的张量，里面的数是0-15，表示批次里的图像。
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # 将每个正例锚匹配的金标准索引号，加上偏移量（batch_ind * self.n_max_boxes）来和展平后的gt_labels对齐。
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        # 提取金标准标签。flatten()让gt_labels变成一维，target_gt_idx提供了每个正样本锚对应的金标准索引号，从而获得每个正样本锚点的类别标签。

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]  # 类似上面，提取每个锚样本的金标准外接矩形。

        # Assigned target scores
        target_labels.clamp_(0)  # 确保金标准标签不会小于0。避免非法的标签值。

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # shape是[16(批大小), 8400, 28(类别数)]，准备存放每个锚框的一热标签。
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)  # 给每个锚框，都构建一热标签。

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # 把fg_mask先扩充一维，然后复制self.num_classes(28)次。好像就是为了下一句用它过滤一遍的。
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        # 上句，根据fg_scores_mask的值来选择目标分数。如果锚是正样本，则保留一热标签；否则，将一热标签里的热的也变凉。
        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]  # 8400
        bs, n_boxes, _ = gt_bboxes.shape  # 16(批大小) 和 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数)
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # 左上角点和右下角点坐标
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # 上句，计算锚点到边界框的相对偏移。每个锚点到目标框的偏移量包括4个值：[dx1, dy1, dx2, dy2]。
        return bbox_deltas.amin(3).gt_(eps)
        # 输出形状是[16(批大小), 9(当前批次数据中、含有金标准个数最多的图像中的金标准个数), 8400]，
        #     其中如果值为True就表示该锚点的中心落在目标框内。这一步骤，amin(3)是可微操作，梯度能够正常通过它回传。
        #     gt_(eps)不是可微操作，会截断梯度，但这在代码中无关紧要。GPT的解释是，因为其结果只用于生成掩码，梯度流不依赖于它。但是给我的感觉是，可能截断了梯度的是那些小于eps的目标，那截断了就截断了吧，反正这些目标也是要抛弃的。

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        一个锚点可能与多个金标准框重合，这个函数是要选择重叠最大、最合适的目标框作为该锚点的匹配目标。这是为了确保每个锚点只与一个目标框匹配，并且总是选择最优匹配。目测就是nms的可反向传播版本啊。。
        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)  这个张量是正例锚掩码，他的每一位表示每个锚点是否与某个目标框有重叠。mask_pos[b, i, j] = 1就表示第b张图像中，第i个锚点与第j个位置的目标框有重叠。
            overlaps (Tensor): shape(b, n_max_boxes, h*w)  锚点与目标框之间的 IoU 重叠程度。overlaps[b, i, j]表示第b张图像中，第i个锚点与第j个位置的目标框之间的IoU。

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)  # shape是[16(批大小), 8400]，每一行的8400个元素应该就代表着，这张图像上的8400个锚，其中的每一个究竟与多少个金标准匹配上了。
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes 一个锚点与多个金标准匹配
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)  是否与多个金标准匹配
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            # 与该锚点的、最大IoU的金标准外接矩形的索引号
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            # 以上，先创建与 mask_pos 形状相同的零张量，然后将最大 IoU 的目标框位置标记为 1。
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # 如果一个锚框匹配了多个金标准，就只留最大的。
            fg_mask = mask_pos.sum(-2)  #
            # 更新fg_mask，即，更新了每个锚框匹配上了多少个金标准（不过感觉这么操作下来，似乎要么是0个要么是1个了？查了一下，也确实是这样）
        target_gt_idx = mask_pos.argmax(-2)  # shanp是(16, 8400)，每一行的8400个数，应该是表示相应的锚点对应的是第几个金标准（Find each grid serve which gt(index)）。
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []  # 初始化锚点列表，分别用于储存每层特征图生成的锚点坐标 和 对应的步长张量。
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):  # for循环表示feats[i]和stride[i]是对应的。
        _, _, h, w = feats[i].shape  # 该特征图的长宽
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        # 以上，生成从 0 到 w-1 和 h-1 的均匀坐标序列，并加上偏移量grid_cell_offset。这个偏移量将锚点放置在每个网格的中心。
        # Use a universal meshgrid call compatible with PyTorch <1.10 (no 'indexing' kwarg)
        sy, sx = torch.meshgrid(sy, sx)  # 将 sx 和 sy 扩展为网格坐标。
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))  # 将 sx 和 sy 堆叠在一起，形成形状为 [h, w, 2] 的张量并展平，形成 [h * w, 2] 的张量，每行表示一个锚点的 (x, y) 坐标。
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))  # 生成形状为 [h * w, 1] 的张量，每一行都对应相应锚点的步长。
    return torch.cat(anchor_points), torch.cat(stride_tensor)  # 输出锚点坐标是[8400, 2]的，步长是[8400, 1]的。8400是所有特征图中的像素点个数。


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):  # 将预测的边界框分量结合锚点坐标，计算边界框的实际坐标。
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt  # 左上角点，即锚点的xy坐标分别减掉l和t
    x2y2 = anchor_points + rb  # 右下角点，即锚点的xy坐标分别加上r和b
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
