import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime



def get_top_predictions(pred_bbox, pred_score, confidence_threshold=0.3, top_k=5):
    """
    获取前k个预测框（置信度最高的k个）

    参数:
    - pred_bbox: 预测框数组
    - pred_score: 置信度数组
    - confidence_threshold: 最低置信度阈值
    - top_k: 要获取的预测框数量

    返回:
    - top_boxes: 前k个预测框
    - top_scores: 对应的置信度
    """
    if len(pred_bbox) == 0 or len(pred_score) == 0:
        return np.array([]), np.array([])

    # 过滤低置信度的预测
    high_conf_indices = pred_score >= confidence_threshold

    if not np.any(high_conf_indices):
        # 如果没有高置信度的预测，返回空数组
        return np.array([]), np.array([])

    # 获取高置信度的框和分数
    high_conf_boxes = pred_bbox[high_conf_indices]
    high_conf_scores = pred_score[high_conf_indices]

    # 按置信度降序排序
    sorted_indices = np.argsort(high_conf_scores)[::-1]

    # 取前k个
    k = min(top_k, len(high_conf_boxes))
    top_boxes = high_conf_boxes[sorted_indices[:k]]
    top_scores = high_conf_scores[sorted_indices[:k]]

    return top_boxes, top_scores


def visualize_top_predictions_original_style(top_k=5):
    """
    显示前k个预测框（置信度最高的k个），保持原始颜色方案

    原始颜色方案：
    - 绿色 (0, 255, 0): 真实框 (GT)
    - 黄色 (0, 255, 255): 预测框 (Pred) 带置信度

    参数:
    - top_k: 要显示的预测框数量
    """
    # 设置路径
    pred_boxlists_path = './output/2025_01-10_21-20-00_test_stft/inference/CVCVideo_val_videos/pred_boxlists.pth'
    gt_boxlists_path = './output/2025_01-10_21-20-00_test_stft/inference/CVCVideo_val_videos/gt_boxlists.pth'
    filenames_path = './output/2025_01-10_21-20-00_test_stft/inference/CVCVideo_val_videos/filename_lists.pth'

    current_time = datetime.datetime.now().strftime("%Y_%m-%d_%H-%M-%S")

    base_output_dir = f'./output/result_visualization/{current_time}/top{top_k}_predictions_original_style'
    base_image_path = '../data/Downstream_Data/KUMC/KUMC/processed/val2019/Image/'

    os.makedirs(base_output_dir, exist_ok=True)

    # 加载数据
    pred_boxlists = torch.load(pred_boxlists_path)
    gt_boxlists = torch.load(gt_boxlists_path)
    filenames = torch.load(filenames_path)

    print(f"加载了 {len(pred_boxlists)} 个预测boxlists")
    print(f"加载了 {len(gt_boxlists)} 个真实boxlists")
    print(f"加载了 {len(filenames)} 个文件名")

    # 置信度阈值
    confidence_threshold = 0.3

    successful = 0

    # 处理每个样本
    for image_id in range(len(filenames)):
        gt_boxlist = gt_boxlists[image_id]
        pred_boxlist = pred_boxlists[image_id]
        filename = filenames[image_id]

        if image_id % 100 == 0:
            print(f"处理进度: {image_id}/{len(filenames)}")

        # 构建完整图像路径
        if not filename.endswith('.jpg'):
            filename = filename + '.jpg'

        full_image_path = os.path.join(base_image_path, filename)

        # 检查文件是否存在
        if not os.path.exists(full_image_path):
            print(f"文件不存在: {full_image_path}")
            continue

        # 读取图像
        original_img = cv2.imread(full_image_path)
        if original_img is None:
            print(f"无法读取图像: {full_image_path}")
            continue

        # 从真实boxlist中获取边界框和标签
        try:
            gt_bbox = gt_boxlist.bbox.numpy()
            gt_label = gt_boxlist.get_field("labels").numpy()
        except AttributeError:
            if hasattr(gt_boxlist.bbox, 'numpy'):
                gt_bbox = gt_boxlist.bbox.numpy()
            else:
                gt_bbox = gt_boxlist.bbox
                if torch.is_tensor(gt_bbox):
                    gt_bbox = gt_bbox.cpu().numpy()

            if hasattr(gt_boxlist, 'get_field'):
                gt_label = gt_boxlist.get_field("labels").numpy()
            elif hasattr(gt_boxlist, 'extra_fields') and 'labels' in gt_boxlist.extra_fields:
                gt_label = gt_boxlist.extra_fields['labels'].numpy()
            else:
                gt_label = np.ones(len(gt_bbox))

        # 从预测boxlist中获取边界框和置信度
        try:
            pred_bbox = pred_boxlist.bbox.numpy()
            pred_score = pred_boxlist.get_field("scores").numpy()
        except AttributeError:
            if hasattr(pred_boxlist.bbox, 'numpy'):
                pred_bbox = pred_boxlist.bbox.numpy()
            else:
                pred_bbox = pred_boxlist.bbox
                if torch.is_tensor(pred_bbox):
                    pred_bbox = pred_bbox.cpu().numpy()

            if hasattr(pred_boxlist, 'get_field'):
                pred_score = pred_boxlist.get_field("scores").numpy()
            elif hasattr(pred_boxlist, 'extra_fields') and 'scores' in pred_boxlist.extra_fields:
                pred_score = pred_boxlist.extra_fields['scores'].numpy()
            else:
                pred_score = np.ones(len(pred_bbox))

        # 获取前k个预测框
        top_boxes, top_scores = get_top_predictions(
            pred_bbox, pred_score, confidence_threshold, top_k
        )

        # 打印问题样本的详细信息
        if image_id == 4253:
            print(f"\n=== 问题样本 4253 详细信息 ===")
            print(f"原始预测框数量: {len(pred_bbox)}")
            print(f"置信度范围: {pred_score.min():.3f} ~ {pred_score.max():.3f}")
            print(f"过滤后保留: {len(top_boxes)} 个预测框")
            if len(top_scores) > 0:
                print(f"前5个预测框置信度: {top_scores[:5]}")

        # 创建对应的输出文件夹结构
        if '/' in filename:
            folder_name, file_name = os.path.split(filename)
            output_folder = os.path.join(base_output_dir, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # 只取文件名，不带路径
            simple_filename = file_name
        else:
            output_folder = base_output_dir
            simple_filename = filename

        # 移除文件扩展名
        base_name = os.path.splitext(simple_filename)[0]

        # 保存带有所有真实框的基础图像
        img_with_gt = original_img.copy()
        if gt_label.sum() > 0:
            for gt_idx in range(int(gt_label.sum())):
                if gt_idx < len(gt_bbox):
                    x1, y1, x2, y2 = map(int, gt_bbox[gt_idx])
                    cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gt_output_path = os.path.join(output_folder, f"{base_name}_ground_truth.jpg")
        cv2.imwrite(gt_output_path, img_with_gt)

        # 为每个预测框创建单独的图像
        if len(top_boxes) > 0:
            for i, (box, score) in enumerate(zip(top_boxes, top_scores)):
                # 创建当前预测框的图像副本
                img_single_box = original_img.copy()

                # 绘制当前预测框（黄色）
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img_single_box, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cv2.putText(img_single_box, f"{score:.2f}",
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)

                # 保存单预测框图像
                output_filename = f"{base_name}_pred_{i:02d}_conf_{score:.2f}_cnt_{successful + 1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, img_single_box)
                successful += 1

                # 显示前5个样本和问题样本4253的前几个预测框
                if (image_id < 5 or image_id == 4253) and i < 3:  # 只显示前3个预测框
                    display_img = cv2.cvtColor(img_single_box, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(display_img)
                    plt.title(f'Sample {image_id}: 预测框 {i + 1}/{len(top_boxes)} (置信度: {score:.2f})')
                    plt.axis('off')
                    plt.show()
        else:
            # 如果没有预测框，也记录一下
            print(f"图像 {filename} 没有预测框，只保存了真实框图像")

        # 可选：保存所有预测框在一张图上的版本（与原始代码相同）
        img_all_preds = original_img.copy()
        if gt_label.sum() > 0:
            for gt_idx in range(int(gt_label.sum())):
                if gt_idx < len(gt_bbox):
                    x1, y1, x2, y2 = map(int, gt_bbox[gt_idx])
                    cv2.rectangle(img_all_preds, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(top_boxes) > 0:
            for i, (box, score) in enumerate(zip(top_boxes, top_scores)):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img_all_preds, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img_all_preds, f"{score:.2f}",
                            (int(x1 + 10), int(y1 + 10 + 15 * i)),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)

        all_preds_output_path = os.path.join(output_folder, f"{base_name}_all_predictions.jpg")
        cv2.imwrite(all_preds_output_path, img_all_preds)

    print(f"处理完成，成功生成了 {successful} 个预测框图像")
    print(f"结果保存在: {base_output_dir}")


def apply_nms(boxes, scores, iou_threshold=0.5, confidence_threshold=0.3, max_boxes=5):
    """
    应用非极大值抑制（NMS）

    参数:
    - boxes: 边界框数组
    - scores: 置信度数组
    - iou_threshold: IoU阈值
    - confidence_threshold: 置信度阈值
    - max_boxes: 最大保留框数

    返回:
    - nms_boxes: NMS后的边界框
    - nms_scores: NMS后的置信度
    """
    if len(boxes) == 0 or len(scores) == 0:
        return np.array([]), np.array([])

    # 1. 过滤低置信度的预测
    high_conf_indices = scores >= confidence_threshold
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]

    if len(boxes) == 0:
        return np.array([]), np.array([])

    # 2. 按置信度降序排序
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]

    # 3. 应用NMS
    keep_indices = []

    while len(boxes) > 0 and len(keep_indices) < max_boxes:
        # 取当前置信度最高的框
        keep_indices.append(0)
        current_box = boxes[0]

        if len(boxes) == 1:
            break

        # 计算当前框与剩余框的IoU
        ious = []
        for i in range(1, len(boxes)):
            iou = calculate_iou(current_box, boxes[i])
            ious.append(iou)

        ious = np.array(ious)

        # 保留IoU小于阈值的框
        keep_mask = ious < iou_threshold
        boxes = boxes[1:][keep_mask]
        scores = scores[1:][keep_mask]

    # 获取原始索引对应的框和分数
    if len(sorted_indices) > 0 and len(keep_indices) > 0:
        # 由于我们一直在操作排序后的数组，需要映射回原始索引
        final_indices = []
        for idx in keep_indices:
            # 找到原始索引
            if idx < len(sorted_indices):
                final_indices.append(sorted_indices[idx])

        # 获取最终的框和分数
        final_indices = np.array(final_indices)
        nms_boxes = boxes[:len(final_indices)] if len(boxes) > 0 else np.array([])
        nms_scores = scores[:len(final_indices)] if len(scores) > 0 else np.array([])

        return nms_boxes, nms_scores

    return np.array([]), np.array([])


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    if union_area == 0:
        return 0.0

    return inter_area / union_area


if __name__ == '__main__':
    visualize_top_predictions_original_style(top_k=200)