def progressive_loss(pred_boxes, gt_boxes, epoch, total_epochs):
    """
    渐进式损失：前期以CIoU为主，后期逐步引入Shape-IoU
    """
    # 动态权重调整
    progress = epoch / total_epochs
    lambda1 = 1.0 - 0.3 * progress  # CIoU权重从1.0降到0.7
    lambda2 = 0.3 * progress        # Shape-IoU权重从0增到0.3
    
    ciou_loss = calculate_ciou_loss(pred_boxes, gt_boxes)
    shape_iou_loss = calculate_shape_iou_loss(pred_boxes, gt_boxes)
    
    return lambda1 * ciou_loss + lambda2 * shape_iou_loss