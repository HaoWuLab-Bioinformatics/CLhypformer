from sklearn.metrics import average_precision_score, accuracy_score, f1_score
import torch


def acc_f1(output, labels, average='binary'):
    from sklearn.metrics import accuracy_score, f1_score

    # === [核心修复] 判断 input 是 logits 还是已经处理过的 preds ===
    # 如果 output 是一维张量，说明已经是预测类别索引
    if output.dim() == 1:
        preds = output.type_as(labels)
    else:
        # 如果是二维张量（logits），需要取 argmax
        preds = output.max(1)[1].type_as(labels)
    
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()

    # === 【修復點：嚴格轉換為 Numpy 數組】 ===
    preds = preds.numpy()
    labels = labels.numpy()
    # ========================================

    accuracy = accuracy_score(preds, labels)
    # 增加 zero_division=0 防止某個 Batch 缺乏特定類別時觸發警告
    f1 = f1_score(preds, labels, average=average, zero_division=0)
    
    # === [新增诊断] 检测 acc 和 f1 是否异常相同 ===
    if abs(accuracy - f1) < 1e-6 and average != 'binary':
        print(f"\n[!] 警告: Accuracy ({accuracy:.4f}) 和 F1 ({f1:.4f}) 几乎相同")
        print(f"    这可能表明:")
        print(f"    1. 使用了 average='{average}' (micro-average 等同于 accuracy)")
        print(f"    2. 或者数据集中某些类别占主导地位")
        print(f"    建议改用 average='weighted' 或 'macro'\n")

    return accuracy, f1


class MarginLoss(torch.nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, preds):
        correct_preds = preds[..., 0:1]
        return torch.nn.functional.relu(self.margin - correct_preds + preds).mean()
