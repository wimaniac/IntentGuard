# Báo cáo IntentGuard

Báo cáo này được sinh từ artifact và cấu hình hiện hành; test gốc chỉ được đọc ở bước này.

## In-domain

| Backend | Accuracy | Macro-F1 | Top-3 | NLL sau calibration | ECE sau calibration | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8982 | 0.8939 | 0.9697 | 0.3955 | 0.0222 | 0.387913 |
| deep | 0.8894 | 0.8832 | 0.9617 | 0.4697 | 0.0210 | 0.506853 |

## OOD

OOD ngân hàng dùng 100 FAQ có URL do người dùng xác nhận. Codex AI đã rà từng nhãn với 77 intent và thu gọn hai câu nhiều vế; nhãn này chưa phải bộ nhãn vàng do hai người gán nhãn độc lập. Các cặp gần trùng ngữ nghĩa được giữ cùng split.

```json
{
  "massive_validation": {
    "baseline": {
      "auroc": 0.9750506419752666,
      "auprc": 0.9658767421343052,
      "ood_unknown_recall": 0.8854003139717426,
      "in_domain_false_rejection_rate": 0.04983748645720477
    },
    "deep": {
      "auroc": 0.94615707771566,
      "auprc": 0.9211641432290661,
      "ood_unknown_recall": 0.6907378335949764,
      "in_domain_false_rejection_rate": 0.04983748645720477
    }
  },
  "massive_test": {
    "baseline": {
      "auroc": 0.981358542288886,
      "auprc": 0.9626958753246636,
      "ood_unknown_recall": 0.8998923573735199,
      "in_domain_false_rejection_rate": 0.03784693019343986
    },
    "deep": {
      "auroc": 0.9479341035197962,
      "auprc": 0.88494393377666,
      "ood_unknown_recall": 0.7179763186221744,
      "in_domain_false_rejection_rate": 0.04163162321278385
    }
  },
  "bank_faq_validation": {
    "baseline": {
      "auroc": 0.9425352112676056,
      "auprc": 0.5325384691806775,
      "ood_unknown_recall": 0.74,
      "in_domain_false_rejection_rate": 0.04983748645720477
    },
    "deep": {
      "auroc": 0.9478439869989166,
      "auprc": 0.46532222059857375,
      "ood_unknown_recall": 0.68,
      "in_domain_false_rejection_rate": 0.04983748645720477
    }
  },
  "bank_faq_test": {
    "baseline": {
      "auroc": 0.9603195962994112,
      "auprc": 0.5592964068060026,
      "ood_unknown_recall": 0.74,
      "in_domain_false_rejection_rate": 0.03784693019343986
    },
    "deep": {
      "auroc": 0.9546425567703953,
      "auprc": 0.37475855456465856,
      "ood_unknown_recall": 0.64,
      "in_domain_false_rejection_rate": 0.04163162321278385
    }
  }
}
```
