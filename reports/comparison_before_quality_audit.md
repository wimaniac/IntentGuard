# Báo cáo IntentGuard

Báo cáo này được sinh từ artifact và cấu hình hiện hành; test gốc chỉ được đọc ở bước này.

## In-domain

| Backend | Accuracy | Macro-F1 | Top-3 | NLL sau calibration | ECE sau calibration | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8982 | 0.8939 | 0.9697 | 0.3955 | 0.0222 | 0.387913 |
| deep | 0.7250 | 0.7030 | 0.8982 | 0.9577 | 0.0229 | 0.289402 |

## OOD

OOD ngân hàng mới gồm 94 nhãn DeepSeek và 6 nhãn duyệt thủ công. Metric trên tập này là kết quả với nhãn tự động, chưa phải đánh giá trên nhãn vàng độc lập.

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
      "auroc": 0.7908465161212415,
      "auprc": 0.741627223309331,
      "ood_unknown_recall": 0.37676609105180536,
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
      "auroc": 0.790379791070098,
      "auprc": 0.6361935794254642,
      "ood_unknown_recall": 0.38428417653390745,
      "in_domain_false_rejection_rate": 0.049201009251471826
    }
  },
  "bank_validation": {
    "baseline": {
      "auroc": 0.9779631635969664,
      "auprc": 0.7910513798296678,
      "ood_unknown_recall": 0.88,
      "in_domain_false_rejection_rate": 0.04983748645720477
    },
    "deep": {
      "auroc": 0.8057204767063922,
      "auprc": 0.22305467441392246,
      "ood_unknown_recall": 0.32,
      "in_domain_false_rejection_rate": 0.04983748645720477
    }
  },
  "bank_test": {
    "baseline": {
      "auroc": 0.9755929352396973,
      "auprc": 0.7538576241216719,
      "ood_unknown_recall": 0.82,
      "in_domain_false_rejection_rate": 0.03784693019343986
    },
    "deep": {
      "auroc": 0.737031118587048,
      "auprc": 0.08070249475339147,
      "ood_unknown_recall": 0.28,
      "in_domain_false_rejection_rate": 0.049201009251471826
    }
  }
}
```
