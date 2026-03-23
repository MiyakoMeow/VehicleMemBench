import argparse
import json


def fmt_pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_num(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def get_metric(metric, key, default=0):
    return metric.get(key, default)


def print_overall(metric):
    print("--- Overall Metrics ---")
    print(f"Exact Match Rate:     {fmt_pct(get_metric(metric, 'exact_match_rate'))}")
    print()
    print("  [Field-Level] (Whether the correct fields changed)")
    print(f"  Acc Positive:       {fmt_pct(get_metric(metric, 'state_acc_positive'))}  (TP / fields that should change)")
    print(f"  Prec Positive:      {fmt_pct(get_metric(metric, 'state_precision_positive'))}  (TP / changed fields)")
    print(f"  F1 Positive:        {fmt_pct(get_metric(metric, 'state_f1_positive'))}")
    print("  ---")
    print(f"  Acc Negative:       {fmt_pct(get_metric(metric, 'state_acc_negative'))}  (unchanged / fields that should stay unchanged)")
    print(f"  F1 Negative:        {fmt_pct(get_metric(metric, 'state_f1_negative'))}")
    print()
    print("  [Value-Level] (Whether the correct value/trend was applied)")
    print(f"  Change Accuracy:    {fmt_pct(get_metric(metric, 'change_accuracy'))}  (correct values / fields that should change)")
    print(f"  Prec Change:        {fmt_pct(get_metric(metric, 'state_precision_change'))}  (correct values / changed fields)")
    print(f"  F1 Change:          {fmt_pct(get_metric(metric, 'state_f1_change'))}")
    print()
    print("  [Efficiency]")
    print(f"  Avg Pred Calls:     {fmt_num(get_metric(metric, 'avg_pred_calls'))}")
    print(f"  Avg Output Token:   {fmt_num(get_metric(metric, 'avg_output_token'))}")


def print_by_reasoning_type(metric):
    by_type = metric.get("by_reasoning_type", {})
    if not by_type:
        return

    print("\n--- Metrics by Reasoning Type ---")
    for rtype in sorted(by_type.keys()):
        rmetrics = by_type[rtype]
        print(f"\n[{rtype}] (n={rmetrics.get('count', 0)})")
        print(f"  Exact Match:        {fmt_pct(rmetrics.get('exact_match_rate', 0))}")
        print(
            "  [Field] Acc/Prec/F1:  "
            f"{fmt_pct(rmetrics.get('state_acc_positive', 0))} / "
            f"{fmt_pct(rmetrics.get('state_precision_positive', 0))} / "
            f"{fmt_pct(rmetrics.get('state_f1_positive', 0))}"
        )
        print(
            "  [Value] Acc/Prec/F1:  "
            f"{fmt_pct(rmetrics.get('change_accuracy', 0))} / "
            f"{fmt_pct(rmetrics.get('state_precision_change', 0))} / "
            f"{fmt_pct(rmetrics.get('state_f1_change', 0))}"
        )
        print(
            "  [Negative] Acc/F1:    "
            f"{fmt_pct(rmetrics.get('state_acc_negative', 0))} / "
            f"{fmt_pct(rmetrics.get('state_f1_negative', 0))}"
        )


def main():
    parser = argparse.ArgumentParser(description="Format and print VehicleAgentBench metric.json")
    parser.add_argument("metric_path", type=str, help="Path to metric.json")
    parser.add_argument("--with_by_type", action="store_true", help="Also print by_reasoning_type metrics")
    args = parser.parse_args()

    with open(args.metric_path, "r", encoding="utf-8") as f:
        metric = json.load(f)

    print_overall(metric)
    if args.with_by_type:
        print_by_reasoning_type(metric)


if __name__ == "__main__":
    main()
