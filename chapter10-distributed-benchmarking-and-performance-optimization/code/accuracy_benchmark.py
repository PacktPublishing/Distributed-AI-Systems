"""
Accuracy and Quality Benchmarking for distributed AI systems.

Performance benchmarking measures speed; accuracy benchmarking measures correctness.
Both are essential. Optimizations like quantization, distributed training, or
different serving engines can introduce accuracy regressions.
"""
from evaluate import load
from scipy import stats


def evaluate_glue_sst2(predictions, references):
    """
    Evaluate on GLUE SST-2 (sentiment classification).
    
    The predictions array contains model outputs (0 or 1 for SST-2),
    and references contains ground truth labels.
    """
    glue = load("glue", "sst2")
    results = glue.compute(predictions=predictions, references=references)
    print(f"Accuracy: {results['accuracy']:.3f}")
    return results


def compare_centralized_vs_distributed_accuracy(
    model_centralized,
    model_distributed,
    test_dataset,
    evaluate_model_fn
):
    """
    Compare accuracy between centralized and distributed training.
    
    A critical check: does your distributed training produce the same model
    quality as single-GPU training? Bugs in gradient synchronization, different
    batch size effects, or numerical precision issues can cause distributed
    training to converge to worse solutions.
    """
    acc_centralized = evaluate_model_fn(model_centralized, test_dataset)
    acc_distributed = evaluate_model_fn(model_distributed, test_dataset)
    
    accuracy_drop = acc_centralized - acc_distributed
    
    print(f"Centralized accuracy: {acc_centralized:.4f}")
    print(f"Distributed accuracy: {acc_distributed:.4f}")
    print(f"Accuracy drop: {accuracy_drop:.4f}")
    
    if accuracy_drop > 0.01:  # More than 1% drop
        print("⚠️ Warning: Significant accuracy drop detected!")
    
    return {
        'centralized': acc_centralized,
        'distributed': acc_distributed,
        'drop': accuracy_drop
    }


def evaluate_quantization_impact(model_fp32, model_int8, test_dataset, evaluate_model_fn):
    """
    Compare accuracy between FP32 and INT8 quantized models.
    
    Quantization trades precision for speed. Before deploying a quantized
    model, measure the accuracy cost to ensure the speedup is worth the
    quality tradeoff.
    """
    acc_fp32 = evaluate_model_fn(model_fp32, test_dataset)
    acc_int8 = evaluate_model_fn(model_int8, test_dataset)
    
    accuracy_drop = acc_fp32 - acc_int8
    
    return {
        'fp32_accuracy': acc_fp32,
        'int8_accuracy': acc_int8,
        'drop': accuracy_drop,
        'relative_drop': accuracy_drop / acc_fp32 * 100
    }


def test_statistical_significance(model1_scores, model2_scores, alpha=0.05):
    """
    Test if accuracy difference between two models is statistically significant.
    
    Small accuracy differences may be noise, not signal. Use statistical tests
    to determine if differences are meaningful.
    
    A 0.5% accuracy drop with p=0.3 is probably noise.
    A 0.5% drop with p=0.001 is real.
    """
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    is_significant = p_value < alpha
    
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if is_significant:
        print(f"Statistically significant difference (p < {alpha})")
    else:
        print(f"No statistically significant difference (p >= {alpha})")
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant': is_significant
    }
