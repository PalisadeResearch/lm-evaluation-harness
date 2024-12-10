from typing import List, Union, Tuple
import numpy as np
import wandb


try:
    import tinyBenchmarks as tb
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`tinyBenchmarks` is required for tinyBenchmarks task metric calculation, install via \
`pip install git+https://github.com/felipemaiapolo/tinyBenchmarks`"
    )


def agg_pirt(items: List[float], benchmark: str) -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["pirt"]


def agg_gpirt_arc(items: List[float], benchmark: str = "arc") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_gsm8k(items: List[float], benchmark: str = "gsm8k") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_hellaswag(items: List[float], benchmark: str = "hellaswag") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_mmlu(items: List[float], benchmark: str = "mmlu", model_name: str = "gpt-4o") -> float:
    if 'gpt-4o' in model_name:
        def compare_answers(pred: str, target: str) -> int:
        # Extract first letter and normalize both prediction and target
            try:
                pred_letter = pred[0].strip().upper() if pred else ''
                target_letter = target[0].strip().upper() if target else ''
                
                # Debug print
                print(f"Comparing: pred='{pred_letter}' vs target='{target_letter}'")
                
                # Return 1 for correct, 0 for incorrect
                return 1 if pred_letter == target_letter else 0
            except Exception as e:
                print(f"Error comparing answers: {e}")
                return 0

        try:
            # Debug prints
            print("\nProcessing batch of answers:")
            print("First few items:", items[:5])
            
            # Calculate accuracy for each item
            scores = []
            for pred, target in items:
                score = compare_answers(pred, target)
                scores.append(score)
                
            # Calculate mean accuracy
            accuracy = sum(scores) / len(scores) if scores else 0.0
            print(f"\nFinal accuracy: {accuracy}")
            
            return accuracy
                
        except Exception as e:
            print(f"Error in aggregation: {e}")
            return 0.0
    
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_truthfulqa(items: List[float], benchmark: str = "truthfulqa") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_winogrande(items: List[float], benchmark: str = "winogrande") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]
