#!/usr/bin/env python3

import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import re
import unicodedata

from gliner import GLiNER
from gliner.data_processing.collator import DataCollator


def to_safe_filename(filename: str, replacement: str = '_', max_length: int = 255) -> str:
    """
    Convert a string to a safe filename by removing or replacing characters that
    are illegal in most file systems.
    """
    # Normalize Unicode characters to ASCII (e.g. é -> e)
    normalized = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Replace invalid characters for Windows and many other file systems
    safe = re.sub(r'[\\/*?:"<>|]', replacement, normalized)
    # Replace any whitespace with the replacement character
    safe = re.sub(r'\s+', replacement, safe)
    # Strip leading/trailing spaces, dots, and replacement characters
    safe = safe.strip(" ."+replacement)
    # Truncate if needed
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe if safe else "untitled"


def get_unique_entity_types(data_path: str) -> List[str]:
    """Extract all unique entity types from dataset and sort them alphabetically."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unique_types = set()
    for item in data:
        for _, _, label in item['ner']:
            unique_types.add(label)
    return sorted(list(unique_types))


class NEREvaluator:
    """Core evaluation logic for NER tasks."""
    def __init__(self, dataset: List[Dict], true_entities: List, pred_entities: List,
                 span_based: bool = False):
        self.dataset = dataset
        self.true_entities = true_entities
        self.pred_entities = pred_entities
        self.span_based = span_based
        self.raw_comparisons = []
        
    def _process_ground_truth(self, tokens, entities: List) -> List:
        """Format ground truth entities."""
        processed = []
        for start, end, _, label in entities:
            if self.span_based:
                processed.append((start, end, label))
            else:
                entity_text = ' '.join(tokens[start:end+1])
                processed.append((entity_text, label))
        return processed

    def _process_predictions(self, tokens: List[str], entities: List) -> List:
        """Format predicted entities."""
        processed = []
        for start, end, label, _ in entities:
            if self.span_based:
                processed.append((start, end, label))
            else:
                entity_text = ' '.join(tokens[start:end+1])
                processed.append((entity_text, label))
        return processed

    def _align_data(self) -> tuple:
        """Align ground truth and predictions for comparison."""
        aligned_true, aligned_pred = [], []
        for idx, (true, pred) in enumerate(zip(self.true_entities, self.pred_entities)):
            tokens = self.dataset[idx]['tokenized_text']
            aligned_true.append(self._process_ground_truth(tokens, true))
            aligned_pred.append(self._process_predictions(tokens, pred))
        return aligned_true, aligned_pred

    def _calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 with complete TP/FP/FN tracking."""
        true, pred = self._align_data()
        metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        global_counts = {'tp': 0, 'fp': 0, 'fn': 0}
        precision_scores, recall_scores, f1_scores = [], [], []

        if self.span_based:
            # Relaxed span evaluation: a prediction is correct if its span overlaps with a ground truth span of the same label.
            for t_entities, p_entities in zip(true, pred):
                self.raw_comparisons.append({
                    'ground_truth': list(t_entities),
                    'predictions': list(p_entities)
                })
                matched_true = [False] * len(t_entities)
                for p in p_entities:
                    p_start, p_end, p_label = p
                    found = False
                    for i, t in enumerate(t_entities):
                        t_start, t_end, t_label = t
                        if not matched_true[i] and p_label == t_label:
                            # Check for overlap: spans overlap if p_start <= t_end and t_start <= p_end.
                            if p_start <= t_end and t_start <= p_end:
                                matched_true[i] = True
                                found = True
                                metrics[p_label]['tp'] += 1
                                global_counts['tp'] += 1
                                break
                    if not found:
                        metrics[p_label]['fp'] += 1
                        global_counts['fp'] += 1
                for t, matched in zip(t_entities, matched_true):
                    if not matched:
                        t_label = t[2]
                        metrics[t_label]['fn'] += 1
                        global_counts['fn'] += 1
        else:
            for t_entities, p_entities in zip(true, pred):
                true_set = set(t_entities)
                pred_set = set(p_entities)
                self.raw_comparisons.append({
                    'ground_truth': list(true_set),
                    'predictions': list(pred_set)
                })
                for item in pred_set:
                    key = item[1]
                    if item in true_set:
                        metrics[key]['tp'] += 1
                        global_counts['tp'] += 1
                    else:
                        metrics[key]['fp'] += 1
                        global_counts['fp'] += 1
                for item in true_set:
                    key = item[1]
                    if item not in pred_set:
                        metrics[key]['fn'] += 1
                        global_counts['fn'] += 1

        # Calculate per-class metrics
        results = {}
        for label, counts in metrics.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            results[label] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # Calculate micro-averaged metrics
        micro_tp = global_counts['tp']
        micro_fp = global_counts['fp']
        micro_fn = global_counts['fn']
        
        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = (2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

        # Calculate macro-averaged metrics
        macro_precision = sum(precision_scores)/len(precision_scores) if precision_scores else 0.0
        macro_recall = sum(recall_scores)/len(recall_scores) if recall_scores else 0.0
        macro_f1 = sum(f1_scores)/len(f1_scores) if f1_scores else 0.0

        return {
            'per_class': results,
            'micro': {
                'tp': micro_tp,
                'fp': micro_fp,
                'fn': micro_fn,
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1,
                'support': micro_tp + micro_fn
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
                'support': len(f1_scores)
            }
        }

    def evaluate(self) -> Dict:
        """Run the full evaluation pipeline."""
        metrics = self._calculate_metrics()
        correct = sum(1 for comp in self.raw_comparisons 
                      if set(comp['predictions']) == set(comp['ground_truth']))
        accuracy = correct / len(self.raw_comparisons) if self.raw_comparisons else 0
        
        with open('raw_results.json', 'w') as f:
            json.dump(self.raw_comparisons, f, indent=2)
        
        return {
            'entity_metrics': metrics['per_class'],
            'micro_metrics': metrics['micro'],
            'macro_metrics': metrics['macro'],
            'accuracy': round(accuracy, 4),
            'total_samples': len(self.raw_comparisons)
        }


class GLiNEREvaluationPipeline:
    """End-to-end evaluation pipeline for GLiNER models."""
    def __init__(self, model_path: str, data_path: Union[str, Path], device: str = 'cuda:0'):
        self.model = GLiNER.from_pretrained(model_path, max_length=1024).to(device)
        self.test_data = self._load_and_preprocess_data(data_path)
        
    def _load_and_preprocess_data(self, data_path: Union[str, Path]) -> List[Dict]:
        """Load and format evaluation data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        processed = []
        for item in raw_data:
            tokens = item['tokenized_text']
            entities = [
                [start, end, ' '.join(tokens[start:end+1]), str(label)]
                for start, end, label in item['ner']
            ]
            processed.append({
                'tokenized_text': tokens,
                'ner': entities
            })
        return processed

    def run_evaluation(
        self,
        threshold: float = 0.5,
        batch_size: int = 8,
        flat_ner: bool = False,
        multi_label: bool = False,
        entity_types: Optional[List[str]] = None,
        span_evaluation: bool = False
    ) -> Dict:
        """Execute model evaluation with the specified parameters."""
        self.model.eval()
        collator = DataCollator(
            self.model.config,
            data_processor=self.model.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=entity_types
        )
        loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False
        )
        predictions, ground_truth = [], []
        for batch in tqdm(loader, desc="Evaluating batches"):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**batch)[0]
            if not isinstance(outputs, torch.Tensor):
                outputs = torch.from_numpy(outputs)
                
            decoded = self.model.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                outputs,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label
            )
            predictions.extend(decoded)
            ground_truth.extend(batch["entities"])

        evaluator = NEREvaluator(
            dataset=self.test_data,
            true_entities=ground_truth,
            pred_entities=predictions,
            span_based=span_evaluation
        )
        return evaluator.evaluate()


def main(args):
    args.entity_types = get_unique_entity_types(args.data_path)
    
    pipeline = GLiNEREvaluationPipeline(
        model_path=args.model_path,
        data_path=args.data_path,
        device=args.device
    )
    
    results = pipeline.run_evaluation(
        threshold=args.threshold,
        batch_size=args.batch_size,
        flat_ner=args.flat_ner,
        multi_label=args.multi_label,
        entity_types=args.entity_types,
        span_evaluation=args.span_evaluation
    )
    
    print("\nEvaluation Results:")
    print(f"Sample Accuracy: {results['accuracy']}")
    print(f"Total Samples: {results['total_samples']}")
    
    print("\nMicro-averaged Metrics:")
    print(f"Precision: {results['micro_metrics']['precision']}")
    print(f"Recall: {results['micro_metrics']['recall']}")
    print(f"F1-score: {results['micro_metrics']['f1']}")
    print(f"Support: {results['micro_metrics']['support']}")
    
    print("\nMacro-averaged Metrics:")
    print(f"Precision: {results['macro_metrics']['precision']}")
    print(f"Recall: {results['macro_metrics']['recall']}")
    print(f"F1-score: {results['macro_metrics']['f1']}")
    print(f"Supported Classes: {results['macro_metrics']['support']}")
    
    print("\nPer-class Metrics:")
    for entity, metrics in results['entity_metrics'].items():
        print(f"\nEntity: {entity}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    if args.output_file:
        output_file = args.output_file
    else:
        default_name = f"evaluation_results_{args.model_path}.json"
        output_file = to_safe_filename(default_name)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPer-entity metrics saved in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLiNER Evaluation Script")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path or identifier for the GLiNER model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test JSON file")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for entity extraction")
    
    parser.add_argument("--output_file", type=str, default=None, help="File to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (only batch size 1 is supported)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device for evaluation (e.g., 'cuda:0' or 'cpu')")
    
    # Only relaxed span evaluation is supported; default is relaxed span evaluation.
    parser.add_argument("--span_evaluation", dest="span_evaluation", action="store_true",
                        help="Enable relaxed span-based evaluation (default)")
    parser.set_defaults(span_evaluation=True)
    
    parser.add_argument("--flat_ner", dest="flat_ner", action="store_true", help="Enable flat NER decoding")
    parser.set_defaults(flat_ner=False)
    
    parser.add_argument("--multi_label", dest="multi_label", action="store_true", help="Enable multi-label NER")
    parser.set_defaults(multi_label=False)
    
    args = parser.parse_args()
    main(args)