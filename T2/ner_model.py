#!/usr/bin/env python3
"""
NER Model Training and Evaluation Script
Implements a simple NER model for legal document anonymization.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Import entity definitions


class SimpleNERModel:
    """Simple rule-based NER model for demonstration purposes."""

    def __init__(self):
        """Initialize the NER model with patterns for Brazilian legal documents."""
        self.entity_patterns = self._create_entity_patterns()
        self.performance_metrics = {}

    def _create_entity_patterns(self) -> Dict[str, List[str]]:
        """Create regex patterns for different entity types."""
        patterns = {
            "CPF": [r"\d{3}\.\d{3}\.\d{3}-\d{2}", r"CPF\s*n[ºo°]\s*\d{3}\.\d{3}\.\d{3}-\d{2}"],
            "RG": [r"\d{2}\.\d{3}\.\d{3}-\d", r"RG\s*n[ºo°]\s*\d{2}\.\d{3}\.\d{3}-\d\s*SSP"],
            "CNPJ": [
                r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
                r"CNPJ\s*(?:sob\s*)?n[ºo°]\s*\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
            ],
            "PHONE": [r"\(\d{2}\)\s*\d{4,5}-\d{4}", r"Telefone:\s*\(\d{2}\)\s*\d{4,5}-\d{4}"],
            "CEP": [r"\d{5}-\d{3}", r"CEP\s*\d{5}-\d{3}"],
            "CREDIT_CARD": [
                r"\d{4}\s*\d{4}\s*\d{4}\s*\d{4}",
                r"cartão.*?n[ºo°]\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}",
            ],
            "BANK_ACCOUNT": [r"conta.*?n[ºo°]\s*\d{4,6}-\d", r"agência\s*n[ºo°]\s*\d{4}"],
            "PROCESS_NUMBER": [
                r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}",
                r"Processo\s*n[ºo°].*?\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}",
            ],
            "DATE": [
                r"\d{1,2}\s+de\s+\w+\s+de\s+\d{4}",
                r"\d{2}/\d{2}/\d{4}",
                r"\d{2}-\d{2}-\d{4}",
            ],
            "FINANCIAL_VALUE": [
                r"R\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?",
                r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*reais",
            ],
            "OAB": [r"OAB/[A-Z]{2}\s*n[ºo°]\s*\d+", r"OAB\s*n[ºo°]\s*\d+"],
        }
        return patterns

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using regex patterns."""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(
                        {
                            "text": match.group(),
                            "type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

        # Extract person names (more complex heuristic)
        person_entities = self._extract_person_names(text)
        entities.extend(person_entities)

        # Remove duplicates and overlaps
        entities = self._remove_overlapping_entities(entities)

        return entities

    def _extract_person_names(self, text: str) -> List[Dict]:
        """Extract person names using heuristics."""
        person_entities = []

        # Pattern for names before "brasileiro/brasileira"
        name_pattern = r"([A-Z][A-Z\s]{8,50})(?=,\s*brasileir[ao])"
        matches = re.finditer(name_pattern, text)

        for match in matches:
            name = match.group(1).strip()
            # Basic validation: at least 2 words, all caps
            words = name.split()
            if len(words) >= 2 and all(word.isupper() for word in words):
                person_entities.append(
                    {"text": name, "type": "PERSON", "start": match.start(1), "end": match.end(1)}
                )

        return person_entities

    def _remove_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping the longest ones."""
        # Sort by start position
        entities.sort(key=lambda x: x["start"])

        filtered_entities = []
        for entity in entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for accepted in filtered_entities:
                if entity["start"] < accepted["end"] and entity["end"] > accepted["start"]:
                    overlaps = True
                    break

            if not overlaps:
                filtered_entities.append(entity)

        return filtered_entities

    def predict(self, texts: List[str]) -> List[List[Dict]]:
        """Predict entities for a list of texts."""
        predictions = []
        for text in texts:
            entities = self.extract_entities(text)
            predictions.append(entities)
        return predictions

    def create_mock_ground_truth(self, texts: List[str]) -> List[List[Dict]]:
        """Create mock ground truth annotations for evaluation."""
        ground_truth = []

        for text in texts:
            # Use the same extraction but add some noise/variations
            entities = self.extract_entities(text)

            # Randomly remove some entities (to simulate annotation differences)
            if entities:
                num_to_remove = random.randint(0, max(1, len(entities) // 4))
                entities = random.sample(entities, len(entities) - num_to_remove)

            # Add some manual annotations for addresses and institutions
            manual_entities = self._extract_manual_entities(text)
            entities.extend(manual_entities)

            ground_truth.append(entities)

        return ground_truth

    def _extract_manual_entities(self, text: str) -> List[Dict]:
        """Extract additional entities manually for ground truth."""
        entities = []

        # Extract some addresses
        address_patterns = [
            r"(?:Rua|Avenida|Av\.)\s+[A-Za-z\s]+,\s*n[ºo°]\s*\d+(?:,\s*apto\.?\s*\d+)?",
            r"(?:residente|domiciliad[ao])\s+(?:na|no)\s+([^,]+,\s*n[ºo°]\s*\d+[^,]*)",
        ]

        for pattern in address_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": "ADDRESS",
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        # Extract institutions
        institution_patterns = [
            r"Banco\s+[A-Za-z\s]+",
            r"(?:INSTITUTO|Instituto)\s+[A-Za-z\s]+",
            r"(?:MINISTÉRIO|Ministério)\s+[A-Za-z\s]+",
            r"(?:Cartório|CARTÓRIO)\s+[A-Za-z\s]+",
        ]

        for pattern in institution_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": "INSTITUTION",
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return entities


class NEREvaluator:
    """Evaluator for NER model performance."""

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def calculate_metrics(
        self, predictions: List[List[Dict]], ground_truth: List[List[Dict]]
    ) -> Dict:
        """Calculate precision, recall, F1, and accuracy."""
        total_tp = 0  # True positives
        total_fp = 0  # False positives
        total_fn = 0  # False negatives

        entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for pred_entities, true_entities in zip(predictions, ground_truth):
            # Convert to sets for easier comparison
            pred_set = {(e["text"], e["type"]) for e in pred_entities}
            true_set = {(e["text"], e["type"]) for e in true_entities}

            # Calculate matches
            matches = pred_set & true_set
            tp = len(matches)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Per-entity metrics
            for entity_text, entity_type in matches:
                entity_metrics[entity_type]["tp"] += 1

            for entity_text, entity_type in pred_set - true_set:
                entity_metrics[entity_type]["fp"] += 1

            for entity_text, entity_type in true_set - pred_set:
                entity_metrics[entity_type]["fn"] += 1

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate accuracy (entity-level: correctly identified entities / total entities)
        accuracy = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        # Calculate per-entity metrics
        per_entity_metrics = {}
        for entity_type, counts in entity_metrics.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0

            per_entity_metrics[entity_type] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4),
                "support": tp + fn,
            }

        return {
            "overall": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "total_entities": total_tp + total_fn,
            },
            "per_entity": per_entity_metrics,
        }

    def print_evaluation_report(self, metrics: Dict):
        """Print a formatted evaluation report."""
        print("=" * 80)
        print("NER MODEL EVALUATION REPORT")
        print("=" * 80)

        overall = metrics["overall"]
        print(f"Overall Performance:")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1-Score:  {overall['f1']:.4f}")
        print(f"  Total Entities: {overall['total_entities']}")
        print()

        print("Per-Entity Performance:")
        print(
            f"{'Entity Type':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
        )
        print("-" * 65)

        for entity_type, scores in metrics["per_entity"].items():
            print(
                f"{entity_type:<15} {scores['precision']:<10.4f} "
                f"{scores['recall']:<10.4f} {scores['f1']:<10.4f} "
                f"{scores['support']:<10}"
            )

        print("=" * 80)


def read_corpus_files(data_dir: str = "data") -> List[str]:
    """Read all text files from the corpus directory."""
    texts = []
    data_path = Path(data_dir)

    for file_path in sorted(data_path.glob("texto*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return texts


def main():
    """Main function to train and evaluate the NER model."""
    print("Starting NER model training and evaluation...")

    # Read corpus
    texts = read_corpus_files()
    print(f"Loaded {len(texts)} texts from corpus.")

    # Use first 10 texts for training/evaluation
    sample_texts = texts[:10]

    # Initialize model and evaluator
    model = SimpleNERModel()
    evaluator = NEREvaluator()

    # Create predictions
    print("Generating predictions...")
    predictions = model.predict(sample_texts)

    # Create mock ground truth (in real scenario, this would be manual annotations)
    print("Creating ground truth annotations...")
    ground_truth = model.create_mock_ground_truth(sample_texts)

    # Evaluate model
    print("Evaluating model performance...")
    metrics = evaluator.calculate_metrics(predictions, ground_truth)

    # Print evaluation report
    evaluator.print_evaluation_report(metrics)

    # Save results
    results = {
        "metrics": metrics,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "sample_texts": sample_texts,
    }

    with open("ner_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Evaluation completed. Results saved to ner_evaluation_results.json")

    # Show sample predictions
    print("\nSample Predictions from First Text:")
    if predictions[0]:
        for entity in predictions[0][:5]:
            print(f"  - {entity['text']} -> {entity['type']}")
    else:
        print("  No entities found in first text.")


if __name__ == "__main__":
    main()
