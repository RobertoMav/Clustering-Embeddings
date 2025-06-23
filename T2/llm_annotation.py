#!/usr/bin/env python3
"""
LLM Annotation Script for NER Assignment
Performs entity annotation using OpenAI GPT models with zero-shot and few-shot prompts.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

# Import our entity definitions
from entity_definitions import get_entity_labels


class LLMAnnotator:
    """Class for performing LLM-based entity annotation."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM annotator."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: No OpenAI API key provided. Using mock responses for demonstration.")
            self.use_mock = True
        else:
            self.use_mock = False
            try:
                import openai

                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print("OpenAI library not installed. Using mock responses.")
                self.use_mock = True

    def create_zero_shot_prompt(self, text: str) -> str:
        """Create zero-shot prompt for entity extraction."""
        entity_list = ", ".join(get_entity_labels())

        prompt = f"""You are an expert in Brazilian legal document analysis and anonymization. 
Extract and classify named entities for anonymization from the following legal text.

Identify the following entity types: {entity_list}

Instructions:
- Extract entities that need to be anonymized for privacy protection
- Use the format: entity_text -> ENTITY_TYPE
- Be precise with entity boundaries
- Only extract entities that clearly match the specified types
- Focus on sensitive information that should be anonymized

Text to analyze:
{text[:1000]}{"..." if len(text) > 1000 else ""}

Extracted entities:"""

        return prompt

    def create_few_shot_prompt(self, text: str) -> str:
        """Create few-shot prompt with examples for entity extraction."""
        examples = """Examples of entity extraction:

Text: "JOÃO PEDRO ALMEIDA SOUZA, brasileiro, casado, engenheiro civil, CPF nº 123.456.789-10"
Entities:
- JOÃO PEDRO ALMEIDA SOUZA -> PERSON
- 123.456.789-10 -> CPF

Text: "residente na Rua das Palmeiras, nº 123, apto. 401, bairro Água Verde, Curitiba/PR, CEP 80240-000"
Entities:
- Rua das Palmeiras, nº 123, apto. 401 -> ADDRESS
- Água Verde -> NEIGHBORHOOD
- Curitiba/PR -> CITY
- 80240-000 -> CEP

Text: "conta bancária nº 12345-6, agência nº 0001, Banco do Brasil"
Entities:
- 12345-6, agência nº 0001 -> BANK_ACCOUNT
- Banco do Brasil -> INSTITUTION

Text: "Telefone: (41) 3222-1234"
Entities:
- (41) 3222-1234 -> PHONE

Text: "cartão de crédito VISA nº 4111 1111 1111 1111, validade 12/2028"
Entities:
- 4111 1111 1111 1111 -> CREDIT_CARD
- 12/2028 -> DATE"""

        entity_list = ", ".join(get_entity_labels())

        prompt = f"""You are an expert in Brazilian legal document analysis and anonymization.
Extract and classify named entities for anonymization following the examples below.

{examples}

Entity types to identify: {entity_list}

Now extract entities from this text:
{text[:1000]}{"..." if len(text) > 1000 else ""}

Extracted entities:"""

        return prompt

    def mock_llm_response(self, text: str, is_few_shot: bool = False) -> str:
        """Generate mock response for testing when no API key is available."""
        # Simple regex-based entity extraction for demonstration
        entities = []

        # Extract some common patterns
        cpf_pattern = r"\d{3}\.\d{3}\.\d{3}-\d{2}"
        rg_pattern = r"\d{2}\.\d{3}\.\d{3}-\d"
        phone_pattern = r"\(\d{2}\)\s*\d{4}-\d{4}"
        cep_pattern = r"\d{5}-\d{3}"

        # Find CPFs
        for match in re.finditer(cpf_pattern, text):
            entities.append(f"- {match.group()} -> CPF")

        # Find RGs
        for match in re.finditer(rg_pattern, text):
            entities.append(f"- {match.group()} -> RG")

        # Find phones
        for match in re.finditer(phone_pattern, text):
            entities.append(f"- {match.group()} -> PHONE")

        # Find CEPs
        for match in re.finditer(cep_pattern, text):
            entities.append(f"- {match.group()} -> CEP")

        # Extract some names (simple heuristic)
        name_pattern = r"[A-Z][A-Z\s]{10,50}(?=,|\s+brasileir[ao])"
        for match in re.finditer(name_pattern, text):
            name = match.group().strip()
            if len(name.split()) >= 2:  # At least two words
                entities.append(f"- {name} -> PERSON")

        return "\n".join(entities) if entities else "No entities found."

    def call_llm(self, prompt: str) -> str:
        """Call the LLM API or return mock response."""
        if self.use_mock:
            return "Mock response - entities would be extracted here"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using cheaper model for demonstration
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in legal document analysis and entity extraction for anonymization purposes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"

    def annotate_text(self, text: str, use_few_shot: bool = False) -> Dict:
        """Annotate a single text with entities."""
        if use_few_shot:
            prompt = self.create_few_shot_prompt(text)
            prompt_type = "few-shot"
        else:
            prompt = self.create_zero_shot_prompt(text)
            prompt_type = "zero-shot"

        # Call LLM or use mock
        if self.use_mock:
            response = self.mock_llm_response(text, use_few_shot)
        else:
            response = self.call_llm(prompt)

        return {
            "text": text,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "response": response,
            "entities": self.parse_llm_response(response),
        }

    def parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract entities."""
        entities = []

        # Parse entities in format "- text -> ENTITY_TYPE"
        pattern = r"-\s*(.+?)\s*->\s*([A-Z_]+)"

        for match in re.finditer(pattern, response):
            entity_text = match.group(1).strip()
            entity_type = match.group(2).strip()

            entities.append({"text": entity_text, "type": entity_type})

        return entities

    def annotate_corpus_sample(self, texts: List[str], sample_size: int = 10) -> Dict:
        """Annotate a sample of the corpus with both zero-shot and few-shot."""
        sample_texts = texts[:sample_size]
        results = {"zero_shot": [], "few_shot": []}

        print(f"Annotating {len(sample_texts)} texts with both zero-shot and few-shot...")

        for i, text in enumerate(sample_texts):
            print(f"Processing text {i + 1}/{len(sample_texts)}...")

            # Zero-shot annotation
            zero_shot_result = self.annotate_text(text, use_few_shot=False)
            results["zero_shot"].append(zero_shot_result)

            # Few-shot annotation
            few_shot_result = self.annotate_text(text, use_few_shot=True)
            results["few_shot"].append(few_shot_result)

            # Small delay to avoid rate limiting
            if not self.use_mock:
                time.sleep(1)

        return results


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
    """Main function to run LLM annotation."""
    print("Starting LLM annotation process...")

    # Initialize annotator
    annotator = LLMAnnotator()

    # Read corpus
    texts = read_corpus_files()
    print(f"Loaded {len(texts)} texts from corpus.")

    # Annotate sample with both methods
    results = annotator.annotate_corpus_sample(texts, sample_size=5)

    # Save results
    with open("llm_annotation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("LLM annotation completed. Results saved to llm_annotation_results.json")

    # Print summary
    print("\nAnnotation Summary:")
    print(f"Zero-shot annotations: {len(results['zero_shot'])}")
    print(f"Few-shot annotations: {len(results['few_shot'])}")

    # Show sample entities
    if results["zero_shot"]:
        print(f"\nSample Zero-shot entities from first text:")
        for entity in results["zero_shot"][0]["entities"][:5]:
            print(f"  - {entity['text']} -> {entity['type']}")

    if results["few_shot"]:
        print(f"\nSample Few-shot entities from first text:")
        for entity in results["few_shot"][0]["entities"][:5]:
            print(f"  - {entity['text']} -> {entity['type']}")


if __name__ == "__main__":
    main()
