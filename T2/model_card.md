# Legal Document Anonymization NER Model Card

## Model Description

This model is designed for Named Entity Recognition (NER) in Brazilian legal documents for anonymization purposes. It identifies sensitive information that should be anonymized to protect privacy.

**Architecture**: Rule-based pattern matching system
**Language**: Portuguese (Brazilian legal documents)
**Domain**: Family law and civil law documents

## Intended Use

**Primary Use Cases**:
- Anonymization of legal documents for privacy protection
- Identification of sensitive entities in Brazilian legal texts
- Pre-processing for legal document analysis

**Primary Users**:
- Legal professionals
- Document processing systems
- Privacy compliance tools

## Entity Types

The model identifies 20+ entity types including:
- Personal identifiers (PERSON, CPF, RG)
- Contact information (PHONE, EMAIL)
- Location data (ADDRESS, CITY, STATE, CEP)
- Financial information (BANK_ACCOUNT, CREDIT_CARD, FINANCIAL_VALUE)
- Legal references (PROCESS_NUMBER, OAB)
- Temporal data (DATE, AGE)

## Performance Metrics

Based on evaluation with mock annotations:
- **Accuracy**: ~0.76-0.85 (entity-level accuracy)
- **Precision**: ~0.85-0.95 (varies by entity type)
- **Recall**: ~0.75-0.90 (varies by entity type)
- **F1-Score**: ~0.80-0.92 (varies by entity type)

*Note: These metrics are based on synthetic evaluation data and should be validated with real annotated data.*

## Training Data and Methodology

**Training Approach**: Rule-based pattern matching using regex patterns
**Data**: 30 Brazilian legal documents (family law and civil law cases)
**Annotation**: Simulated annotations for demonstration purposes

## Limitations and Biases

1. **Pattern-based limitations**: May miss entities that don't match predefined patterns
2. **Domain specificity**: Optimized for legal documents, may not generalize to other domains
3. **Language limitations**: Designed specifically for Portuguese legal terminology
4. **False positives**: May identify non-sensitive text that matches patterns
5. **Evaluation limitations**: Current metrics based on synthetic data

## Ethical Considerations

**Privacy**: This model is designed to protect privacy by identifying sensitive information
**Bias**: May have biases based on the training document patterns
**Responsibility**: Users should validate results before using for actual anonymization

## Usage Instructions

```python
from ner_model import SimpleNERModel

# Initialize model
model = SimpleNERModel()

# Extract entities from text
entities = model.extract_entities(text)

# Process results
for entity in entities:
    print(f"{entity['text']} -> {entity['type']}")
```

## Model Versioning

**Version**: 1.0.0
**Last Updated**: December 2024
**Status**: Demonstration/Prototype

## Contact Information

This model was developed as part of an academic assignment for NER and anonymization in legal documents.

## References

- Brazilian legal document patterns and terminology
- Privacy protection requirements for legal data
- NER evaluation methodologies

---

*This model card follows the format recommended for responsible AI model documentation.*
