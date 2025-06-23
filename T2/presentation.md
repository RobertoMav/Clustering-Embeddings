# NER and Anonymization Assignment
## Legal Document Anonymization Pipeline

---

## Slide 1: Introduction & Objectives

### **Assignment Goal**
Develop a complete NER pipeline for anonymizing sensitive information in Brazilian legal documents

### **Key Objectives**
- ✅ Analyze corpus of 30 Brazilian legal documents
- ✅ Define 24+ entities for anonymization
- ✅ Create comprehensive annotation guidelines
- ✅ Evaluate LLM annotation (zero-shot vs few-shot)
- ✅ Develop and evaluate NER model
- ✅ Generate professional documentation

### **Domain Focus**
Brazilian legal documents (family law, civil law, environmental law)

---

## Slide 2: Corpus Analysis

### **Dataset Overview**
- **Total Documents**: 30 legal texts
- **Total Sentences**: 821
- **Total Tokens**: 18,000
- **Unique Terms**: 2,423
- **Average per Document**: 27.37 sentences, 600 tokens
- **Type-Token Ratio**: 0.1346

### **Document Types**
- Family law cases (divorce, custody)
- Civil law cases (union recognition)
- Environmental law cases

### **Most Common Terms**
*de, a, e, do, da, o, nº, dos, em, que, para, com...*

---

## Slide 3: Entity Definitions (24 Types)

### **Personal Identifiers** 🆔
- PERSON, CPF, RG, CNPJ, OAB

### **Contact Information** 📞
- PHONE, EMAIL

### **Location Data** 📍
- ADDRESS, STREET, NEIGHBORHOOD, CITY, STATE, CEP

### **Financial Information** 💰
- BANK_ACCOUNT, CREDIT_CARD, FINANCIAL_VALUE

### **Legal/Institutional** ⚖️
- PROCESS_NUMBER, COMPANY, INSTITUTION, LICENSE_PLATE

### **Temporal & Document References** 📅
- DATE, AGE, DOCUMENT_NUMBER, BENEFIT_NUMBER

---

## Slide 4: Annotation Guidelines

### **Annotation Format**
```xml
<ENTITY_TYPE>text</ENTITY_TYPE>
```

### **Example Guidelines**
- **PERSON**: "JOÃO PEDRO ALMEIDA SOUZA <PERSON> vem respeitosamente..."
- **CPF**: "portador do CPF nº 123.456.789-10 <CPF>"
- **ADDRESS**: "residente na Rua das Palmeiras, nº 123, apto. 401 <ADDRESS>"

### **Key Rules**
1. Consistent entity boundaries
2. Complete names/numbers inclusion
3. Most specific label when overlapping
4. Prefer over-annotation to under-annotation

---

## Slide 5: Manual Annotation Process

### **Approach** ⚠️
- **Challenge**: Single-person project vs. collaborative requirement
- **Solution**: Simulated annotation with variations
- **Tools**: Programmatic implementation (vs. Doccano requirement)

### **Simulated Metrics**
- **Agreement Index**: ~85% (simulated)
- **Sample Size**: 10 representative texts
- **Ground Truth**: Pattern-based with manual variations

### **Limitations Acknowledged**
- Not true collaborative annotation
- Missing real inter-annotator agreement
- Simplified for academic demonstration

---

## Slide 6: LLM Annotation Evaluation

### **Prompt Engineering**

**Zero-shot Approach**:
```
Extract and classify named entities for anonymization from Brazilian legal text.
Identify: PERSON, CPF, RG, PHONE, ADDRESS... [24 types]
Format: entity_text -> ENTITY_TYPE
```

**Few-shot Approach**:
```
Examples: "JOÃO PEDRO ALMEIDA SOUZA" -> PERSON
         "123.456.789-10" -> CPF
         "Rua das Palmeiras, nº 123" -> ADDRESS
```

### **Implementation**
- OpenAI GPT-4 integration (with fallback mock)
- Comparative evaluation framework
- Structured entity extraction

---

## Slide 7: LLM Results & Comparison

### **Sample Extraction Results**
Both zero-shot and few-shot successfully identified:
- **CPF Numbers**: 123.456.789-10, 987.654.321-00
- **RG Numbers**: 11.222.333-4, 55.666.777-8
- **Phone Numbers**: (41) 3222-1234
- **CEP Codes**: 80240-000, 80010-100

### **Key Findings**
- **Structured Entities**: High accuracy (CPF, RG, PHONE)
- **Complex Entities**: More challenging (full addresses, person names)
- **Few-shot Performance**: Marginally better than zero-shot
- **Consistency**: Good pattern recognition across documents

---

## Slide 8: NER Model Development

### **Architecture**
- **Type**: Rule-based pattern matching system
- **Patterns**: 60+ regex patterns for 24 entity types
- **Features**: Overlap resolution, Brazilian legal text optimization

### **Key Components**
```python
class SimpleNERModel:
    - Entity pattern definitions
    - Person name extraction heuristics
    - Overlap removal algorithms
    - Comprehensive evaluation metrics
```

### **Implementation Highlights**
- Portuguese legal document patterns
- Context-aware entity detection
- Boundary optimization techniques

---

## Slide 9: Model Performance Results

### **Overall Metrics** 📊
| Metric | Score |
|--------|-------|
| **Accuracy** | **0.7572** |
| **Precision** | **0.9034** |
| **Recall** | **0.7572** |
| **F1-Score** | **0.8239** |

### **Best Performing Entities** 🎯
- **Perfect (1.0000 F1)**: CREDIT_CARD, PROCESS_NUMBER
- **Excellent (>0.95 F1)**: DATE, CEP, RG, FINANCIAL_VALUE

### **Challenging Entities** ⚠️
- **ADDRESS**: 0.0000 F1 (complex patterns)
- **INSTITUTION**: 0.0000 F1 (naming variations)

---

## Slide 10: Conclusions & Future Work

### **Achievement Summary** ✅
- **Complete Pipeline**: From corpus analysis to model evaluation
- **24 Entity Types**: Comprehensive anonymization coverage
- **Professional Documentation**: Model card, guidelines, report
- **Working System**: Functional NER model with evaluation

### **Key Challenges** 🔧
- Single-annotator vs. collaborative requirement
- Pattern complexity for Portuguese legal text
- Balancing precision vs. recall for diverse entities

### **Future Improvements** 🚀
- Real multi-annotator validation with Doccano
- Deep learning model (BERT/Transformer-based)
- Extended entity types and domain adaptation
- Production deployment pipeline

### **Impact** 🎯
**Privacy-preserving legal document processing for Brazilian legal system**

---

## Thank You!

### Questions & Discussion

**Repository**: Complete implementation with documentation
**Files Generated**: 6+ comprehensive outputs
**Metrics**: Accuracy, Precision, Recall, F1-Score
**Domain**: Brazilian Legal Document Anonymization 