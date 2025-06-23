# Annotation Guidelines for Legal Document Anonymization

This document provides guidelines for annotating sensitive entities in Brazilian legal documents for anonymization purposes.

**Total entities defined:** 24

## Entity Definitions

### 1. PERSON

**Description:** Full names, first names, and surnames of individuals mentioned in legal documents

**Example:** JOÃO PEDRO ALMEIDA SOUZA <PERSON> vem respeitosamente perante Vossa Excelência

### 2. CPF

**Description:** Brazilian taxpayer registry numbers (CPF) in various formats

**Example:** portador do CPF nº 123.456.789-10 <CPF>

### 3. RG

**Description:** Brazilian identity document numbers with issuing authority

**Example:** RG nº 11.222.333-4 SSP/PR <RG>

### 4. CNPJ

**Description:** Brazilian company registry numbers for legal entities

**Example:** inscrita no CNPJ sob nº 12.345.678/0001-90 <CNPJ>

### 5. OAB

**Description:** Brazilian Bar Association registration numbers for lawyers

**Example:** Dr. CARLOS EDUARDO LIMA FERREIRA OAB/PR nº 12.345 <OAB>

### 6. PHONE

**Description:** Phone numbers in Brazilian format

**Example:** Telefone: (41) 3222-1234 <PHONE>

### 7. EMAIL

**Description:** Email addresses of individuals or organizations

**Example:** contato@escritorio.com.br <EMAIL>

### 8. ADDRESS

**Description:** Complete addresses including street, number, and complement

**Example:** residente na Rua das Palmeiras, nº 123, apto. 401 <ADDRESS>

### 9. STREET

**Description:** Street names and numbers when mentioned separately

**Example:** situada na Rua das Flores, nº 890 <STREET>

### 10. NEIGHBORHOOD

**Description:** District or neighborhood names

**Example:** bairro Água Verde <NEIGHBORHOOD>

### 11. CITY

**Description:** City names, typically in legal jurisdiction context

**Example:** Curitiba/PR <CITY>

### 12. STATE

**Description:** Brazilian state names or abbreviations

**Example:** Estado do Paraná <STATE>

### 13. CEP

**Description:** Brazilian postal codes (CEP)

**Example:** CEP 80240-000 <CEP>

### 14. BANK_ACCOUNT

**Description:** Bank account numbers with agency information

**Example:** conta bancária nº 12345-6, agência nº 0001 <BANK_ACCOUNT>

### 15. CREDIT_CARD

**Description:** Credit card numbers (partially masked or full)

**Example:** cartão de crédito VISA nº 4111 1111 1111 1111 <CREDIT_CARD>

### 16. FINANCIAL_VALUE

**Description:** Monetary amounts mentioned in legal contexts

**Example:** valor de R$ 2.500,00 (dois mil e quinhentos reais) <FINANCIAL_VALUE>

### 17. PROCESS_NUMBER

**Description:** Legal process numbers and case identifiers

**Example:** Processo nº 0005678-90.2015.8.16.0001 <PROCESS_NUMBER>

### 18. COMPANY

**Description:** Company, organization, and business names

**Example:** ECOPROJETOS AMBIENTAIS LTDA. <COMPANY>

### 19. INSTITUTION

**Description:** Government institutions, banks, and public organizations

**Example:** INSTITUTO NACIONAL DO SEGURO SOCIAL – INSS <INSTITUTION>

### 20. LICENSE_PLATE

**Description:** Vehicle license plate numbers

**Example:** Veículo Toyota Corolla, placa XYZ-1234 <LICENSE_PLATE>

### 21. DATE

**Description:** Specific dates in various formats

**Example:** contraíram matrimônio em 10 de março de 2012 <DATE>

### 22. AGE

**Description:** Age information of individuals

**Example:** o requerido conta com 25 anos de idade <AGE>

### 23. DOCUMENT_NUMBER

**Description:** Various document numbers like registration, matricula, etc.

**Example:** registrado sob matrícula nº 56789 <DOCUMENT_NUMBER>

### 24. BENEFIT_NUMBER

**Description:** Social security and benefit numbers

**Example:** número de benefício 123.456.789-0 <BENEFIT_NUMBER>

## Annotation Rules

1. Annotate entities using the format: `<ENTITY_LABEL>text</ENTITY_LABEL>`
2. Be consistent with entity boundaries - include complete names/numbers
3. When in doubt, prefer over-annotation to under-annotation
4. For composite entities (e.g., full addresses), use the most specific label
5. Always verify entity type against the definitions above
