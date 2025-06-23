#!/usr/bin/env python3
"""
Entity Definitions and Annotation Guidelines for Legal Document Anonymization
"""

from typing import List, NamedTuple


class EntityDefinition(NamedTuple):
    """Structure for entity definitions."""

    label: str
    description: str
    example: str


# Define 20+ entities for anonymization based on legal documents
ENTITY_DEFINITIONS: List[EntityDefinition] = [
    # Personal Identifiers
    EntityDefinition(
        label="PERSON",
        description="Full names, first names, and surnames of individuals mentioned in legal documents",
        example="JOÃO PEDRO ALMEIDA SOUZA <PERSON> vem respeitosamente perante Vossa Excelência",
    ),
    EntityDefinition(
        label="CPF",
        description="Brazilian taxpayer registry numbers (CPF) in various formats",
        example="portador do CPF nº 123.456.789-10 <CPF>",
    ),
    EntityDefinition(
        label="RG",
        description="Brazilian identity document numbers with issuing authority",
        example="RG nº 11.222.333-4 SSP/PR <RG>",
    ),
    EntityDefinition(
        label="CNPJ",
        description="Brazilian company registry numbers for legal entities",
        example="inscrita no CNPJ sob nº 12.345.678/0001-90 <CNPJ>",
    ),
    EntityDefinition(
        label="OAB",
        description="Brazilian Bar Association registration numbers for lawyers",
        example="Dr. CARLOS EDUARDO LIMA FERREIRA OAB/PR nº 12.345 <OAB>",
    ),
    # Contact Information
    EntityDefinition(
        label="PHONE",
        description="Phone numbers in Brazilian format",
        example="Telefone: (41) 3222-1234 <PHONE>",
    ),
    EntityDefinition(
        label="EMAIL",
        description="Email addresses of individuals or organizations",
        example="contato@escritorio.com.br <EMAIL>",
    ),
    # Location Data
    EntityDefinition(
        label="ADDRESS",
        description="Complete addresses including street, number, and complement",
        example="residente na Rua das Palmeiras, nº 123, apto. 401 <ADDRESS>",
    ),
    EntityDefinition(
        label="STREET",
        description="Street names and numbers when mentioned separately",
        example="situada na Rua das Flores, nº 890 <STREET>",
    ),
    EntityDefinition(
        label="NEIGHBORHOOD",
        description="District or neighborhood names",
        example="bairro Água Verde <NEIGHBORHOOD>",
    ),
    EntityDefinition(
        label="CITY",
        description="City names, typically in legal jurisdiction context",
        example="Curitiba/PR <CITY>",
    ),
    EntityDefinition(
        label="STATE",
        description="Brazilian state names or abbreviations",
        example="Estado do Paraná <STATE>",
    ),
    EntityDefinition(
        label="CEP", description="Brazilian postal codes (CEP)", example="CEP 80240-000 <CEP>"
    ),
    # Financial Information
    EntityDefinition(
        label="BANK_ACCOUNT",
        description="Bank account numbers with agency information",
        example="conta bancária nº 12345-6, agência nº 0001 <BANK_ACCOUNT>",
    ),
    EntityDefinition(
        label="CREDIT_CARD",
        description="Credit card numbers (partially masked or full)",
        example="cartão de crédito VISA nº 4111 1111 1111 1111 <CREDIT_CARD>",
    ),
    EntityDefinition(
        label="FINANCIAL_VALUE",
        description="Monetary amounts mentioned in legal contexts",
        example="valor de R$ 2.500,00 (dois mil e quinhentos reais) <FINANCIAL_VALUE>",
    ),
    # Legal and Institutional Data
    EntityDefinition(
        label="PROCESS_NUMBER",
        description="Legal process numbers and case identifiers",
        example="Processo nº 0005678-90.2015.8.16.0001 <PROCESS_NUMBER>",
    ),
    EntityDefinition(
        label="COMPANY",
        description="Company, organization, and business names",
        example="ECOPROJETOS AMBIENTAIS LTDA. <COMPANY>",
    ),
    EntityDefinition(
        label="INSTITUTION",
        description="Government institutions, banks, and public organizations",
        example="INSTITUTO NACIONAL DO SEGURO SOCIAL – INSS <INSTITUTION>",
    ),
    EntityDefinition(
        label="LICENSE_PLATE",
        description="Vehicle license plate numbers",
        example="Veículo Toyota Corolla, placa XYZ-1234 <LICENSE_PLATE>",
    ),
    # Temporal Data
    EntityDefinition(
        label="DATE",
        description="Specific dates in various formats",
        example="contraíram matrimônio em 10 de março de 2012 <DATE>",
    ),
    EntityDefinition(
        label="AGE",
        description="Age information of individuals",
        example="o requerido conta com 25 anos de idade <AGE>",
    ),
    # Document References
    EntityDefinition(
        label="DOCUMENT_NUMBER",
        description="Various document numbers like registration, matricula, etc.",
        example="registrado sob matrícula nº 56789 <DOCUMENT_NUMBER>",
    ),
    EntityDefinition(
        label="BENEFIT_NUMBER",
        description="Social security and benefit numbers",
        example="número de benefício 123.456.789-0 <BENEFIT_NUMBER>",
    ),
]


def get_entity_labels() -> List[str]:
    """Return list of all entity labels."""
    return [entity.label for entity in ENTITY_DEFINITIONS]


def get_entity_by_label(label: str) -> EntityDefinition | None:
    """Get entity definition by label."""
    for entity in ENTITY_DEFINITIONS:
        if entity.label == label:
            return entity
    return None


def print_annotation_guidelines():
    """Print the complete annotation guidelines."""
    print("=" * 80)
    print("ANNOTATION GUIDELINES FOR LEGAL DOCUMENT ANONYMIZATION")
    print("=" * 80)
    print()
    print("This document provides guidelines for annotating sensitive entities in")
    print("Brazilian legal documents for anonymization purposes.")
    print()
    print(f"Total entities defined: {len(ENTITY_DEFINITIONS)}")
    print()

    for i, entity in enumerate(ENTITY_DEFINITIONS, 1):
        print(f"{i}. ENTITY: {entity.label}")
        print(f"   DESCRIPTION: {entity.description}")
        print(f"   EXAMPLE: {entity.example}")
        print()

    print("=" * 80)
    print("ANNOTATION RULES:")
    print("=" * 80)
    print("1. Annotate entities using the format: <ENTITY_LABEL>text</ENTITY_LABEL>")
    print("2. Be consistent with entity boundaries - include complete names/numbers")
    print("3. When in doubt, prefer over-annotation to under-annotation")
    print("4. For composite entities (e.g., full addresses), use the most specific label")
    print("5. Always verify entity type against the definitions above")
    print("=" * 80)


def export_guidelines_to_markdown():
    """Export annotation guidelines to markdown file."""
    with open("annotation_guidelines.md", "w", encoding="utf-8") as f:
        f.write("# Annotation Guidelines for Legal Document Anonymization\n\n")
        f.write("This document provides guidelines for annotating sensitive entities in ")
        f.write("Brazilian legal documents for anonymization purposes.\n\n")
        f.write(f"**Total entities defined:** {len(ENTITY_DEFINITIONS)}\n\n")

        f.write("## Entity Definitions\n\n")
        for i, entity in enumerate(ENTITY_DEFINITIONS, 1):
            f.write(f"### {i}. {entity.label}\n\n")
            f.write(f"**Description:** {entity.description}\n\n")
            f.write(f"**Example:** {entity.example}\n\n")

        f.write("## Annotation Rules\n\n")
        f.write("1. Annotate entities using the format: `<ENTITY_LABEL>text</ENTITY_LABEL>`\n")
        f.write("2. Be consistent with entity boundaries - include complete names/numbers\n")
        f.write("3. When in doubt, prefer over-annotation to under-annotation\n")
        f.write("4. For composite entities (e.g., full addresses), use the most specific label\n")
        f.write("5. Always verify entity type against the definitions above\n")

    print("Annotation guidelines exported to annotation_guidelines.md")


if __name__ == "__main__":
    print_annotation_guidelines()
    export_guidelines_to_markdown()
