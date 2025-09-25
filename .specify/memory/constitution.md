# PyOxynet Constitution

## Core Principles

### I. Scientific Accuracy (NON-NEGOTIABLE)
All CPET analysis features MUST maintain scientific validity; Model outputs must be validated against established exercise physiology principles; No feature may compromise the accuracy of exercise domain classifications or physiological interpretations

### II. TensorFlow Lite First
New inference features MUST use TFLite models for production deployment; Full TensorFlow only for training and development; Models must be optimized for performance and memory efficiency; Support offline operation

### III. Test-First Development (NON-NEGOTIABLE) 
TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced; All scientific calculations must have unit tests with known-good reference data

### IV. Data Privacy & Security
CPET data is medical information requiring protection; No data persistence beyond processing session; Secure file handling with validation; Clear data cleanup policies; User consent for any data usage

### V. Simplicity & Reusability
Build on existing pyoxynet capabilities; Avoid unnecessary architectural complexity; Reuse proven patterns from existing Flask app; Maintain backward compatibility with existing models

## [SECTION_2_NAME]
<!-- Example: Additional Constraints, Security Requirements, Performance Standards, etc. -->

[SECTION_2_CONTENT]
<!-- Example: Technology stack requirements, compliance standards, deployment policies, etc. -->

## [SECTION_3_NAME]
<!-- Example: Development Workflow, Review Process, Quality Gates, etc. -->

[SECTION_3_CONTENT]
<!-- Example: Code review requirements, testing gates, deployment approval process, etc. -->

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

[GOVERNANCE_RULES]
<!-- Example: All PRs/reviews must verify compliance; Complexity must be justified; Use [GUIDANCE_FILE] for runtime development guidance -->

**Version**: [CONSTITUTION_VERSION] | **Ratified**: [RATIFICATION_DATE] | **Last Amended**: [LAST_AMENDED_DATE]
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->