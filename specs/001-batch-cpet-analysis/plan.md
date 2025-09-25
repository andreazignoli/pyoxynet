# Implementation Plan: Batch CPET Analysis

**Branch**: `001-batch-cpet-analysis` | **Date**: 2025-09-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-batch-cpet-analysis/spec.md`

## Summary
Enable researchers to process multiple CPET files simultaneously through a web-based batch processing system that validates data, performs exercise domain analysis, and generates comprehensive reports.

## Technical Context
**Language/Version**: Python 3.11  
**Primary Dependencies**: Flask, pandas, numpy, pyoxynet  
**Storage**: File system for temporary storage, SQLite for job tracking  
**Testing**: pytest with coverage reporting  
**Target Platform**: Linux server (containerized deployment)
**Project Type**: web - Flask backend + simple frontend  
**Performance Goals**: Process 100+ CPET files in under 5 minutes  
**Constraints**: Memory usage <2GB per batch, support files up to 10MB each  
**Scale/Scope**: Support up to 1000 files per batch, 50 concurrent users

## Constitution Check
*GATE: Must pass before Phase 0 research.*

Based on constitution principles:
- ✅ Builds on existing pyoxynet capabilities
- ✅ Maintains simple architecture
- ⚠️ Adds web interface complexity - justified for user accessibility
- ✅ Reuses existing TFLite models

## Project Structure

### Documentation (this feature)
```
specs/001-batch-cpet-analysis/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/tasks command)
```

### Source Code (repository root)
```
# Option 2: Web application
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/
```

**Structure Decision**: Web application structure due to Flask backend + frontend interface

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context**:
   - Research Flask best practices for file upload handling
   - Investigate batch processing patterns for scientific data
   - Study memory optimization for large dataset processing

2. **Generate and dispatch research agents**:
   ```
   Task: "Research Flask file upload best practices for batch processing"
   Task: "Find memory optimization patterns for pandas batch processing"
   Task: "Research progress tracking for long-running web tasks"
   ```

3. **Consolidate findings** in `research.md`

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts

1. **Extract entities from feature spec** → `data-model.md`:
   - BatchJob: id, status, created_at, file_count
   - CPETFile: filename, validation_status, analysis_results
   - ProcessingResult: domain_probabilities, efficiency_metrics, errors

2. **Generate API contracts** from functional requirements:
   - POST /batch/upload - File upload endpoint
   - GET /batch/{id}/status - Job status tracking  
   - GET /batch/{id}/results - Download results
   - POST /batch/{id}/cancel - Cancel processing

3. **Generate contract tests** from contracts:
   - Test file upload validation
   - Test status tracking responses
   - Test result download formats

4. **Update agent file**: Run update script for Claude context

**Output**: data-model.md, contracts/, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach

**Task Generation Strategy**:
- Contract tests for each API endpoint [P]
- Model creation for BatchJob, CPETFile entities [P]
- Service layer for batch processing logic
- Integration tests for complete user workflows
- Frontend components for file upload and progress tracking

**Ordering Strategy**:
- Models and validation logic first
- API endpoints and tests
- Service layer for batch processing
- Frontend integration last
- All independent file tasks marked [P]

**Estimated Output**: 28-32 numbered, ordered tasks in tasks.md

## Complexity Tracking
*Fill ONLY if Constitution Check has violations*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Web interface addition | User accessibility for batch uploads | CLI-only would limit adoption by researchers |

## Progress Tracking

**Phase Status**:
- [ ] Phase 0: Research complete
- [ ] Phase 1: Design complete  
- [ ] Phase 2: Task planning complete
- [ ] Phase 3: Tasks generated
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (with justified web complexity)
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented