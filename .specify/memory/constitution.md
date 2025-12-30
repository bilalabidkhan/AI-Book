<!-- SYNC IMPACT REPORT:
Version change: N/A → 1.0.0
Modified principles: N/A (new constitution)
Added sections: All sections (new constitution)
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ Updated to reflect new principles
  - .specify/templates/spec-template.md ✅ Updated to reflect new standards
  - .specify/templates/tasks-template.md ✅ Updated to reflect new constraints
Follow-up TODOs: None
-->
# Spec-driven technical book with embedded RAG chatbot Constitution

## Core Principles

### Spec-first development
Spec-first development: All features and functionality must be defined in specifications before implementation begins; Specifications must be complete and approved before coding starts; Clear requirements and acceptance criteria required for every feature

### Verified technical accuracy
Verified technical accuracy: All technical content must be fact-checked and verified; Code examples must be runnable and tested; Technical claims must be supported by evidence or authoritative sources

### Clear, professional writing
Clear, professional writing: Content must be written at Flesch-Kincaid grade 10-12 level; Professional tone and terminology required throughout; Clear explanations with appropriate context for target audience

### Reproducible workflows
Reproducible workflows: All development processes must be documented and reproducible; Build and deployment processes must be automated and version-controlled; Development environment setup must be clearly documented

### Grounded AI responses only
Grounded AI responses only: The RAG chatbot must only respond based on book content; Out-of-scope queries must be rejected; Responses must be traceable to specific content in the book

### Quality assurance and deployment
Quality assurance and deployment: All content must be reviewed before publication; Deployment processes must be documented and secure; Secrets must be managed via environment variables

## Standards

- Book platform: Docusaurus → GitHub Pages deployment
- Authoring tools: Claude Code + Spec-Kit Plus for development
- Code examples: Must be runnable and tested before inclusion
- Writing level: Flesch-Kincaid grade 10-12 for accessibility
- Technology stack: OpenAI Agents/ChatKit, FastAPI, Neon Postgres, Qdrant for RAG chatbot

## RAG chatbot specifications

- Integration: Chatbot embedded directly in the book interface
- Technology stack: OpenAI Agents/ChatKit, FastAPI, Neon Postgres, Qdrant
- Content scope: Answers must derive only from book content
- Feature support: Selected-text question answering capability
- Query handling: Out-of-scope queries must be rejected appropriately
- Data source: Book content only, no external knowledge sources

## Constraints and Success Criteria

- Content format: Markdown-based for documentation
- Security: Secrets must be managed via environment variables only
- Deployment: Process must be documented and reproducible
- Success metrics: Book publicly deployed and accessible
- Content quality: Spec-driven content creation and maintenance
- Chatbot accuracy: Non-hallucinated, accurate responses based on book content

## Governance

This constitution governs all development activities for the spec-driven technical book project. All implementation must align with these principles. Changes to these principles require explicit documentation and approval following the established amendment process. Development workflows must verify compliance with these principles at each stage.

All PRs and reviews must verify compliance with these principles; Complexity must be justified with clear benefits; Use this constitution for development guidance and decision-making.

**Version**: 1.0.0 | **Ratified**: 2025-12-27 | **Last Amended**: 2025-12-27