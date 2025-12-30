---
id: 1
title: Update Constitution
stage: constitution
date: 2025-12-27
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: Claude Code
command: /sp.constitution
labels: [constitution,spec-driven-book,rag-chatbot]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - null
---

## Prompt

/sp.constitution

Project: Spec-driven technical book with embedded RAG chatbot

Principles:
- Spec-first development
- Verified technical accuracy
- Clear, professional writing
- Reproducible workflows
- Grounded AI responses only

Standards:
- Book: Docusaurus → GitHub Pages
- Authored via Claude Code + Spec-Kit Plus
- Runnable code examples
- Flesch-Kincaid grade 10–12

RAG chatbot:
- Embedded in book
- Stack: OpenAI Agents/ChatKit, FastAPI, Neon Postgres, Qdrant
- Answers from book content only
- Supports selected-text QA
- Rejects out-of-scope queries

Constraints:
- Markdown content
- Secure secrets via env vars
- Documented deployment

Success:
- Book publicly deployed
- Spec-driven content
- Accurate, non-hallucinated chatbot

## Response snapshot

Updated the project constitution file (.specify/memory/constitution.md) with the specified principles, standards, constraints, and success criteria for the spec-driven technical book with embedded RAG chatbot project. The constitution now includes:

- Five core principles: Spec-first development, Verified technical accuracy, Clear professional writing, Reproducible workflows, and Grounded AI responses only
- Standards section covering book platform, authoring tools, code examples, writing level, and technology stack
- RAG chatbot specifications detailing integration, technology stack, content scope, features, and query handling
- Constraints and success criteria covering content format, security, deployment, and success metrics
- Governance section outlining how the constitution governs development activities

## Outcome

- ✅ Impact: Constitution file updated with project-specific principles and guidelines
- 🧪 Tests: No tests required for documentation update
- 📁 Files: .specify/memory/constitution.md created/updated
- 🔁 Next prompts: Plan and implement book features based on constitution
- 🧠 Reflection: Constitution provides clear governance framework for the technical book project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin feature specification based on constitution