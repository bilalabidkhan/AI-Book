---
description: "Task list for Professional Landing Page for Docusaurus site"
---

# Tasks: Professional Landing Page for My AI Book

**Input**: Design documents from `/specs/7-home-landing-page/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: No explicit test requirements were specified in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus project**: `src/`, `docs/`, `static/` at repository root
- **Pages**: `src/pages/` at repository root
- **CSS**: `src/css/` at repository root
- Adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Verify current project structure and existing files
- [X] T002 [P] Set up local development environment and verify current site functionality
- [X] T003 Create backup of existing configuration files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Create directory structure for landing page components in src/pages/ and src/components/
- [X] T005 [P] Verify Docusaurus routing configuration supports index.js as root page
- [X] T006 [P] Ensure existing documentation content remains unchanged

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Access Professional Home Page (Priority: P1) 🎯 MVP

**Goal**: Replace "Page Not Found" error with professional landing page that displays hero section, book title, and description

**Independent Test**: Visit the root URL and verify the professional landing page displays correctly with all required elements.

### Implementation for User Story 1

- [X] T007 [P] [US1] Create basic index.js page component in src/pages/index.js
- [X] T008 [US1] Implement hero section structure with book title and description
- [X] T009 [US1] Add course focus information to hero section
- [X] T010 [US1] Style hero section with basic layout
- [X] T011 [US1] Test that root URL displays landing page instead of "Page Not Found" error
- [X] T012 [US1] Verify book title, description, and course focus are clearly visible

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Navigate Using Clear CTAs (Priority: P1)

**Goal**: Provide clear CTA buttons ("Read Docs", "Start Module 1", "GitHub") that link correctly to existing documentation and modules

**Independent Test**: Verify the CTA buttons exist and link correctly to their destinations.

### Implementation for User Story 2

- [X] T013 [P] [US2] Design CTA button components with appropriate styling
- [X] T014 [US2] Implement "Read Docs" button with correct link to documentation
- [X] T015 [US2] Implement "Start Module 1" button with correct link to first module
- [X] T016 [US2] Implement "GitHub" button with correct link to repository
- [X] T017 [US2] Style CTA buttons to match academic theme
- [X] T018 [US2] Test that all CTA buttons are clearly visible and properly styled
- [X] T019 [US2] Verify all navigation links correctly route to their intended destinations

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Experience Consistent Design (Priority: P2)

**Goal**: Maintain consistent typography and layout with existing documentation theme

**Independent Test**: Compare the landing page design with existing documentation pages for consistency.

### Implementation for User Story 3

- [X] T020 [P] [US3] Extract typography variables from existing custom.css
- [X] T021 [US3] Apply consistent typography to landing page elements
- [X] T022 [US3] Implement consistent layout patterns matching documentation theme
- [X] T023 [US3] Apply consistent color scheme from documentation theme
- [X] T024 [US3] Test typography consistency with existing documentation pages
- [X] T025 [US3] Verify layout patterns match existing documentation theme
- [X] T026 [US3] Ensure visual consistency is maintained with documentation

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 6: User Story 4 - Access Across All Devices (Priority: P2)

**Goal**: Ensure landing page is fully responsive and works well on desktop and mobile devices

**Independent Test**: View the page on different screen sizes and devices.

### Implementation for User Story 4

- [X] T027 [P] [US4] Implement responsive layout with CSS Flexbox/Grid
- [X] T028 [US4] Add responsive breakpoints matching documentation theme
- [X] T029 [US4] Test layout on desktop screen sizes
- [X] T030 [US4] Test layout on tablet screen sizes
- [X] T031 [US4] Test layout on mobile screen sizes
- [X] T032 [US4] Optimize touch targets for mobile devices
- [X] T033 [US4] Verify page displays correctly across all device sizes

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T034 [P] Update documentation files to reflect new landing page
- [X] T035 Optimize landing page performance and loading times
- [X] T036 Run accessibility checks to ensure WCAG 2.1 AA compliance
- [X] T037 [P] Verify site works correctly in both local development and deployment
- [X] T038 Run quickstart.md validation to ensure all functionality works as expected
- [X] T039 Final testing across browsers and devices

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each User Story

- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in priority order
- Different user stories can be worked on sequentially by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all parallelizable tasks for User Story 1 together:
Task: "Create basic index.js page component in src/pages/index.js"
Task: "Style hero section with basic layout"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Sequential Team Strategy

With a single developer:

1. Complete Setup + Foundational together
2. Once Foundational is done:
   - Complete User Story 1 (P1)
   - Complete User Story 2 (P1)
   - Complete User Story 3 (P2)
   - Complete User Story 4 (P2)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files or can be done in parallel, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All existing documentation content must remain unchanged while only adding landing page