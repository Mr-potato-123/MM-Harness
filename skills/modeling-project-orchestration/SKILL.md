---
name: modeling-project-orchestration
description: Coordinate a multi-question modeling project through explicit stage interfaces, durable artifacts, acceptance gates, and dependency-aware progress.
---

# Modeling Project Orchestration

Represent the project as a DAG of question-level deliverables rather than a fixed number of chapters. Preserve the original problem, identify only explicit top-level questions, record direct dependencies and required upstream outputs, and give every node an acceptance condition.

Maintain distinct interfaces for intake, modeling, coding, review, and publication. Each handoff names inputs, outputs, file purposes, validation IDs, uncertainty, and unresolved risks. A later stage may consume accepted reports and registered files but must not reinterpret a path as content or overwrite upstream evidence.

Track progress and decisions durably. On failure, choose the narrowest code, model, or full revision; invalidate affected downstream state without erasing history. Completion requires accepted task reports, required files, final publication artifacts, and a terminal verification result—not merely that every worker returned text.

