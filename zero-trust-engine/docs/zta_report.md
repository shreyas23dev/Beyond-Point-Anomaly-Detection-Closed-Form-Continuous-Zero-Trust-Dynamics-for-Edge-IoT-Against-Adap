# Zero Trust Decision Engine (Phase 3) Report

## Overview
This standalone microservice consumes inputs from the Dynamic Trust Engine and evaluates them against deterministic policies to output `ALLOW`, `VERIFY`, or `BLOCK` access decisions.

## Architecture
- **Stateless Execution**: The engine does not store trust values or recompute them.
- **Deterministic**: A given input payload against a given policy version will always yield the exact same decision.
- **Audit Logger**: Every decision is persisted to a SQLite database (`data/policy_log.db`) for full auditability and trace tracking via unique UUIDs (`audit_id`).

## Decision Algorithm
1. **Input Validation**: Enforces input bounds `[0,1]` and schema contracts.
2. **Policy Load & Validation**: Parses `policy.yaml` checking for conflicts.
3. **Evaluation**: Matches the trust input to the highest-priority deterministic rule.
4. **Resolution**: If there is a tie in priority, the most restrictive action wins (`BLOCK > VERIFY > ALLOW`).
5. **Logging**: Writes to SQLite.
6. **Return Contract**: Returns the precise JSON Phase 4 contract mapping with NO trust score data.

## Metrics & Complexity
- Database Writes: O(1)
- Policy Match: O(1)
- State Transitions: O(1)

## Setup & Deployment
Run via Docker Compose:
`docker-compose up -d`
