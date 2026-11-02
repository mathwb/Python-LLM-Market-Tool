# Project Guidelines & Claude Instructions

## Core Principles
* **KISS (Keep It Simple, Stupid):** Always implement the simplest working solution or fix.
* **Complex Solutions Exception:** Only introduce complexity, abstraction, or additional dependencies if there are significant gains in futureproofing, performance, reliability, or security.
* **Code Clarity:** Write self-documenting, understandable code. Avoid unnecessary cleverness.

## Documentation & Comments
* **Comment Style:** Prefer multiple concise, bite-sized comments over long, dense paragraphs.
* **Comment Intent:** Explain *why* complex decisions were made, not *what* clear code is doing.
* **Keep Updated:** Update comments and docs whenever refactoring code.

## Code Style & Standards
* **Error Handling:** Fail gracefully and handle edge cases explicitly without over-engineering.
* **Modularity:** Keep functions and components small, focused, and single-purpose.

## Workflow & Development
* **Minimal Scope:** Touch only the files necessary to complete the task. Avoid broad, unsolicited refactoring.
* **Testing:** Add or update unit/integration tests for any non-trivial changes or bug fixes.
* **Verification:** Run standard linting and test suites before considering a task complete.