# Problem brief schema

Use these stable headings (or the equivalent JSON fields):

1. `question`: exact request, decision variable, units, and acceptance criteria.
2. `artifacts`: logical name, digest, detected media type, provenance, and
   whether the content was directly inspected or only registered.
3. `observations`: facts directly supported by an artifact, each with an
   artifact reference.
4. `assumptions`: assumptions required to proceed, with a reason and risk.
5. `unknowns`: missing or unreadable evidence that must not be filled in.
6. `handoff`: validation IDs and expected outputs for modeling/coding.

