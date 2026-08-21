# Default layout profile

Use a compact article layout suitable for a single technical question:

- `geometry`, `amsmath`, `amssymb`, `booktabs`, `graphicx`, `xcolor`, and
  `hyperref` are acceptable; add `ctex` only when Chinese text is present.
- Title page metadata, abstract, problem statement, method, results,
  validation/evidence, limitations, and conclusion are the minimum sections.
- Prefer `booktabs` tables and consistent SI units. Keep captions factual and
  tie each table/figure to an artifact or omit it.
- Run the Harness structural checks first. Never imply a PDF exists until a
  separately recorded compiler run succeeds.

