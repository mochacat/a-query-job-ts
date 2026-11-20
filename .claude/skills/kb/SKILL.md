---
name: kb
description: Search the local knowledge base and answer questions from ingested documents
---

Search the knowledge base stored in DuckDB for relevant documents matching the user's question, then answer based on those documents.

1. Run `npx tsx query.ts "<question>"` from the project directory (`/home/cat/Developer/practice/typescript/file-manipulation/`)
2. Parse the JSON output — it contains `query` and `results` (top 5 chunks with `file`, `content`, `score`)
3. Answer the user's question based on the retrieved content. Cite the source file names.

If the command errors with "No knowledge.db found", tell the user to ingest documents first by running `npx tsx kb.ts` after placing .txt files in the docs/ directory.