import duckdb from "duckdb";
import OpenAI from "openai";
import dotenv from "dotenv";
import path from "node:path";
import fs from "node:fs";
import {
  loadConfig,
  QueryResultRowSchema,
  QueryOutput,
  ErrorOutputSchema,
  Config,
} from "./schemas/index.js";

// Suppress dotenv stderr noise
const stderr = process.stderr.write;
process.stderr.write = () => true;
dotenv.config({ path: path.resolve(import.meta.dirname, ".env") });
process.stderr.write = stderr;

type Database = duckdb.Database;
type QueryResultRow = {
  file_name: string;
  chunk_idx: number;
  content: string;
  score: number;
};

function queryDb(db: Database, sql: string, params: string): Promise<QueryResultRow[]> {
  return new Promise((resolve, reject) =>
    db.all(sql, params, (err: Error | null, rows: unknown[]) => {
      if (err) reject(err);
      else resolve(rows as QueryResultRow[]);
    }),
  );
}

async function main() {
  const config: Config = loadConfig({
    dbPath: path.resolve(import.meta.dirname, "knowledge.db"),
    docsDir: path.resolve(import.meta.dirname, "docs"),
  });
  const question: string | undefined = process.argv[2];

  if (!question) {
    console.error("Usage: npx tsx query.ts <your question>");
    process.exit(1);
  }

  if (!fs.existsSync(config.dbPath)) {
    console.error("No knowledge.db found. Run npm run kb first to ingest files.");
    process.exit(1);
  }

  const db: Database = new duckdb.Database(config.dbPath);
  await new Promise<void>((resolve, reject) =>
    db.exec("LOAD vss", (err: Error | null) => (err ? reject(err) : resolve())),
  );

  // Embed query via OpenRouter
  const openai = new OpenAI({
    apiKey: config.openRouterApiKey,
    baseURL: config.openRouterBaseUrl,
  });
  const resp = await openai.embeddings.create({
    model: config.openRouterModel,
    input: [question],
  });
  const qVec: number[] = resp.data[0].embedding;
  const vecStr: string = JSON.stringify(qVec);

  // DuckDB does the cosine distance natively (0 = identical, 2 = opposite)
  // Convert to similarity score (1 - distance) so higher = more relevant
  const rows: QueryResultRow[] = await queryDb(
    db,
    `SELECT file_name, chunk_idx, content, 1 - array_cosine_distance(embedding, $1::FLOAT[${config.embeddingDims}]) AS score
     FROM chunks
     ORDER BY score DESC
     LIMIT ${config.topK}`,
    vecStr,
  );

  // Validate each row
  const validated = rows.map((r) => QueryResultRowSchema.parse(r));

  const output: QueryOutput = {
    query: question,
    results: validated.map((r) => ({
      file: r.file_name,
      content: r.content,
      score: r.score,
    })),
  };

  console.log(JSON.stringify(output));

  await new Promise<void>((resolve) => db.close(() => resolve()));
}

main().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  console.error(JSON.stringify(ErrorOutputSchema.parse({ error: message })));
  process.exit(1);
});