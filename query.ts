import duckdb from "duckdb";
const Database = duckdb.Database;
import OpenAI from "openai";
import dotenv from "dotenv";
import path from "node:path";
import fs from "node:fs";

// Suppress dotenv stderr noise
const stderr = process.stderr.write;
process.stderr.write = () => true;
dotenv.config({ path: path.resolve(import.meta.dirname, ".env") });
process.stderr.write = stderr;

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_MODEL = process.env.OPENROUTER_EMBEDDING_MODEL ?? "openai/text-embedding-3-small";
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
const DB_PATH = path.resolve(import.meta.dirname, "knowledge.db");
const TOP_K = 5;
const EMBEDDING_DIMS = 1536;

const query = process.argv[2];
if (!query) {
  console.error("Usage: npx tsx query.ts <your question>");
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENROUTER_API_KEY, baseURL: OPENROUTER_BASE_URL });

async function main() {
  if (!OPENROUTER_API_KEY) {
    console.error("OPENROUTER_API_KEY not set");
    process.exit(1);
  }
  if (!fs.existsSync(DB_PATH)) {
    console.error("No knowledge.db found. Run npx tsx kb.ts first to ingest files.");
    process.exit(1);
  }

  const db = new Database(DB_PATH);
  await new Promise<void>((resolve, reject) =>
    db.exec("LOAD vss", (err) => (err ? reject(err) : resolve())),
  );

  // Embed query
  const resp = await openai.embeddings.create({ model: OPENROUTER_MODEL, input: [query] });
  const qVec = resp.data[0].embedding;
  const vecStr = JSON.stringify(qVec);

  // DuckDB does the cosine distance natively (0 = identical, 2 = opposite)
  // Convert to similarity score (1 - distance) so higher = more relevant
  const rows = await new Promise<any[]>((resolve, reject) =>
    db.all(
      `SELECT file_name, chunk_idx, content, 1 - array_cosine_distance(embedding, $1::FLOAT[${EMBEDDING_DIMS}]) AS score
       FROM chunks
       ORDER BY score DESC
       LIMIT ${TOP_K}`,
      vecStr,
      (err, r) => (err ? reject(err) : resolve(r)),
    ),
  );

  console.log(JSON.stringify({
    query,
    results: rows.map((r) => ({
      file: r.file_name,
      content: r.content,
      score: r.score,
    })),
  }));

  await new Promise(r => db.close(r));
}

main().catch((err) => {
  console.error(JSON.stringify({ error: err.message }));
  process.exit(1);
});