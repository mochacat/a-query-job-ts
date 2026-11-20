import duckdb from "duckdb";
const Database = duckdb.Database;
import OpenAI from "openai";
import dotenv from "dotenv";
import fs from "node:fs";
import path from "node:path";

// Suppress dotenv stderr noise
const stderr = process.stderr.write;
process.stderr.write = () => true;
dotenv.config({ path: path.resolve(import.meta.dirname, ".env") });
process.stderr.write = stderr;

// ── Config ──────────────────────────────────────────────────────────
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_MODEL = process.env.OPENROUTER_EMBEDDING_MODEL ?? "openai/text-embedding-3-small";
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

const DB_PATH = path.resolve(import.meta.dirname, "knowledge.db");
const DOCS_DIR = path.resolve(import.meta.dirname, "docs");
const CHUNK_SIZE = 512;
const CHUNK_OVERLAP = 64;
const EMBEDDING_DIMS = 1536; // text-embedding-3-small

type DB = InstanceType<typeof Database>;

// ── Init OpenAI (OpenRouter) ───────────────────────────────────────
const openai = new OpenAI({ apiKey: OPENROUTER_API_KEY, baseURL: OPENROUTER_BASE_URL });

// ── DuckDB helper ──────────────────────────────────────────────────
function dbExec(db: DB, sql: string): Promise<void> {
  return new Promise((resolve, reject) =>
    db.exec(sql, (err) => (err ? reject(err) : resolve())),
  );
}

// ── Chunking ────────────────────────────────────────────────────────
function chunkText(text: string, fileName: string): { fileName: string; chunkIndex: number; text: string }[] {
  const paragraphs = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const chunks: { fileName: string; chunkIndex: number; text: string }[] = [];
  let current = "";
  let index = 0;

  for (const para of paragraphs) {
    if (current.length + para.length > CHUNK_SIZE && current.length > 0) {
      chunks.push({ fileName, chunkIndex: index++, text: current });
      current = current.slice(-CHUNK_OVERLAP) + "\n\n" + para;
    } else {
      current += (current ? "\n\n" : "") + para;
    }
  }
  if (current) chunks.push({ fileName, chunkIndex: index, text: current });
  return chunks;
}

// ── Embedding ──────────────────────────────────────────────────────
async function embed(texts: string[]): Promise<number[][]> {
  const resp = await openai.embeddings.create({
    model: OPENROUTER_MODEL,
    input: texts,
  });
  return resp.data.map((d) => d.embedding);
}

// ── Main ────────────────────────────────────────────────────────────
async function main() {
  if (!OPENROUTER_API_KEY) {
    console.error("Set OPENROUTER_API_KEY environment variable");
    process.exit(1);
  }

  if (!fs.existsSync(DOCS_DIR)) {
    fs.mkdirSync(DOCS_DIR, { recursive: true });
    console.log(`Created docs directory at ${DOCS_DIR}`);
    console.log("   Put .txt files in there and run again.\n");
  }

  const db = new Database(DB_PATH);

  // Enable VSS extension
  await dbExec(db, "INSTALL vss");
  await dbExec(db, "LOAD vss");

  // Create table with FLOAT[] column
  await dbExec(db, `
    CREATE TABLE IF NOT EXISTS chunks (
      id        INTEGER,
      file_name TEXT NOT NULL,
      chunk_idx INTEGER NOT NULL,
      content   TEXT NOT NULL,
      embedding FLOAT[${EMBEDDING_DIMS}]
    )
  `);
  await dbExec(db, "CREATE SEQUENCE IF NOT EXISTS chunks_seq START 1");

  const txtFiles = fs.readdirSync(DOCS_DIR).filter((f) => f.endsWith(".txt"));
  if (txtFiles.length === 0) {
    console.log("No .txt files found in docs/");
    await new Promise(r => db.close(r));
    return;
  }

  console.log(`Found ${txtFiles.length} file(s), chunking...`);

  const allChunks: { fileName: string; chunkIndex: number; text: string }[] = [];
  for (const f of txtFiles) {
    const content = fs.readFileSync(path.join(DOCS_DIR, f), "utf-8");
    allChunks.push(...chunkText(content, f));
  }
  console.log(`Split into ${allChunks.length} chunk(s)`);

  // Embed in batches
  const BATCH = 20;
  const embeddings: number[][] = [];
  for (let i = 0; i < allChunks.length; i += BATCH) {
    const batch = allChunks.slice(i, i + BATCH);
    const texts = batch.map((c) => c.text);
    process.stdout.write(`Embedding batch ${Math.floor(i / BATCH) + 1}/${Math.ceil(allChunks.length / BATCH)}...`);
    const vecs = await embed(texts);
    embeddings.push(...vecs);
    console.log(" done");
  }

  // Wipe old data and insert with proper FLOAT[] casting
  await dbExec(db, "DELETE FROM chunks");
  for (let i = 0; i < allChunks.length; i++) {
    const c = allChunks[i];
    const escaped = c.text.replace(/'/g, "''");
    const vecStr = JSON.stringify(embeddings[i]);
    await dbExec(
      db,
      `INSERT INTO chunks VALUES (nextval('chunks_seq'), '${c.fileName.replace(/'/g, "''")}', ${c.chunkIndex}, '${escaped}', '${vecStr}'::FLOAT[${EMBEDDING_DIMS}])`,
    );
  }
  console.log(`Stored ${allChunks.length} chunk(s) in DuckDB`);

  await new Promise(r => db.close(r));
}

main().catch((err) => {
  console.error("Unresolved error", err);
  process.exit(1);
});