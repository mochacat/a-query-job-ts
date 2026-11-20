import duckdb from "duckdb";
import OpenAI from "openai";
import dotenv from "dotenv";
import fs from "node:fs";
import path from "node:path";
import { loadConfig, RawChunk, RawChunkSchema, Config } from "./schemas/index.js";

// Suppress dotenv stderr noise
const stderr = process.stderr.write;
process.stderr.write = () => true;
dotenv.config({ path: path.resolve(import.meta.dirname, ".env") });
process.stderr.write = stderr;

type Database = duckdb.Database;
type Callback<T> = (err: duckdb.DuckDbError | null) => void;

// ── DuckDB promisified helpers ──────────────────────────────────────
function dbExec(db: Database, sql: string): Promise<void> {
  return new Promise((resolve, reject) =>
    db.exec(sql, (err) => (err ? reject(err) : resolve())),
  );
}

// ── Chunking ────────────────────────────────────────────────────────
function chunkText(text: string, fileName: string, chunkSize: number, chunkOverlap: number): RawChunk[] {
  // Split on sentence boundaries (period, question mark, exclamation followed by space/newline)
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const chunks: RawChunk[] = [];
  let current = "";
  let index = 0;

  for (const sentence of sentences) {
    // If adding this sentence would exceed the limit and we have content, flush
    if (current.length + sentence.length > chunkSize && current.length > 0) {
      chunks.push(RawChunkSchema.parse({ fileName, chunkIndex: index++, text: current.trim() }));
      // overlap: keep the last chunkOverlap chars worth of trailing sentences
      const overlapSentences: string[] = [];
      let overlapLen = 0;
      for (const s of current.split(/(?<=[.!?])\s+/).reverse()) {
        if (overlapLen + s.length > chunkOverlap && overlapSentences.length > 0) break;
        overlapSentences.unshift(s);
        overlapLen += s.length;
      }
      current = overlapSentences.join(" ") + " " + sentence;
    } else {
      current += (current ? " " : "") + sentence;
    }
  }
  if (current.trim()) chunks.push(RawChunkSchema.parse({ fileName, chunkIndex: index, text: current.trim() }));
  return chunks;
}

// ── Escape single-quoted SQL string ──────────────────────────────────
function sqlEscape(val: string): string {
  return val.replace(/'/g, "''");
}

// ── Main ────────────────────────────────────────────────────────────
async function main() {
  const config: Config = loadConfig({
    dbPath: path.resolve(import.meta.dirname, "knowledge.db"),
    docsDir: path.resolve(import.meta.dirname, "docs"),
  });

  if (!fs.existsSync(config.docsDir)) {
    fs.mkdirSync(config.docsDir, { recursive: true });
    console.log(`Created docs directory at ${config.docsDir}`);
    console.log("   Put .txt files in there and run again.\n");
    return;
  }

  const db: Database = new duckdb.Database(config.dbPath);

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
      embedding FLOAT[${config.embeddingDims}]
    )
  `);
  await dbExec(db, "CREATE SEQUENCE IF NOT EXISTS chunks_seq START 1");

  // Read .txt files
  const txtFiles: string[] = fs.readdirSync(config.docsDir).filter((f) => f.endsWith(".txt"));
  if (txtFiles.length === 0) {
    console.log("No .txt files found in docs/");
    await new Promise<void>((resolve) => db.close(() => resolve()));
    return;
  }

  console.log(`Found ${txtFiles.length} file(s), chunking...`);

  // Chunk all files
  const allChunks: RawChunk[] = txtFiles.flatMap((f) => {
    const content = fs.readFileSync(path.join(config.docsDir, f), "utf-8");
    return chunkText(content, f, config.chunkSize, config.chunkOverlap);
  });
  console.log(`Split into ${allChunks.length} chunk(s)`);

  // Embed in batches via OpenRouter
  const openai = new OpenAI({
    apiKey: config.openRouterApiKey,
    baseURL: config.openRouterBaseUrl,
  });

  const embeddings: number[][] = [];
  for (let i = 0; i < allChunks.length; i += config.embedBatchSize) {
    const batch = allChunks.slice(i, i + config.embedBatchSize);
    const texts = batch.map((c) => c.text);
    process.stdout.write(`Embedding batch ${Math.floor(i / config.embedBatchSize) + 1}/${Math.ceil(allChunks.length / config.embedBatchSize)}...`);
    const resp = await openai.embeddings.create({
      model: config.openRouterModel,
      input: texts,
    });
    embeddings.push(...resp.data.map((d) => d.embedding));
    console.log(" done");
  }

  // Wipe old data and insert
  await dbExec(db, "DELETE FROM chunks");
  for (let i = 0; i < allChunks.length; i++) {
    const c = allChunks[i];
    const vecStr = JSON.stringify(embeddings[i]);
    await dbExec(
      db,
      `INSERT INTO chunks VALUES (nextval('chunks_seq'), '${sqlEscape(c.fileName)}', ${c.chunkIndex}, '${sqlEscape(c.text)}', '${vecStr}'::FLOAT[${config.embeddingDims}])`,
    );
  }
  console.log(`Stored ${allChunks.length} chunk(s) in DuckDB`);

  await new Promise<void>((resolve) => db.close(() => resolve()));
}

main().catch((err) => {
  console.error("Unresolved error", err);
  process.exit(1);
});