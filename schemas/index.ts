import { z } from "zod";

// ── Embedding ────────────────────────────────────────────────────────
export const EmbeddingSchema = z.array(z.number());
export type Embedding = z.infer<typeof EmbeddingSchema>;

// ── Raw text chunk before embedding ─────────────────────────────────
export const RawChunkSchema = z.object({
  fileName: z.string(),
  chunkIndex: z.number().int().nonnegative(),
  text: z.string(),
});
export type RawChunk = z.infer<typeof RawChunkSchema>;

// ── Chunk as stored in DuckDB ──────────────────────────────────────
export const DbChunkSchema = z.object({
  id: z.number().int().nonnegative(),
  file_name: z.string(),
  chunk_idx: z.number().int().nonnegative(),
  content: z.string(),
  embedding: z.string(), // JSON-serialized array
});
export type DbChunk = z.infer<typeof DbChunkSchema>;

// ── Query result row from DuckDB ───────────────────────────────────
export const QueryResultRowSchema = z.object({
  file_name: z.string(),
  chunk_idx: z.number(),
  content: z.string(),
  score: z.number(),
});
export type QueryResultRow = z.infer<typeof QueryResultRowSchema>;

// ── Final query output (JSON to stdout) ────────────────────────────
export const QueryOutputSchema = z.object({
  query: z.string(),
  results: z.array(
    z.object({
      file: z.string(),
      content: z.string(),
      score: z.number(),
    }),
  ),
});
export type QueryOutput = z.infer<typeof QueryOutputSchema>;

// ── Error output ────────────────────────────────────────────────────
export const ErrorOutputSchema = z.object({
  error: z.string(),
});

// ── Config ────────────────────────────────────────────────────────────
export const ConfigSchema = z.object({
  openRouterApiKey: z.string().min(1, "OPENROUTER_API_KEY is required"),
  openRouterModel: z.string().default("openai/text-embedding-3-small"),
  openRouterBaseUrl: z.string().url().default("https://openrouter.ai/api/v1"),
  dbPath: z.string().min(1),
  docsDir: z.string().min(1),
  chunkSize: z.number().int().positive().default(512),
  chunkOverlap: z.number().int().nonnegative().default(64),
  embeddingDims: z.number().int().positive().default(1536),
  embedBatchSize: z.number().int().positive().default(20),
  topK: z.number().int().positive().default(5),
});
export type Config = z.infer<typeof ConfigSchema>;

// ── Build config from env + dir overrides ──────────────────────────
export function loadConfig(overrides: { dbPath?: string; docsDir?: string } = {}): Config {
  return ConfigSchema.parse({
    openRouterApiKey: process.env.OPENROUTER_API_KEY,
    openRouterModel: process.env.OPENROUTER_EMBEDDING_MODEL,
    ...overrides,
    chunkSize: 512,
    chunkOverlap: 64,
    embeddingDims: 1536,
    embedBatchSize: 20,
    topK: 5,
  });
}