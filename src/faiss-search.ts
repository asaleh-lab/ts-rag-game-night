/** Index the binder in FAISS, then search with no LLM. */

import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

import { createRequire } from "node:module";

import type { Document } from "@langchain/core/documents";
import type { Index as FaissIndex } from "faiss-node";

import { QUERY, binderDocs, embeddings, requireEnv, titleOf } from "./binder.js";

const require = createRequire(import.meta.url);
const faiss = require("faiss-node") as typeof import("faiss-node");
export const IndexFlatL2 = faiss.IndexFlatL2;
export const Index = faiss.Index;
export const MetricType = faiss.MetricType;

export type IndexedSheet = {
  doc: Document;
  vector: number[];
};

export function l2Squared(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

export function flatten(vectors: number[][]): number[] {
  return vectors.flat();
}

export function addAll(index: FaissIndex, vectors: number[][]): void {
  index.add(flatten(vectors));
}

export async function embedBinder(): Promise<{
  sheets: IndexedSheet[];
  query: number[];
  dim: number;
}> {
  requireEnv();
  const docs = binderDocs();
  const embed = embeddings();
  const vectors = await embed.embedDocuments(docs.map((doc) => doc.pageContent));
  const query = await embed.embedQuery(QUERY);
  return {
    sheets: docs.map((doc, i) => ({ doc, vector: vectors[i] })),
    query,
    dim: vectors[0]?.length ?? 0,
  };
}

function show(
  label: string,
  index: FaissIndex,
  query: number[],
  sheets: IndexedSheet[],
): void {
  const { distances, labels } = index.search(query, 3);
  console.log(`\n${label}`);
  for (let i = 0; i < labels.length; i += 1) {
    const id = labels[i];
    if (id < 0) {
      continue;
    }
    console.log(`  ${distances[i].toFixed(4)}  ${titleOf(sheets[id].doc)}`);
  }
}

async function main(): Promise<void> {
  const { sheets, query, dim } = await embedBinder();
  const vectors = sheets.map((sheet) => sheet.vector);

  console.log(`Query: ${QUERY}`);
  console.log(`${sheets.length} sheets, dim ${dim}`);

  // Brute-force: every query against every sheet. Exact L2.
  const flat = new IndexFlatL2(dim);
  addAll(flat, vectors);
  show("IndexFlat (brute-force)", flat, query, sheets);

  // HNSW: a neighbour graph. Same vectors, approximate search.
  const hnsw = Index.fromFactory(dim, "HNSW32,Flat", MetricType.METRIC_L2);
  addAll(hnsw, vectors);
  show("IndexHNSWFlat", hnsw, query, sheets);
}

const runningThisFile =
  process.argv[1] !== undefined &&
  resolve(fileURLToPath(import.meta.url)) === resolve(process.argv[1]);

if (runningThisFile) {
  await main();
}
