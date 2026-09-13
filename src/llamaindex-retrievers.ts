/**
 * Same binder through six retriever ideas.
 *
 * LlamaIndex.TS ships VectorStoreIndex and a BM25 package, but not
 * DocumentSummaryIndex, AutoMergingRetriever, RecursiveRetriever, or
 * QueryFusionRetriever. We code the same six ideas with OpenAI embeddings
 * and gpt-4o-mini so the script still runs on this stack.
 */

import { ChatOpenAI } from "@langchain/openai";

import {
  QUERY,
  binderDocs,
  embeddings,
  loadGames,
  requireEnv,
  titleOf,
} from "./binder.js";

requireEnv();

type Hit = { title: string; text: string; score: number };

const docs = binderDocs();
const games = loadGames();
const embed = embeddings();
const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

const texts = docs.map((doc) => doc.pageContent);
const vectors = await embed.embedDocuments(texts);
const queryVec = await embed.embedQuery(QUERY);

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function topK(
  scored: { title: string; text: string; score: number }[],
  k: number,
): Hit[] {
  return [...scored].sort((x, y) => y.score - x.score).slice(0, k);
}

function show(label: string, hits: Hit[]): void {
  console.log(`\n${label}`);
  const seen = new Set<string>();
  for (const hit of hits) {
    if (seen.has(hit.title)) {
      continue;
    }
    seen.add(hit.title);
    console.log(`  ${hit.title}  ${hit.text.length} chars`);
  }
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

/** Okapi BM25 over the binder sheets. Keyword scores, no embeddings. */
function bm25Search(query: string, k: number): Hit[] {
  const k1 = 1.5;
  const b = 0.75;
  const docsTokens = texts.map(tokenize);
  const df = new Map<string, number>();
  let totalLen = 0;
  for (const tokens of docsTokens) {
    totalLen += tokens.length;
    for (const term of new Set(tokens)) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }
  const avgdl = totalLen / docsTokens.length;
  const n = docsTokens.length;
  const qTerms = tokenize(query);
  const scored = docs.map((doc, i) => {
    const tokens = docsTokens[i];
    const tf = new Map<string, number>();
    for (const term of tokens) {
      tf.set(term, (tf.get(term) ?? 0) + 1);
    }
    let score = 0;
    for (const term of qTerms) {
      const freq = tf.get(term) ?? 0;
      if (freq === 0) {
        continue;
      }
      const nq = df.get(term) ?? 0;
      const idf = Math.log(1 + (n - nq + 0.5) / (nq + 0.5));
      const denom = freq + k1 * (1 - b + b * (tokens.length / avgdl));
      score += idf * ((freq * (k1 + 1)) / denom);
    }
    return { title: titleOf(doc), text: doc.pageContent, score };
  });
  return topK(scored, k);
}

function vectorSearch(queryEmbedding: number[], k: number): Hit[] {
  const scored = docs.map((doc, i) => ({
    title: titleOf(doc),
    text: doc.pageContent,
    score: cosine(queryEmbedding, vectors[i]),
  }));
  return topK(scored, k);
}

function splitBySize(text: string, size: number): string[] {
  if (text.length <= size) {
    return [text];
  }
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + size, text.length);
    if (end < text.length) {
      const slice = text.slice(start, end);
      const breakAt = Math.max(slice.lastIndexOf("\n"), slice.lastIndexOf(" "));
      if (breakAt > size * 0.4) {
        end = start + breakAt;
      }
    }
    const chunk = text.slice(start, end).trim();
    if (chunk.length > 0) {
      chunks.push(chunk);
    }
    start = end;
  }
  return chunks;
}

console.log(`Query: ${QUERY}`);

// Vector index: nearest neighbours, same idea as vector-retriever.ts.
show("Vector index", vectorSearch(queryVec, 3));

// BM25-like: keyword scores on the same sheets, no embeddings.
show("BM25", bm25Search(QUERY, 3));

// Document-summary: LLM writes a short card per sheet, then we embed those cards.
console.log("\nSummarising each sheet once...");
const cards: { title: string; text: string; card: string; vector: number[] }[] =
  [];
for (const doc of docs) {
  const reply = await llm.invoke(
    `One sentence: the title, how the table feels, and who it is for.\n\n${doc.pageContent}`,
  );
  const card = String(reply.content);
  cards.push({
    title: titleOf(doc),
    text: doc.pageContent,
    card,
    vector: await embed.embedQuery(card),
  });
}
show(
  "Document-summary index",
  topK(
    cards.map((card) => ({
      title: card.title,
      text: card.text,
      score: cosine(queryVec, card.vector),
    })),
    3,
  ),
);

// Auto-merging: embed the small leaves, hand back a parent if enough children hit.
type Leaf = {
  title: string;
  parent: string;
  text: string;
  vector: number[];
};
const leaves: Leaf[] = [];
for (const doc of docs) {
  const parts = splitBySize(doc.pageContent, 80);
  const leafVecs = await embed.embedDocuments(parts);
  for (let i = 0; i < parts.length; i += 1) {
    leaves.push({
      title: titleOf(doc),
      parent: doc.pageContent,
      text: parts[i],
      vector: leafVecs[i],
    });
  }
}
const leafHits = [...leaves]
  .map((leaf) => ({ leaf, score: cosine(queryVec, leaf.vector) }))
  .sort((a, b) => b.score - a.score)
  .slice(0, 6);
const childCount = new Map<string, number>();
for (const leaf of leaves) {
  childCount.set(leaf.title, (childCount.get(leaf.title) ?? 0) + 1);
}
const hitCount = new Map<string, { hits: number; parent: string; score: number }>();
for (const { leaf, score } of leafHits) {
  const prev = hitCount.get(leaf.title);
  if (prev) {
    prev.hits += 1;
    prev.score = Math.max(prev.score, score);
  } else {
    hitCount.set(leaf.title, { hits: 1, parent: leaf.parent, score });
  }
}
const merged: Hit[] = [];
for (const [title, info] of hitCount) {
  const total = childCount.get(title) ?? 1;
  const ratio = info.hits / total;
  merged.push({
    title,
    text: ratio >= 0.5 ? info.parent : leafHits.find((h) => h.leaf.title === title)?.leaf.text ?? info.parent,
    score: info.score,
  });
}
show("Auto-merging", topK(merged, 6));

// Recursive: the sheet we embed is Patchwork, the link is the paired box (Azul).
const recursiveHits: Hit[] = [];
for (const hit of vectorSearch(queryVec, 3)) {
  recursiveHits.push(hit);
  const game = games.find((item) => item.title === hit.title);
  const paired = docs.find((doc) => doc.metadata.id === game?.pairs_with);
  if (paired) {
    recursiveHits.push({
      title: titleOf(paired),
      text: paired.pageContent,
      score: hit.score,
    });
  }
}
show("Recursive (follows pairs_with)", recursiveHits);

// Query fusion: rewrite the question, run vector + BM25, fuse ranks with RRF.
const rewriteReply = await llm.invoke(
  `Write three alternative search queries for this need, one per line, and nothing else:\n${QUERY}`,
);
const rewrites = String(rewriteReply.content)
  .split(/\r?\n/)
  .map((line) => line.replace(/^\d+[\).\s-]+/, "").trim())
  .filter((line) => line.length > 0)
  .slice(0, 3);
const queries = [QUERY, ...rewrites];
const rrf = new Map<string, { title: string; text: string; score: number }>();
const rrfK = 60;
for (const q of queries) {
  const qVec = q === QUERY ? queryVec : await embed.embedQuery(q);
  const lists = [vectorSearch(qVec, 3), bm25Search(q, 3)];
  for (const list of lists) {
    list.forEach((hit, rank) => {
      const add = 1 / (rrfK + rank + 1);
      const prev = rrf.get(hit.title);
      if (prev) {
        prev.score += add;
      } else {
        rrf.set(hit.title, { title: hit.title, text: hit.text, score: add });
      }
    });
  }
}
show("Query fusion (RRF)", topK([...rrf.values()], 3));
