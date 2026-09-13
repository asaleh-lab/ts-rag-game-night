/** LLM rewrites the question, then we union the hits. */

import { MultiQueryRetriever } from "@langchain/classic/retrievers/multi_query";
import type { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";

import {
  QUERY,
  chatModel,
  gameStore,
  requireEnv,
  titleOf,
  uniqueByTitle,
} from "./binder.js";

requireEnv();

const store = await gameStore();

// Plain nearest-3, so we can see what one phrasing misses.
const base = store.asRetriever({ searchType: "similarity", k: 3 });

// gpt-4o-mini writes other phrasings. JS has no include_original flag, so we
// keep our question by unioning the base hits with the rewrite hits.
const multi = MultiQueryRetriever.fromLLM({
  retriever: base,
  llm: new ChatOpenAI({ model: chatModel(), temperature: 0 }),
  verbose: true,
});

async function invokeWithOriginal(): Promise<Document[]> {
  const [rewritten, original] = await Promise.all([
    multi.invoke(QUERY),
    base.invoke(QUERY),
  ]);
  return uniqueByTitle([...original, ...rewritten]);
}

async function show(label: string, hits: Document[]): Promise<void> {
  console.log(`\n${label}`);
  for (const doc of hits) {
    console.log(`  ${titleOf(doc)}`);
  }
}

console.log(`Query: ${QUERY}`);
await show("Similarity only", await base.invoke(QUERY));
await show("Multi-query (rewrites, then union)", await invokeWithOriginal());
