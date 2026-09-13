/** Same query through similarity, MMR, and a score threshold. */

import { ScoreThresholdRetriever } from "@langchain/classic/retrievers/score_threshold";
import type { VectorStoreRetrieverInterface } from "@langchain/core/vectorstores";

import { QUERY, gameStore, requireEnv, titleOf } from "./binder.js";

requireEnv();

const store = await gameStore();

async function show(
  label: string,
  retriever: Pick<VectorStoreRetrieverInterface, "invoke">,
): Promise<void> {
  console.log(`\n${label}`);
  const hits = await retriever.invoke(QUERY);
  if (hits.length === 0) {
    console.log("  (no hits)");
    return;
  }
  for (const doc of hits) {
    console.log(`  ${titleOf(doc)}`);
  }
}

console.log(`Query: ${QUERY}`);
// Nearest neighbours only. k=3, no extra ranking.
await show(
  "Similarity",
  store.asRetriever({ searchType: "similarity", k: 3 }),
);
// Fetch 8, then keep 3 that are close and not copies of each other. lambda 0.3 leans diverse.
await show(
  "MMR",
  store.asRetriever({
    searchType: "mmr",
    k: 3,
    searchKwargs: { fetchK: 8, lambda: 0.3 },
  }),
);
// Drop anything under 0.4 cosine. maxK is a cap, not a promise of 8 titles.
await show(
  "Score threshold",
  ScoreThresholdRetriever.fromVectorStore(store, {
    minSimilarityScore: 0.4,
    maxK: 8,
  }),
);
