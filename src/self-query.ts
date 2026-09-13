/** LLM splits the question into meaning plus a metadata filter. */

import { SelfQueryRetriever } from "@langchain/classic/retrievers/self_query";
import { AttributeInfo } from "@langchain/classic/schema/query_constructor";
import type { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";

import {
  MemoryTranslator,
  QUERY,
  chatModel,
  gameStore,
  requireEnv,
  titleOf,
} from "./binder.js";

requireEnv();

const store = await gameStore();

// Names and types the query constructor is allowed to filter on.
// The descriptions are how gpt-4o-mini decides what "short" and "two" mean.
const fields = [
  new AttributeInfo(
    "players_min",
    "integer",
    "Smallest player count. Two people need players_min <= 2.",
  ),
  new AttributeInfo(
    "players_max",
    "integer",
    "Largest player count. Two people need players_max >= 2.",
  ),
  new AttributeInfo(
    "minutes",
    "integer",
    "Typical play time in minutes. Short after-work games are 45 or under.",
  ),
  new AttributeInfo(
    "weight",
    "integer",
    "Rules weight from 1 (light) to 4 (heavy).",
  ),
  new AttributeInfo(
    "cooperative",
    "boolean",
    "True if players win together, false if they play against each other.",
  ),
  new AttributeInfo(
    "on_shelf",
    "boolean",
    "True if the box is on the open shelf tonight.",
  ),
];

const base = store.asRetriever({ searchType: "similarity", k: 3 });

// gpt-4o-mini keeps "calm" as the search text and turns "short" / "two" into filters.
const selfq = SelfQueryRetriever.fromLLM({
  llm: new ChatOpenAI({ model: chatModel(), temperature: 0 }),
  vectorStore: store,
  documentContents: "Staff binder notes for board games at The Setup cafe",
  attributeInfo: fields,
  structuredQueryTranslator: new MemoryTranslator(),
  searchParams: { k: 3 },
});

function show(label: string, hits: Document[]): void {
  console.log(`\n${label}`);
  for (const doc of hits) {
    const meta = doc.metadata;
    console.log(
      `  ${titleOf(doc)}  ${meta.players_min}-${meta.players_max}p  ${meta.minutes} min`,
    );
  }
}

console.log(`Query: ${QUERY}`);
show("Similarity only", await base.invoke(QUERY));

// Print the split before the hits so we can see the filter the model built.
const parsed = await selfq.queryConstructor.invoke({ query: QUERY });
console.log(`\nMeaning: ${parsed.query}`);
console.log(`Filter: ${String(parsed.filter)}`);
show("Self-query (meaning + filter)", await selfq.invoke(QUERY));
