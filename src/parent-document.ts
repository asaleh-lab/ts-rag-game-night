/** Embed small chunks, then hand back the parent binder page. */

import { ParentDocumentRetriever } from "@langchain/classic/retrievers/parent_document";
import { InMemoryStore } from "@langchain/classic/storage/in_memory";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import {
  QUERY,
  binderDocs,
  embeddings,
  loadGames,
  requireEnv,
  titleOf,
} from "./binder.js";

requireEnv();

const docs = binderDocs();

// Each sheet is two paragraphs, ~280 to 400 chars. 250 cuts on the blank line.
const childSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 250,
  chunkOverlap: 0,
});

// Children go in the vector store. Parents sit in a separate key-value store.
const childStore = new MemoryVectorStore(embeddings());
const parentStore = new InMemoryStore<Uint8Array>();

const retriever = new ParentDocumentRetriever({
  vectorstore: childStore,
  byteStore: parentStore,
  childSplitter,
  childK: 3,
});
// Keep the game id as the parent key so two child hits from one sheet collapse.
await retriever.addDocuments(docs, { ids: loadGames().map((game) => game.id) });

console.log(`Query: ${QUERY}`);
console.log(`${docs.length} parent sheets`);

// The vector store only sees the small chunks. We print those first.
console.log("\nChild chunks (what we embed)");
for (const doc of await childStore.similaritySearch(QUERY, 3)) {
  const preview = doc.pageContent.replace(/\n/g, " ").slice(0, 80);
  console.log(`  ${titleOf(doc)}  ${doc.pageContent.length} chars`);
  console.log(`    ${preview}`);
}

// Same search. The retriever looks up each child parent and returns the full page.
console.log("\nParent sheets (what we return)");
for (const doc of await retriever.invoke(QUERY)) {
  console.log(`  ${titleOf(doc)}  ${doc.pageContent.length} chars`);
}
