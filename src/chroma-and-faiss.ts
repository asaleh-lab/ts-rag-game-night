/**
 * Same binder vectors in a Chroma-shaped store and in FAISS.
 *
 * chromadb for Node talks to a running server, not a local folder like
 * Python PersistentClient. We keep one embedding pass and put those vectors
 * in an in-process L2 collection (same add/query shape as Chroma) and in
 * FAISS IndexFlatL2.
 */

import { QUERY, requireEnv, titleOf } from "./binder.js";
import { IndexFlatL2, addAll, embedBinder, l2Squared } from "./faiss-search.js";

requireEnv();

type ChromaItem = {
  id: string;
  document: string;
  embedding: number[];
  title: string;
};

function createL2Collection() {
  const items: ChromaItem[] = [];
  return {
    add(input: {
      ids: string[];
      documents: string[];
      embeddings: number[][];
      metadatas: { title: string }[];
    }): void {
      for (let i = 0; i < input.ids.length; i += 1) {
        items.push({
          id: input.ids[i],
          document: input.documents[i],
          embedding: input.embeddings[i],
          title: input.metadatas[i].title,
        });
      }
    },
    query(input: { queryEmbeddings: number[][]; nResults: number }): {
      metadatas: { title: string }[][];
      distances: number[][];
    } {
      const q = input.queryEmbeddings[0];
      const ranked = items
        .map((item) => ({
          title: item.title,
          dist: l2Squared(q, item.embedding),
        }))
        .sort((a, b) => a.dist - b.dist)
        .slice(0, input.nResults);
      return {
        metadatas: [ranked.map((row) => ({ title: row.title }))],
        distances: [ranked.map((row) => row.dist)],
      };
    },
  };
}

const { sheets, query, dim } = await embedBinder();
const texts = sheets.map((sheet) => sheet.doc.pageContent);
const vectors = sheets.map((sheet) => sheet.vector);

console.log(`Query: ${QUERY}`);
console.log(`${sheets.length} sheets, dim ${dim}`);

// Collection name matches the Python script. The JS client would use it on a server.
const collection = createL2Collection();
collection.add({
  ids: sheets.map((sheet) => String(sheet.doc.metadata.id)),
  documents: texts,
  embeddings: vectors,
  metadatas: sheets.map((sheet) => ({ title: titleOf(sheet.doc) })),
});

console.log("\nChroma");
const hits = collection.query({ queryEmbeddings: [query], nResults: 3 });
for (let i = 0; i < hits.metadatas[0].length; i += 1) {
  console.log(`  ${hits.distances[0][i].toFixed(4)}  ${hits.metadatas[0][i].title}`);
}

const flat = new IndexFlatL2(dim);
addAll(flat, vectors);
const { distances, labels } = flat.search(query, 3);
console.log("\nFAISS IndexFlat");
for (let i = 0; i < labels.length; i += 1) {
  const id = labels[i];
  if (id < 0) {
    continue;
  }
  console.log(`  ${distances[i].toFixed(4)}  ${titleOf(sheets[id].doc)}`);
}
