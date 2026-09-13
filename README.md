# ts-rag-game-night

This repo demonstrates how to retrieve from a small catalog with more than one LangChain retriever, then keep the same embeddings in two vector stores. The example we will use is the shelf binder at The Setup, a one-room board game cafe. In our case we need a game for two tired people, even when they do not name a title.

## Setup

```powershell
npm install
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## Let's retrieve by similarity, then MMR, then a score threshold

```powershell
npx tsx src/vector-retriever.ts
```

## Let's rewrite the question and union the hits

```powershell
npx tsx src/multi-query.ts
```

## Let's split the question into meaning and a filter

```powershell
npx tsx src/self-query.ts
```

## Let's embed small chunks and return the parent sheet

```powershell
npx tsx src/parent-document.ts
```

## Same binder in LlamaIndex

LlamaIndex.TS does not ship DocumentSummaryIndex, AutoMergingRetriever, RecursiveRetriever, or QueryFusionRetriever, so this script codes the same six ideas (vector, BM25-like, summary cards, auto-merging, recursive `pairs_with`, RRF fusion) with OpenAI embeddings.

```powershell
npx tsx src/llamaindex-retrievers.ts
```

## Let's search FAISS with no LLM

```powershell
npx tsx src/faiss-search.ts
```

## Let's keep the same embeddings in Chroma and FAISS

chromadb for Node talks to a server, not a local `chroma_data` folder. We still do one embedding pass and put those vectors in a Chroma-shaped L2 collection and in the brute-force L2 index.

```powershell
npx tsx src/chroma-and-faiss.ts
```

Same binder, same question. Only the store changes (Chroma then FAISS). You should see Patchwork first. Distances can match. The titles should not change.

## Serve it with a small HTTP page

```powershell
npx tsx src/app.ts
```

Open the local URL the script prints.

This article is a practical implementation of the concepts in Advanced RAG with Vector Databases and Retrievers.
