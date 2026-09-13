/** The Setup binder behind a small HTTP page: FAISS plus multi-query. */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

import { MultiQueryRetriever } from "@langchain/classic/retrievers/multi_query";
import { Document } from "@langchain/core/documents";
import { BaseRetriever } from "@langchain/core/retrievers";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

import {
  QUERY,
  binderDocs,
  chatModel,
  embeddings,
  requireEnv,
  titleOf,
  uniqueByTitle,
} from "./binder.js";
import { IndexFlatL2, addAll } from "./faiss-search.js";

requireEnv();

const SYSTEM =
  "Answer from these games only. If none of them fit, say so. " +
  "Name the game you would put on the table.";

const MODEL = chatModel();
const docs = binderDocs();
const embed: OpenAIEmbeddings = embeddings();
const vectors = await embed.embedDocuments(docs.map((doc) => doc.pageContent));
const index = new IndexFlatL2(vectors[0]?.length ?? 0);
addAll(index, vectors);

class FaissSheetRetriever extends BaseRetriever {
  lc_namespace = ["ts-rag-game-night", "retrievers"];

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const q = await embed.embedQuery(query);
    const { labels } = index.search(q, 3);
    return labels.filter((id) => id >= 0).map((id) => docs[id]);
  }
}

// Same binder, now in FAISS. Multi-query rewrites the question, then we union the hits.
const base = new FaissSheetRetriever();
const retriever = MultiQueryRetriever.fromLLM({
  retriever: base,
  llm: new ChatOpenAI({ model: MODEL, temperature: 0 }),
});
const llm = new ChatOpenAI({ model: MODEL, temperature: 0 });

async function chat(message: string): Promise<string> {
  const [rewritten, original] = await Promise.all([
    retriever.invoke(message),
    base.invoke(message),
  ]);
  const hits = uniqueByTitle([...original, ...rewritten]);
  if (hits.length === 0) {
    return "None of the games on the shelf fit that.";
  }
  const context = hits.map((doc) => doc.pageContent).join("\n\n");
  const answer = await llm.invoke([
    { role: "system", content: SYSTEM },
    { role: "user", content: `Games:\n${context}\n\nNeed: ${message}` },
  ]);
  const titles = [...new Set(hits.map(titleOf))].join(", ");
  return `${String(answer.content)}\n\nHits: ${titles}`;
}

const PAGE = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>The Setup</title>
  <style>
    body { font-family: Georgia, serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.6rem; }
    #log { white-space: pre-wrap; min-height: 8rem; border: 1px solid #ccc; padding: 0.8rem; }
    form { display: flex; gap: 0.5rem; margin-top: 0.8rem; }
    input { flex: 1; padding: 0.4rem; }
    button { padding: 0.4rem 0.8rem; }
    .examples { margin-top: 0.8rem; }
    .examples button { margin-right: 0.4rem; margin-top: 0.4rem; }
  </style>
</head>
<body>
  <h1>The Setup</h1>
  <p>Ask for a game. We rewrite the question, search the binder, and name a box.</p>
  <div id="log"></div>
  <form id="ask">
    <input id="q" name="q" autocomplete="off" placeholder="a short calm game for two people after work" />
    <button type="submit">Ask</button>
  </form>
  <div class="examples">
    <button type="button" data-q="${QUERY}">${QUERY}</button>
    <button type="button" data-q="a loud party game for a crowd">a loud party game for a crowd</button>
  </div>
  <script>
    const log = document.getElementById("log");
    const form = document.getElementById("ask");
    const input = document.getElementById("q");
    async function send(text) {
      log.textContent += "You: " + text + "\\n\\n";
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      log.textContent += "Binder: " + data.answer + "\\n\\n";
    }
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      send(text);
    });
    for (const button of document.querySelectorAll("[data-q]")) {
      button.addEventListener("click", () => send(button.getAttribute("data-q")));
    }
  </script>
</body>
</html>
`;

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      chunks.push(chunk);
    });
    req.on("end", () => {
      resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", reject);
  });
}

const port = Number(process.env.PORT ?? 7860);
const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(PAGE);
    return;
  }
  if (req.method === "POST" && req.url === "/chat") {
    try {
      const body = JSON.parse(await readBody(req)) as { message?: string };
      const message = (body.message ?? "").trim();
      if (!message) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "message is required" }));
        return;
      }
      const answer = await chat(message);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ answer }));
    } catch (err) {
      const detail = err instanceof Error ? err.message : "chat failed";
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: detail }));
    }
    return;
  }
  res.writeHead(404, { "content-type": "text/plain" });
  res.end("Not found");
});

server.listen(port, () => {
  console.log(`Open http://127.0.0.1:${port}`);
});
