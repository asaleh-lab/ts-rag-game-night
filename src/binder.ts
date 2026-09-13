/** The Setup binder, plus the store and translator the LangChain scripts share. */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { Document } from "@langchain/core/documents";
import { FunctionalTranslator } from "@langchain/core/structured_query";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { config } from "dotenv";

config({ path: resolve(process.cwd(), ".env") });

export const QUERY = "a short calm game for two people after work";

export type Game = {
  id: string;
  title: string;
  players_min: number;
  players_max: number;
  minutes: number;
  weight: number;
  cooperative: boolean;
  on_shelf: boolean;
  pairs_with: string;
  how_it_plays: string;
};

export type GameMetadata = {
  id: string;
  title: string;
  players_min: number;
  players_max: number;
  minutes: number;
  weight: number;
  cooperative: boolean;
  on_shelf: boolean;
  pairs_with: string;
};

export function requireEnv(): void {
  const key = process.env.OPENAI_API_KEY;
  if (!key || key.includes("your-key")) {
    console.error(
      "Missing OPENAI_API_KEY. Copy .env.example to .env and put your OpenAI API key there.",
    );
    process.exit(1);
  }
}

export function loadGames(): Game[] {
  const raw = readFileSync(resolve(process.cwd(), "data/games.json"), "utf8");
  return JSON.parse(raw) as Game[];
}

/** Title plus how_it_plays is what we embed. Metadata rides along for filters. */
export function binderDocs(): Document[] {
  return loadGames().map(
    (game) =>
      new Document({
        pageContent: `${game.title}. ${game.how_it_plays}`,
        metadata: {
          id: game.id,
          title: game.title,
          players_min: game.players_min,
          players_max: game.players_max,
          minutes: game.minutes,
          weight: game.weight,
          cooperative: game.cooperative,
          on_shelf: game.on_shelf,
          pairs_with: game.pairs_with,
        } satisfies GameMetadata,
      }),
  );
}

export function embeddings(): OpenAIEmbeddings {
  return new OpenAIEmbeddings({ model: "text-embedding-3-small" });
}

/** In-memory store already returns cosine similarity on a 0 to 1 scale. */
export class GameStore extends MemoryVectorStore {}

export async function gameStore(): Promise<GameStore> {
  return GameStore.fromDocuments(binderDocs(), embeddings());
}

/**
 * The in-memory store takes a callable filter, not Chroma's dict.
 * FunctionalTranslator is the JS visitor that turns the LLM's structured
 * query into that callable. Same job as MemoryTranslator in the Python repo.
 */
export class MemoryTranslator extends FunctionalTranslator {}

export function titleOf(doc: Document): string {
  return String(doc.metadata.title ?? "");
}

export function uniqueByTitle(docs: Document[]): Document[] {
  const seen = new Set<string>();
  const out: Document[] = [];
  for (const doc of docs) {
    const title = titleOf(doc);
    if (seen.has(title)) {
      continue;
    }
    seen.add(title);
    out.push(doc);
  }
  return out;
}

export function chatModel(): string {
  return process.env.OPENAI_MODEL ?? "gpt-4o-mini";
}
