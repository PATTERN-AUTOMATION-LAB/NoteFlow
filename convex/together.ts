import OpenAI from 'openai';
import {
  internalAction,
  internalMutation,
  internalQuery,
} from './_generated/server';
import { v } from 'convex/values';
import { internal } from './_generated/api';
import { z } from 'zod';
import { actionWithUser } from './utils';
import Instructor from '@instructor-ai/instructor';

const openrouterApiKey = process.env.OPENROUTER_API_KEY ?? 'undefined';

// OpenRouter client for LLM extraction
const openrouter = new OpenAI({
  apiKey: openrouterApiKey,
  baseURL: 'https://openrouter.ai/api/v1',
  defaultHeaders: {
    'HTTP-Referer': 'https://localhost:3000',
    'X-Title': 'noteFlow',
  },
});

// Instructor for returning structured JSON
const client = Instructor({
  client: openrouter,
  mode: 'JSON_SCHEMA',
});

const NoteSchema = z.object({
  title: z
    .string()
    .describe('Short descriptive title of what the voice message is about'),
  summary: z
    .string()
    .describe(
      'A short summary in the first person point of view of the person recording the voice message',
    )
    .max(500),
  actionItems: z
    .array(z.string())
    .describe(
      'A list of action items from the voice note, short and to the point. Make sure all action item lists are fully resolved if they are nested',
    ),
});

export const chat = internalAction({
  args: {
    id: v.id('notes'),
    transcript: v.string(),
  },
  handler: async (ctx, args) => {
    const { transcript } = args;

    console.log('Starting summarization for transcript:', transcript.substring(0, 100) + '...');
    
    if (!openrouterApiKey || openrouterApiKey === 'undefined') {
      console.error('OpenRouter API key is missing');
      await ctx.runMutation(internal.together.saveSummary, {
        id: args.id,
        summary: 'OpenRouter API key missing',
        actionItems: [],
        title: 'Configuration Error',
      });
      return;
    }

    try {
      console.log('Calling OpenRouter API...');
      const extract = await client.chat.completions.create({
        messages: [
          {
            role: 'system',
            content:
              'The following is a transcript of a voice message. Extract a title, summary, and action items from it and answer in JSON in this format: {title: string, summary: string, actionItems: [string, string, ...]}',
          },
          { role: 'user', content: transcript },
        ],
        model: 'anthropic/claude-3.5-haiku',
        response_model: { schema: NoteSchema, name: 'SummarizeNotes' },
        max_tokens: 1000,
        temperature: 0.6,
        max_retries: 3,
      });
      
      console.log('OpenRouter API response received');
      const { title, summary, actionItems } = extract;

      await ctx.runMutation(internal.together.saveSummary, {
        id: args.id,
        summary,
        actionItems,
        title,
      });
      console.log('Summary saved successfully');
    } catch (e) {
      console.error('Error extracting from voice message:', e);
      console.error('Full error details:', JSON.stringify(e, null, 2));
      
      await ctx.runMutation(internal.together.saveSummary, {
        id: args.id,
        summary: `Summary failed to generate: ${e instanceof Error ? e.message : String(e)}`,
        actionItems: [],
        title: 'Error',
      });
    }
  },
});

export const getTranscript = internalQuery({
  args: {
    id: v.id('notes'),
  },
  handler: async (ctx, args) => {
    const { id } = args;
    const note = await ctx.db.get(id);
    return note?.transcription;
  },
});

export const saveSummary = internalMutation({
  args: {
    id: v.id('notes'),
    summary: v.string(),
    title: v.string(),
    actionItems: v.array(v.string()),
  },
  handler: async (ctx, args) => {
    const { id, summary, actionItems, title } = args;
    await ctx.db.patch(id, {
      summary: summary,
      title: title,
      generatingTitle: false,
    });

    let note = await ctx.db.get(id);

    if (!note) {
      console.error(`Couldn't find note ${id}`);
      return;
    }
    for (let actionItem of actionItems) {
      await ctx.db.insert('actionItems', {
        task: actionItem,
        noteId: id,
        userId: note.userId,
      });
    }

    await ctx.db.patch(id, {
      generatingActionItems: false,
    });
  },
});

export type SearchResult = {
  id: string;
  score: number;
};

export const similarNotes = actionWithUser({
  args: {
    searchQuery: v.string(),
  },
  handler: async (ctx, args): Promise<SearchResult[]> => {
    const getEmbedding = await openrouter.embeddings.create({
      input: [args.searchQuery.replace('/n', ' ')],
      model: 'text-embedding-3-small',
    });
    const embedding = getEmbedding.data[0].embedding;

    // 2. Then search for similar notes
    const results = await ctx.vectorSearch('notes', 'by_embedding', {
      vector: embedding,
      limit: 16,
      filter: (q) => q.eq('userId', ctx.userId), // Only search my notes.
    });

    console.log({ results });

    return results.map((r) => ({
      id: r._id,
      score: r._score,
    }));
  },
});

export const embed = internalAction({
  args: {
    id: v.id('notes'),
    transcript: v.string(),
  },
  handler: async (ctx, args) => {
    try {
      console.log('Starting embeddings for transcript:', args.transcript.substring(0, 100) + '...');
      
      const getEmbedding = await openrouter.embeddings.create({
        input: [args.transcript.replace('/n', ' ')],
        model: 'text-embedding-3-small',
      });
      
      console.log('Embeddings response:', JSON.stringify(getEmbedding, null, 2));
      
      if (!getEmbedding.data || !getEmbedding.data[0] || !getEmbedding.data[0].embedding) {
        console.error('Invalid embeddings response format');
        return; // Skip embeddings if it fails
      }
      
      const embedding = getEmbedding.data[0].embedding;

      await ctx.runMutation(internal.together.saveEmbedding, {
        id: args.id,
        embedding,
      });
      
      console.log('Embeddings saved successfully');
    } catch (e) {
      console.error('Embeddings failed:', e);
      // Don't throw - embeddings are optional, let the summarization continue
    }
  },
});

export const saveEmbedding = internalMutation({
  args: {
    id: v.id('notes'),
    embedding: v.array(v.float64()),
  },
  handler: async (ctx, args) => {
    const { id, embedding } = args;
    await ctx.db.patch(id, {
      embedding: embedding,
    });
  },
});
