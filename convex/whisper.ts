('use node');

import { internalAction, internalMutation } from './_generated/server';
import { v } from 'convex/values';
import { api, internal } from './_generated/api';


interface whisperOutput {
  detected_language: string;
  segments: any;
  transcription: string;
  translation: string | null;
}

export const chat = internalAction({
  args: {
    fileUrl: v.string(),
    id: v.id('notes'),
  },
  handler: async (ctx, args) => {
    console.log('Starting transcription for file:', args.fileUrl);
    
    const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
    const apiKey = process.env.AZURE_OPENAI_API_KEY!;
    const deployment = process.env.AZURE_OPENAI_WHISPER_DEPLOYMENT!;
    const apiVersion = process.env.AZURE_OPENAI_API_VERSION || '2024-02-15-preview';

    console.log('Azure config:', { endpoint, deployment, apiVersion, hasApiKey: !!apiKey });

    if (!endpoint || !apiKey || !deployment) {
      console.error('Missing Azure env vars:', { endpoint: !!endpoint, apiKey: !!apiKey, deployment: !!deployment });
      throw new Error('Azure OpenAI Whisper env vars are missing');
    }

    // 1) Download audio from storage URL
    console.log('Downloading audio from storage...');
    const audioResp = await fetch(args.fileUrl);
    if (!audioResp.ok) {
      console.error('Failed to fetch audio:', audioResp.status, audioResp.statusText);
      throw new Error(`Failed to fetch audio file: ${audioResp.status}`);
    }
    const audioBuf = await audioResp.arrayBuffer();
    console.log('Audio downloaded, size:', audioBuf.byteLength, 'bytes');

    // 2) Prepare multipart form data for Azure Whisper
    const form = new FormData();
    form.append('file', new Blob([audioBuf]), 'audio.wav');
    form.append('response_format', 'json');
    form.append('language', 'en');

    // 3) Call Azure OpenAI Whisper transcription
    const url = `${endpoint}/openai/deployments/${deployment}/audio/transcriptions?api-version=${apiVersion}`;
    console.log('Calling Azure Whisper API:', url);
    
    const transcribe = await fetch(url, {
      method: 'POST',
      headers: {
        'api-key': apiKey,
      },
      body: form,
    });

    if (!transcribe.ok) {
      const text = await transcribe.text();
      console.error('Azure Whisper API failed:', transcribe.status, text);
      throw new Error(`Azure Whisper failed: ${transcribe.status} ${text}`);
    }
    
    const json = (await transcribe.json()) as { text?: string };
    const transcript = json.text || 'error';
    console.log('Transcription completed:', transcript.substring(0, 100) + '...');

    await ctx.runMutation(internal.whisper.saveTranscript, {
      id: args.id,
      transcript,
    });
  },
});

export const saveTranscript = internalMutation({
  args: {
    id: v.id('notes'),
    transcript: v.string(),
  },
  handler: async (ctx, args) => {
    const { id, transcript } = args;

    await ctx.db.patch(id, {
      transcription: transcript,
      generatingTranscript: false,
    });

    const note = (await ctx.db.get(id))!;
    await ctx.storage.delete(note.audioFileId);

    await ctx.scheduler.runAfter(0, internal.together.chat, {
      id: args.id,
      transcript,
    });

    await ctx.scheduler.runAfter(0, internal.together.embed, {
      id: args.id,
      transcript: transcript,
    });
  },
});
