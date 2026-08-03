import { GoogleGenAI, type Content, type Part } from '@google/genai';
import { BaseProvider } from './base';
import { OpenAIChatRequest, ProviderResponse, OpenAIMessage, ToolCall } from '../types';
import { createOpenAIResponse, StreamSession, generateId } from '../utils/response-mapper';

/**
 * Thought signatures are required by Gemini 3 for function calling.
 * We encode them into the tool call ID so they survive the round-trip
 * through OpenAI-format clients that don't know about signatures.
 *
 * Format: tsig:<base64url_signature>:<randomId>
 */
const TSIG_PREFIX = 'tsig:';

function encodeToolCallId(signature?: string): string {
  const randomPart = generateId(12);
  if (!signature) return `call_${randomPart}`;
  const encoded = btoa(signature).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return `${TSIG_PREFIX}${encoded}:${randomPart}`;
}

function decodeThoughtSignature(toolCallId: string): string | undefined {
  if (!toolCallId.startsWith(TSIG_PREFIX)) return undefined;
  const rest = toolCallId.slice(TSIG_PREFIX.length);
  const colonIdx = rest.lastIndexOf(':');
  if (colonIdx === -1) return undefined;
  const encoded = rest.slice(0, colonIdx);
  return atob(encoded.replace(/-/g, '+').replace(/_/g, '/'));
}

export class GoogleProvider extends BaseProvider {
  private grounding: boolean;

  constructor(model: string, baseUrl?: string, grounding = false) {
    super(model, baseUrl);
    this.grounding = grounding;
  }

  async chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      const ai = new GoogleGenAI({ apiKey });

      const { systemInstruction, contents } = this.convertMessages(request.messages);

      // Build tools array
      const tools: Record<string, unknown>[] = [];

      // User-defined function declarations
      if (request.tools && request.tools.length > 0) {
        tools.push({
          functionDeclarations: request.tools.map((tool) => ({
            name: tool.function.name,
            description: tool.function.description || '',
            parameters: tool.function.parameters || {},
          })),
        });
      }

      // Google Search grounding
      if (this.grounding) {
        tools.push({ googleSearch: {} });
      }

      const config: Record<string, unknown> = {
        temperature: request.temperature,
        maxOutputTokens: request.max_tokens,
        topP: request.top_p,
      };

      if (systemInstruction) {
        config.systemInstruction = systemInstruction;
      }
      if (tools.length > 0) {
        config.tools = tools;
      }

      if (request.stream) {
        return this.handleStream(ai, contents, config);
      }
      return this.handleNonStream(ai, contents, config);
    } catch (error) {
      return this.handleError(error, 'GoogleProvider');
    }
  }

  private async handleNonStream(
    ai: GoogleGenAI,
    contents: Content[],
    config: Record<string, unknown>
  ): Promise<ProviderResponse> {
    const response = await ai.models.generateContent({
      model: this.model,
      contents,
      config,
    });

    const content = response.text || '';
    let toolCalls: ToolCall[] | undefined;

    // Extract function calls with thought signatures
    const parts = response.candidates?.[0]?.content?.parts;
    if (parts) {
      const fcParts = parts.filter((p: Part) => p.functionCall);
      if (fcParts.length > 0) {
        toolCalls = fcParts.map((p: Part) => ({
          id: encodeToolCallId((p as Record<string, unknown>).thoughtSignature as string),
          type: 'function' as const,
          function: {
            name: p.functionCall!.name || '',
            arguments: JSON.stringify(p.functionCall!.args || {}),
          },
        }));
      }
    }

    const candidateFinish = response.candidates?.[0]?.finishReason as string | undefined;
    const finishReason = toolCalls
      ? 'tool_calls'
      : candidateFinish === 'MAX_TOKENS'
        ? 'length'
        : 'stop';

    const usageMeta = response.usageMetadata;
    const usage = usageMeta
      ? {
          prompt_tokens: usageMeta.promptTokenCount ?? 0,
          completion_tokens: usageMeta.candidatesTokenCount ?? 0,
          total_tokens:
            usageMeta.totalTokenCount ??
            (usageMeta.promptTokenCount ?? 0) + (usageMeta.candidatesTokenCount ?? 0),
        }
      : undefined;

    return {
      success: true,
      response: createOpenAIResponse(content, this.model, finishReason, toolCalls, usage),
    };
  }

  private async handleStream(
    ai: GoogleGenAI,
    contents: Content[],
    config: Record<string, unknown>
  ): Promise<ProviderResponse> {
    const response = await ai.models.generateContentStream({
      model: this.model,
      contents,
      config,
    });

    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const session = new StreamSession(this.model);
    const encoder = new TextEncoder();

    (async () => {
      try {
        await writer.write(session.roleChunk());
        let hasToolCalls = false;
        // Cumulative tool call index — must not restart per chunk, or parallel
        // calls arriving in different chunks collide on the same index
        let toolCallCount = 0;
        let lastFinishReason: string | undefined;

        for await (const chunk of response) {
          // Text content
          if (chunk.text) {
            await writer.write(session.textChunk(chunk.text));
          }

          const candidate = chunk.candidates?.[0];
          if (candidate?.finishReason) {
            lastFinishReason = candidate.finishReason as string;
          }

          // Function calls
          const parts = candidate?.content?.parts;
          if (parts) {
            const fcParts = parts.filter((p: Part) => p.functionCall);
            for (const p of fcParts) {
              hasToolCalls = true;
              const index = toolCallCount++;
              const callId = encodeToolCallId(
                (p as Record<string, unknown>).thoughtSignature as string
              );
              await writer.write(
                session.toolCallStartChunk(index, callId, p.functionCall!.name || '')
              );
              await writer.write(
                session.toolCallArgsChunk(index, JSON.stringify(p.functionCall!.args || {}))
              );
            }
          }
        }

        await writer.write(
          session.finishChunk(
            hasToolCalls ? 'tool_calls' : lastFinishReason === 'MAX_TOKENS' ? 'length' : 'stop'
          )
        );
        await writer.write(session.done());
      } catch (error) {
        console.error('[GoogleProvider] Stream error:', error);
        try {
          await writer.write(
            encoder.encode(
              `data: ${JSON.stringify({ error: { message: 'Stream terminated due to upstream error', type: 'stream_error' } })}\n\n`
            )
          );
          await writer.write(session.finishChunk('stop'));
          await writer.write(session.done());
        } catch {
          // Writer already closed
        }
      } finally {
        try {
          await writer.close();
        } catch {
          // Already closed
        }
      }
    })();

    return { success: true, stream: readable };
  }

  /**
   * Convert OpenAI messages to Google GenAI Content format.
   * Preserves thought signatures from encoded tool call IDs.
   */
  private convertMessages(messages: OpenAIMessage[]): {
    systemInstruction?: string;
    contents: Content[];
  } {
    let systemInstruction: string | undefined;
    const contents: Content[] = [];

    // Build tool_call_id → function_name lookup map
    const toolCallNameMap = new Map<string, string>();
    for (const m of messages) {
      if (m.tool_calls) {
        for (const tc of m.tool_calls) {
          toolCallNameMap.set(tc.id, tc.function.name);
        }
      }
    }

    for (const msg of messages) {
      if (msg.role === 'system') {
        const text =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
              : '';
        // Multiple system messages: concatenate instead of overwriting
        systemInstruction = systemInstruction ? `${systemInstruction}\n\n${text}` : text;
      } else if (msg.role === 'user') {
        const parts: Part[] = [];
        if (typeof msg.content === 'string') {
          parts.push({ text: msg.content || '' });
        } else if (msg.content) {
          for (const part of msg.content) {
            if (part.type === 'text') {
              parts.push({ text: part.text });
            } else if (part.type === 'image_url') {
              const dataUri = part.image_url.url;
              if (dataUri.startsWith('http://') || dataUri.startsWith('https://')) {
                // Gemini inlineData requires base64 — remote URLs can't be passed through
                console.warn(
                  '[GoogleProvider] Remote image URL dropped (not supported):',
                  dataUri.slice(0, 100)
                );
                continue;
              }
              // Parse data URI to extract mime type and base64 data
              const commaIdx = dataUri.indexOf(',');
              if (commaIdx !== -1) {
                const header = dataUri.slice(0, commaIdx);
                const base64Data = dataUri.slice(commaIdx + 1);
                const mimeMatch = header.match(/^data:([^;]+)/);
                const mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
                parts.push({
                  inlineData: { mimeType, data: base64Data },
                } as Part);
              }
            }
          }
        }
        if (parts.length > 0) {
          contents.push({ role: 'user', parts });
        }
      } else if (msg.role === 'assistant') {
        const parts: Part[] = [];

        if (msg.content) {
          if (typeof msg.content === 'string') {
            parts.push({ text: msg.content });
          } else {
            for (const part of msg.content) {
              if (part.type === 'text') {
                parts.push({ text: part.text });
              }
            }
          }
        }

        if (msg.tool_calls) {
          for (const toolCall of msg.tool_calls) {
            let parsedArgs: unknown;
            try {
              parsedArgs = JSON.parse(toolCall.function.arguments);
            } catch {
              // Fall back to empty object if arguments are not valid JSON
              parsedArgs = {};
            }
            const part: Record<string, unknown> = {
              functionCall: {
                name: toolCall.function.name,
                args: parsedArgs,
              },
            };
            // Restore thought signature from encoded tool call ID
            const sig = decodeThoughtSignature(toolCall.id);
            if (sig) {
              part.thoughtSignature = sig;
            }
            parts.push(part as Part);
          }
        }

        if (parts.length > 0) {
          contents.push({ role: 'model', parts });
        }
      } else if (msg.role === 'tool') {
        const functionName = toolCallNameMap.get(msg.tool_call_id || '');
        const toolContent =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
              : '';

        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: functionName || 'unknown',
                response: { content: toolContent },
              },
            },
          ],
        });
      }
    }

    return { systemInstruction, contents };
  }
}
