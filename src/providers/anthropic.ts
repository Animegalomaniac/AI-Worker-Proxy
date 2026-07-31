import Anthropic from '@anthropic-ai/sdk';
import { BaseProvider } from './base';
import { AnthropicRequest } from '../anthropic-types';
import { OpenAIChatRequest, ProviderResponse, OpenAIMessage, ToolCall } from '../types';
import { createOpenAIResponse, StreamSession } from '../utils/response-mapper';

export class AnthropicProvider extends BaseProvider {
  async chat(request: OpenAIChatRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      // Retries are owned by the outer TokenManager/Router rotation — disable
      // the SDK's built-in retry to avoid request amplification (3x per key).
      const clientOpts: Record<string, unknown> = { apiKey, maxRetries: 0 };
      if (this.baseUrl) {
        clientOpts.baseURL = this.baseUrl;
      }
      const client = new Anthropic(clientOpts);

      const { system, messages } = this.convertMessages(request.messages);

      const tools = request.tools?.map((tool) => ({
        name: tool.function.name,
        description: tool.function.description || '',
        input_schema: tool.function.parameters || {},
      }));

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const params: any = {
        model: this.model,
        messages,
        max_tokens: request.max_tokens ?? 4096,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: request.stream || false,
      };

      if (system) {
        params.system = system;
      }

      if (tools && tools.length > 0) {
        params.tools = tools;
      }

      // Map OpenAI stop to Anthropic stop_sequences
      if (request.stop) {
        params.stop_sequences = Array.isArray(request.stop) ? request.stop : [request.stop];
      }

      // Map OpenAI tool_choice to Anthropic tool_choice
      if (request.tool_choice) {
        if (request.tool_choice === 'none') {
          params.tool_choice = { type: 'none' };
        } else if (request.tool_choice === 'auto') {
          params.tool_choice = { type: 'auto' };
        } else if (request.tool_choice === 'required') {
          // OpenAI 'required' (must call a tool) ↔ Anthropic 'any'
          params.tool_choice = { type: 'any' };
        } else if (
          typeof request.tool_choice === 'object' &&
          request.tool_choice.type === 'function'
        ) {
          params.tool_choice = {
            type: 'tool',
            name: request.tool_choice.function.name,
          };
        }
      }

      if (request.stream) {
        return this.handleStream(client, params);
      }
      return this.handleNonStream(client, params);
    } catch (error) {
      return this.handleError(error, 'AnthropicProvider');
    }
  }

  /**
   * Anthropic-native chat — bypasses OpenAI format conversion.
   * Accepts AnthropicRequest directly and returns Anthropic-native response.
   */
  async nativeChat(request: AnthropicRequest, apiKey: string): Promise<ProviderResponse> {
    try {
      // Retries are owned by the outer TokenManager/Router rotation — disable
      // the SDK's built-in retry to avoid request amplification (3x per key).
      const clientOpts: Record<string, unknown> = { apiKey, maxRetries: 0 };
      if (this.baseUrl) {
        clientOpts.baseURL = this.baseUrl;
      }
      const client = new Anthropic(clientOpts);

      const params: Record<string, unknown> = {
        model: this.model,
        messages: request.messages,
        max_tokens: request.max_tokens ?? 4096,
        stream: request.stream || false,
      };

      if (request.system) params.system = request.system;
      if (request.temperature !== undefined) params.temperature = request.temperature;
      if (request.top_p !== undefined) params.top_p = request.top_p;
      if (request.top_k !== undefined) params.top_k = request.top_k;
      if (request.stop_sequences) params.stop_sequences = request.stop_sequences;
      if (request.metadata) params.metadata = request.metadata;

      // Pass tools directly (they're already in Anthropic format)
      if (request.tools && request.tools.length > 0) {
        params.tools = request.tools;
      }

      if (request.tool_choice) {
        params.tool_choice = request.tool_choice;
      }

      if (request.stream) {
        return this.handleNativeStream(client, params);
      }
      return this.handleNativeNonStream(client, params);
    } catch (error) {
      return this.handleError(error, 'AnthropicProvider.nativeChat');
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNonStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    const response = await client.messages.create(params);

    let content = '';
    let toolCalls: ToolCall[] | undefined;

    for (const rawBlock of response.content) {
      const block = rawBlock as unknown as Record<string, unknown>;
      const blockType = block.type as string;
      if (blockType === 'thinking' || blockType === 'redacted_thinking') {
        const label = blockType === 'thinking' ? 'thinking' : 'redacted_thinking';
        const thinkingText = typeof block.thinking === 'string' ? block.thinking : '';
        const dataText = typeof block.data === 'string' ? block.data : '';
        content += `\n\n[${label}] ${thinkingText || dataText}[/${label}]\n\n`;
      } else if (blockType === 'text') {
        content += block.text as string;
      } else if (blockType === 'tool_use') {
        if (!toolCalls) toolCalls = [];
        toolCalls.push({
          id: block.id as string,
          type: 'function',
          function: {
            name: block.name as string,
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      }
    }

    const finishReason =
      response.stop_reason === 'tool_use'
        ? 'tool_calls'
        : response.stop_reason === 'max_tokens'
          ? 'length'
          : 'stop';

    const usage = response.usage
      ? {
          prompt_tokens: response.usage.input_tokens,
          completion_tokens: response.usage.output_tokens,
          total_tokens: response.usage.input_tokens + response.usage.output_tokens,
        }
      : undefined;

    return {
      success: true,
      response: createOpenAIResponse(content, this.model, finishReason, toolCalls, usage),
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    // Use messages.create({stream:true}) instead of messages.stream():
    // create() awaits the upstream response, so connection-time errors (401/429/400)
    // reject here and can trigger key rotation / provider fallback. messages.stream()
    // is fire-and-forget — errors only surface mid-stream, after we've already
    // reported success, which silently bypasses the whole failover chain.
    const stream = await client.messages.create({
      ...params,
      stream: true,
    } as Anthropic.MessageCreateParamsStreaming);
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const session = new StreamSession(this.model);

    (async () => {
      try {
        // Send initial role chunk per OpenAI spec
        await writer.write(session.roleChunk());

        // Track tool calls by index for proper incremental streaming
        let toolCallIndex = -1;
        let hasToolCalls = false;
        // Anthropic reports the real stop reason in message_delta before message_stop
        let stopReason: string | null = null;

        for await (const rawEvent of stream) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const event = rawEvent as any;
          if (event.type === 'content_block_start') {
            if (event.content_block.type === 'tool_use') {
              toolCallIndex++;
              hasToolCalls = true;
              // Send tool call start with id, name, empty args
              await writer.write(
                session.toolCallStartChunk(
                  toolCallIndex,
                  event.content_block.id,
                  event.content_block.name
                )
              );
            } else if (
              event.content_block.type === 'thinking' ||
              event.content_block.type === 'redacted_thinking'
            ) {
              await writer.write(session.textChunk('\n[thinking]\n'));
            }
          } else if (event.type === 'content_block_delta') {
            if (event.delta.type === 'text_delta') {
              await writer.write(session.textChunk(event.delta.text));
            } else if (event.delta.type === 'input_json_delta') {
              // Stream arguments incrementally
              if (toolCallIndex >= 0) {
                await writer.write(
                  session.toolCallArgsChunk(toolCallIndex, event.delta.partial_json)
                );
              }
            } else if (event.delta.type === 'thinking_delta') {
              const thinking = event.delta.thinking;
              if (typeof thinking === 'string') {
                await writer.write(session.textChunk(thinking));
              }
            } else if (event.delta.type === 'signature_delta') {
              // Signature delta — end thinking block
              await writer.write(session.textChunk('\n[/thinking]\n'));
            }
          } else if (event.type === 'message_delta') {
            const deltaStopReason = event.delta?.stop_reason;
            if (typeof deltaStopReason === 'string') {
              stopReason = deltaStopReason;
            }
          } else if (event.type === 'message_stop') {
            const reason = hasToolCalls
              ? 'tool_calls'
              : stopReason === 'max_tokens'
                ? 'length'
                : 'stop';
            await writer.write(session.finishChunk(reason));
            await writer.write(session.done());
          }
        }
      } catch (error) {
        console.error('[AnthropicProvider] Stream error:', error);
        try {
          await writer.write(
            new TextEncoder().encode(
              `data: ${JSON.stringify({ error: { message: 'Stream terminated due to upstream error', type: 'stream_error' } })}\n\n`
            )
          );
          await writer.write(session.finishChunk('stop'));
          await writer.write(session.done());
        } catch {
          // Writer may already be closed
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
   * Native non-stream: forward Anthropic SDK Message directly as rawResponse.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNativeNonStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    const response = await client.messages.create(params);
    return {
      success: true,
      rawResponse: response,
    };
  }

  /**
   * Native stream: re-serialize Anthropic SDK StreamEvents as SSE and forward.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async handleNativeStream(client: Anthropic, params: any): Promise<ProviderResponse> {
    // Same as handleStream: create({stream:true}) awaits the upstream response so
    // connection-time errors reject here and failover/rotation still works.
    const stream = await client.messages.create({
      ...params,
      stream: true,
    } as Anthropic.MessageCreateParamsStreaming);
    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    (async () => {
      try {
        for await (const event of stream) {
          const line = `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
          await writer.write(encoder.encode(line));
        }
      } catch (error) {
        console.error('[AnthropicProvider] Native stream error:', error);
        try {
          const errorEvent = {
            type: 'error',
            error: { type: 'stream_error', message: 'Stream terminated due to upstream error' },
          };
          await writer.write(
            encoder.encode(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`)
          );
          const stopEvent = { type: 'message_stop' };
          await writer.write(
            encoder.encode(`event: message_stop\ndata: ${JSON.stringify(stopEvent)}\n\n`)
          );
        } catch {
          // Writer may already be closed
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

  private convertMessages(messages: OpenAIMessage[]): {
    system?: string;
    messages: Anthropic.MessageParam[];
  } {
    let system: string | undefined;
    const convertedMessages: Anthropic.MessageParam[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        const text =
          typeof msg.content === 'string'
            ? msg.content
            : msg.content
              ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
              : '';
        // Multiple system messages: concatenate instead of overwriting
        system = system ? `${system}\n\n${text}` : text;
      } else if (msg.role === 'user' || msg.role === 'assistant') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const content: any[] = [];

        if (msg.content) {
          if (typeof msg.content === 'string') {
            content.push({ type: 'text', text: msg.content });
          } else {
            for (const part of msg.content) {
              if (part.type === 'text') {
                content.push({ type: 'text', text: part.text });
              } else if (part.type === 'image_url') {
                const imageUrl = part.image_url.url;
                if (imageUrl.startsWith('http://') || imageUrl.startsWith('https://')) {
                  // Remote image URL — pass through using Anthropic's URL image source
                  content.push({
                    type: 'image',
                    source: { type: 'url', url: imageUrl },
                  });
                } else {
                  // Parse data URI to extract mime type and base64 data
                  const commaIdx = imageUrl.indexOf(',');
                  if (commaIdx !== -1) {
                    const header = imageUrl.slice(0, commaIdx);
                    const base64Data = imageUrl.slice(commaIdx + 1);
                    const mimeMatch = header.match(/^data:([^;]+)/);
                    const mediaType = mimeMatch ? mimeMatch[1] : 'image/png';
                    content.push({
                      type: 'image',
                      source: { type: 'base64', media_type: mediaType, data: base64Data },
                    });
                  }
                }
              }
            }
          }
        }

        if (msg.tool_calls) {
          for (const toolCall of msg.tool_calls) {
            let parsedInput: unknown;
            try {
              parsedInput = JSON.parse(toolCall.function.arguments);
            } catch (e) {
              // Fall back to empty object if arguments are not valid JSON
              parsedInput = {};
            }
            content.push({
              type: 'tool_use',
              id: toolCall.id,
              name: toolCall.function.name,
              input: parsedInput,
            });
          }
        }

        if (content.length > 0) {
          convertedMessages.push({ role: msg.role, content });
        }
      } else if (msg.role === 'tool') {
        const toolContent =
          typeof msg.content === 'string'
            ? msg.content || ''
            : msg.content
              ? msg.content.map((p) => (p.type === 'text' ? p.text : '')).join(' ')
              : '';
        convertedMessages.push({
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: msg.tool_call_id!,
              content: toolContent,
            },
          ],
        });
      }
    }

    return { system, messages: convertedMessages };
  }
}
