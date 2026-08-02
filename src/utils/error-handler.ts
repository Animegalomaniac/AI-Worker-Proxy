export class ProxyError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
    public code?: string
  ) {
    super(message);
    this.name = 'ProxyError';
  }
}

export function createErrorResponse(error: unknown): Response {
  if (error instanceof ProxyError) {
    return new Response(
      JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error',
          code: error.code,
        },
      }),
      {
        status: error.statusCode,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
  }

  // Generic error
  const message = error instanceof Error ? error.message : 'Unknown error occurred';
  return new Response(
    JSON.stringify({
      error: {
        message,
        type: 'internal_error',
      },
    }),
    {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );
}

interface ErrorLike {
  status?: number;
  statusCode?: number;
  message?: string;
  code?: string;
}

export function isRateLimitError(error: unknown): boolean {
  const e = error as ErrorLike;
  return (
    e.status === 429 ||
    e.statusCode === 429 ||
    (typeof e.message === 'string' && e.message.toLowerCase().includes('rate limit')) ||
    e.code === 'rate_limit_exceeded'
  );
}

export function isRetryableError(error: unknown): boolean {
  const e = error as ErrorLike;
  return (
    isRateLimitError(error) ||
    e.status === 503 ||
    e.statusCode === 503 ||
    e.status === 502 ||
    e.statusCode === 502 ||
    e.status === 504 ||
    e.statusCode === 504 ||
    e.status === 408 ||
    e.statusCode === 408 ||
    (typeof e.message === 'string' && e.message.toLowerCase().includes('timeout')) ||
    (typeof e.message === 'string' && e.message.toLowerCase().includes('overloaded')) ||
    // Connection-level failures (SDK APIConnectionError, Workers fetch errors) carry
    // no HTTP status — recognize them so key rotation can absorb transient blips
    (typeof e.message === 'string' &&
      /connection|fetch failed|econnreset|etimedout|socket hang up/i.test(e.message))
  );
}

/**
 * Single source of truth for "is this failure worth rotating to the next API key".
 * Returns true → try next key; false → give up on this provider.
 */
export function shouldRotateKey(
  statusCode: number | undefined,
  errorMessage?: string
): boolean {
  if (statusCode !== undefined && [429, 502, 503, 504, 408].includes(statusCode)) {
    return true;
  }
  return isRetryableError({ message: errorMessage });
}

/**
 * Default timeout for external API calls (milliseconds).
 * Non-streaming completions with large max_tokens routinely take 30–60s+, so a
 * short timeout kills legitimate requests (waiting on upstream I/O does not count
 * against the Workers CPU limit). Override with the PROVIDER_TIMEOUT_MS env var.
 */
const PROVIDER_TIMEOUT_MS = 120000;

/**
 * Wrap a promise with a timeout, throwing on expiry.
 * The underlying operation continues running but its result is discarded.
 */
export function withTimeout<T>(promise: Promise<T>, ms = PROVIDER_TIMEOUT_MS): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Provider timeout after ${ms}ms`)), ms);
    promise.then(
      (val) => {
        clearTimeout(timer);
        resolve(val);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}
