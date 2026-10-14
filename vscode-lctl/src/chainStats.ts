// Pure helpers for chain statistics. No `vscode` import, so node:test can load the
// compiled output without a VS Code host.

export interface StatEvent {
    seq?: number;
    type: string;
    data?: Record<string, unknown>;
}

export interface TokenCounts {
    input: number;
    output: number;
}

export interface Pricing {
    input: number;
    output: number;
}

// Model pricing (USD per 1M tokens)
export const MODEL_PRICING: Record<string, Pricing> = {
    'claude-opus-4.5': { input: 5.0, output: 25.0 },
    'claude-sonnet-4': { input: 3.0, output: 15.0 },
    'claude-haiku-3': { input: 0.25, output: 1.25 },
    'gpt-4': { input: 30.0, output: 60.0 },
    'gpt-4-turbo': { input: 10.0, output: 30.0 },
    'gpt-3.5-turbo': { input: 0.5, output: 1.5 },
    'default': { input: 3.0, output: 15.0 },
};

export function pricingFor(model: string | undefined): Pricing {
    return (model !== undefined && Object.hasOwn(MODEL_PRICING, model)
        ? MODEL_PRICING[model]
        : MODEL_PRICING['default']);
}

function num(value: unknown): number {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
}

/**
 * Tokens of one event. The Python library writes `data.tokens = {input, output}`;
 * older traces used flat `tokens_in` / `tokens_out`.
 */
export function eventTokens(data: Record<string, unknown>): TokenCounts {
    const nested = data.tokens;
    if (nested && typeof nested === 'object') {
        const t = nested as Record<string, unknown>;
        return { input: num(t.input ?? t.in), output: num(t.output ?? t.out) };
    }
    return { input: num(data.tokens_in), output: num(data.tokens_out) };
}

/** Total tokens of a chain. Like the library's State, only step_end events count. */
export function chainTokens(events: readonly StatEvent[]): TokenCounts {
    const total: TokenCounts = { input: 0, output: 0 };
    for (const event of events) {
        if (event.type === 'step_end' && event.data) {
            const t = eventTokens(event.data);
            total.input += t.input;
            total.output += t.output;
        }
    }
    return total;
}

export function chainCost(tokens: TokenCounts, pricing: Pricing): number {
    return (tokens.input / 1_000_000) * pricing.input + (tokens.output / 1_000_000) * pricing.output;
}

/** First `max` events; a non-numeric `max` means no limit, a negative one means none. */
export function limitEvents<T>(events: readonly T[], max: number): T[] {
    if (Number.isNaN(max)) {
        return events.slice();
    }
    return events.slice(0, Math.max(0, Math.floor(max)));
}
