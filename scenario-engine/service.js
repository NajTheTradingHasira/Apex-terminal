import { replay } from './replay.js';
import { ms, iso } from './time.js';

/** Stateless app integration: reconstruct the same canonical checkpoint history per request. */
export function evaluateThrough(dataset, at, config = {}) {
  if (!Number.isFinite(ms(at))) throw new Error('Valid checkpoint timestamp required');
  const start = ms(dataset.session?.open) - 300000;
  if (!Number.isFinite(start)) throw new Error('Session open required');
  const times = [];
  if (ms(at) < start) times.push(at);
  else {
    // Bound work to one session; do not accidentally replay years from malformed input.
    if (ms(at) - start > 86400000) throw new Error('Checkpoint must be within the session date');
    for (let t = start; t <= ms(at); t += 300000) times.push(iso(t));
    if (ms(times.at(-1)) !== ms(at)) times.push(at);
  }
  const result = replay(dataset, { checkpoints: times, config });
  return result.cards.at(-1);
}
