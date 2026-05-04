'use strict';

/**
 * Lightweight HTTP client for submitting video URLs to the Python pipeline.
 * Uses the built-in `fetch` API (Node >= 18) — no axios dependency.
 */

const DEFAULT_TIMEOUT_MS = 10_000;

/**
 * @param {string} baseUrl  Base URL of the Python app (e.g. "http://app:8000")
 * @param {object} [opts]
 * @param {number} [opts.timeoutMs]
 * @returns {{ submitUrl: Function, healthCheck: Function }}
 */
function createHttpClient(baseUrl, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const base = baseUrl.replace(/\/$/, '');

  /**
   * Submit a single video URL to the pipeline.
   * @param {string} url
   * @param {string|null} [sourceGroup]
   * @returns {Promise<{ video_id: number, canonical_url: string, platform: string, is_duplicate: boolean }>}
   * @throws {Error} on non-2xx response or network failure
   */
  async function submitUrl(url, sourceGroup = null) {
    if (!url || typeof url !== 'string') {
      throw new TypeError(`submitUrl: url must be a non-empty string, got ${JSON.stringify(url)}`);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const body = JSON.stringify({ url, source_group: sourceGroup });
      const response = await fetch(`${base}/api/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(`Pipeline returned HTTP ${response.status}: ${text}`);
      }

      return await response.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(`submitUrl timed out after ${timeoutMs}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Ping the pipeline health endpoint.
   * @returns {Promise<boolean>}  true if healthy
   */
  async function healthCheck() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${base}/health`, { signal: controller.signal });
      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  return { submitUrl, healthCheck };
}

module.exports = { createHttpClient, DEFAULT_TIMEOUT_MS };
