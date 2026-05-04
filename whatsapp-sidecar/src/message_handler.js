'use strict';

/**
 * Factory for the WhatsApp message handler.
 *
 * @param {object} pipelineClient     Result of createHttpClient()
 * @param {Function} extractUrls      URL extractor function
 * @param {string[]} communityIds     Allowed community IDs (empty = allow all)
 * @returns {Function}  async (msg) => void
 */
function createMessageHandler(pipelineClient, extractUrls, communityIds = []) {
  const allowAll = communityIds.length === 0;

  return async function handleMessage(msg) {
    // Ignore non-text and status messages
    if (!msg.body || msg.isStatus) {
      return;
    }

    // Community-ID filtering (if configured)
    if (!allowAll) {
      const chatId = msg.id?.remote || msg.from;
      const inCommunity = communityIds.some((id) => chatId && chatId.includes(id));
      if (!inCommunity) {
        return;
      }
    }

    let urls;
    try {
      urls = extractUrls(msg.body);
    } catch (err) {
      console.error('URL extraction failed:', err.message);
      return;
    }

    if (urls.length === 0) {
      return;
    }

    const sourceGroup = msg.id?.remote || msg.from || null;

    for (const { url, platform } of urls) {
      try {
        const result = await pipelineClient.submitUrl(url, sourceGroup);
        if (result.is_duplicate) {
          console.log(`[${platform}] Duplicate — skipped: ${url}`);
        } else {
          console.log(`[${platform}] Queued video #${result.video_id}: ${url}`);
        }
      } catch (err) {
        console.error(`[${platform}] Failed to submit ${url}: ${err.message}`);
      }
    }
  };
}

module.exports = { createMessageHandler };
