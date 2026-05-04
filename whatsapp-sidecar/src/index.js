'use strict';

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const express = require('express');

const { createMessageHandler } = require('./message_handler');
const { createHttpClient } = require('./pipeline_client');
const { extractUrls } = require('./url_extractor');

const PORT = parseInt(process.env.SIDECAR_PORT || '3000', 10);
const PIPELINE_URL = process.env.PIPELINE_URL || 'http://app:8000';
const COMMUNITY_IDS = (process.env.COMMUNITY_IDS || '').split(',').filter(Boolean);

// ---------------------------------------------------------------------------
// Express health/status API
// ---------------------------------------------------------------------------
const httpApp = express();
httpApp.use(express.json());

let _clientReady = false;

httpApp.get('/health', (_req, res) => {
  res.json({ status: 'ok', whatsapp_ready: _clientReady });
});

httpApp.get('/status', (_req, res) => {
  res.json({
    whatsapp_ready: _clientReady,
    community_ids: COMMUNITY_IDS,
    pipeline_url: PIPELINE_URL,
  });
});

// ---------------------------------------------------------------------------
// WhatsApp client
// ---------------------------------------------------------------------------
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: process.env.WA_SESSION_DIR || '.wwebjs_auth' }),
  puppeteer: {
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  },
});

const pipelineClient = createHttpClient(PIPELINE_URL);
const handleMessage = createMessageHandler(pipelineClient, extractUrls, COMMUNITY_IDS);

client.on('qr', (qr) => {
  console.log('Scan the QR code below to authenticate WhatsApp:');
  qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
  _clientReady = true;
  console.log(`WhatsApp client is ready. Monitoring ${COMMUNITY_IDS.length} community ID(s).`);
});

client.on('disconnected', (reason) => {
  _clientReady = false;
  console.warn(`WhatsApp disconnected: ${reason}`);
});

client.on('message_create', async (msg) => {
  try {
    await handleMessage(msg);
  } catch (err) {
    console.error('Unhandled error in message handler:', err);
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
httpApp.listen(PORT, () => {
  console.log(`Sidecar HTTP server listening on :${PORT}`);
});

client.initialize().catch((err) => {
  console.error('Failed to initialise WhatsApp client:', err);
  process.exit(1);
});
