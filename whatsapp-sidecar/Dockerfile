FROM node:20-slim AS runtime

# Chromium dependencies for puppeteer / whatsapp-web.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src/ ./src/

VOLUME ["/app/.wwebjs_auth"]

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD node -e "fetch('http://localhost:3000/health').then(r=>r.ok?process.exit(0):process.exit(1)).catch(()=>process.exit(1))"

CMD ["node", "src/index.js"]
