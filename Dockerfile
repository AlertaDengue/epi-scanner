FROM node:24-alpine AS base

FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json ./
RUN npm install

FROM deps AS builder
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

ARG UID=1000
ARG GID=1000
ARG FRONTEND_PORT=3000

RUN if ! getent group ${GID} >/dev/null; then \
      addgroup --gid ${GID} nodejs; \
    fi && \
    if ! getent passwd ${UID} >/dev/null; then \
      adduser --uid ${UID} --gid ${GID} --disabled-password nextjs; \
    fi

COPY --from=builder /app/public ./public

RUN mkdir .next
RUN chown -R ${UID}:${GID} .next

COPY --from=builder --chown=${UID}:${GID} /app/.next/standalone ./
COPY --from=builder --chown=${UID}:${GID} /app/.next/static ./.next/static

RUN mkdir -p data && chown ${UID}:${GID} data

USER ${UID}

EXPOSE ${FRONTEND_PORT}

ENV PORT=${FRONTEND_PORT}
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
