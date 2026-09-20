# syntax=docker/dockerfile:1

# PredictIQ frontend image: Next.js standalone production build.
#
# NEXT_PUBLIC_API_BASE_URL is inlined into the client bundle at *build*
# time (Next.js convention for NEXT_PUBLIC_* vars) — it's the URL the
# user's browser calls, not a container-to-container address, so it
# defaults to the backend's host-published port (see docker-compose.yml)
# rather than the "backend" Compose service name.

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8123
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD wget -q -O- http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
