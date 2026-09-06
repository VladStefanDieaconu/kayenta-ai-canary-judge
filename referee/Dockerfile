# Referee: standalone Kayenta UI (Nike-Inc/referee), built from source.
# No official image exists; this follows the known-good community recipe.
# The repo is archived and uses an old toolchain (node:10). On Apple Silicon
# the compose service sets platform: linux/amd64 (see docker-compose / README).
#
# If the upstream master.zip ever breaks on dependency drift, pin REFEREE_REF
# to a specific commit zip instead of the master branch.
FROM node:10

# Arch-aware tini. The generic "tini" release asset is amd64-only; on arm64 it
# triggers a rosetta error at runtime. Detect the image's real architecture at
# build time (dpkg arch names match tini's asset suffixes: amd64 / arm64), so
# the build stays native on both Apple Silicon and x86 hosts.
ENV TINI_VERSION=v0.19.0
RUN ARCH="$(dpkg --print-architecture)" \
 && curl -fsSL "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${ARCH}" -o /tini \
 && chmod +x /tini
ENTRYPOINT ["/tini", "--"]

RUN mkdir -p /app

# Fetch Referee source (master). curl + unzip are present in node:10 (debian).
RUN curl -L https://github.com/Nike-Inc/referee/archive/refs/heads/master.zip -o referee.zip \
 && unzip -o referee.zip -d /app \
 && rm -f referee.zip

WORKDIR /app/referee-master

# Strip the React dev-only "proxy" field (recipe requirement, keep this step).
RUN sed -i '/proxy/d' /app/referee-master/packages/client/package.json

RUN yarn install && yarn bootstrap && yarn build

EXPOSE 3001

# Express server; reads KAYENTA_BASE_URL (set on the compose service).
CMD ["yarn", "server-start"]
