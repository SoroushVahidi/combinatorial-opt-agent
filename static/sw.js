/**
 * Service Worker for Combinatorial Optimization Bot PWA.
 *
 * Strategy: network-first for API/search routes, cache-first for static assets.
 * Offline fallback page shown when the network is unavailable.
 */

const CACHE_NAME = "coa-bot-v1";

// Assets pre-cached on install (app shell)
const PRECACHE_URLS = [
  "/",
  "/offline",
  "/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-180.png",
  "/static/icons/icon-512.png",
];

// ── Install ──────────────────────────────────────────────────────────────────
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

// ── Activate: clean up old caches ────────────────────────────────────────────
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

// ── Fetch: network-first with cache fallback ─────────────────────────────────
self.addEventListener("fetch", (event) => {
  // Only handle GET requests for same-origin resources.
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  // For static assets, use cache-first.
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) =>
          cached ||
          fetch(event.request).then((response) => {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
            return response;
          })
      )
    );
    return;
  }

  // For the main page, use network-first; fall back to cache, then offline page.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Cache a fresh copy of the main page.
        if (url.pathname === "/" && response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
        }
        return response;
      })
      .catch(() =>
        caches.match(event.request).then(
          (cached) => cached || caches.match("/offline").then(
            (offlinePage) =>
              offlinePage ||
              new Response("Offline — please reconnect and reload.", {
                status: 503,
                headers: { "Content-Type": "text/plain" },
              })
          )
        )
      )
  );
});
