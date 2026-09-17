// SatQuery AI — Progressive Web App Service Worker (v1.0.0)
// High-performance caching: Shell pre-cache, Leaflet Map Tiles LRU, Stale-While-Revalidate

const VERSION = 'v1.0.0';
const STATIC_CACHE = `satquery-static-${VERSION}`;
const TILE_CACHE = `satquery-tiles-${VERSION}`;
const PAGES_CACHE = `satquery-pages-${VERSION}`;

const PRECACHE_ASSETS = [
  '/',
  '/offline',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
  '/icons/maskable-icon-512x512.png',
  '/icons/apple-touch-icon.png',
  '/icons/icon.svg',
  '/asteroid.jpg',
  '/planet.jpg'
];

const MAX_TILES = 300; // Limit cached map tiles to conserve device storage

// Trim cache helper (LRU FIFO)
async function trimCache(cacheName, maxItems) {
  try {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    if (keys.length > maxItems) {
      await cache.delete(keys[0]);
      trimCache(cacheName, maxItems);
    }
  } catch (err) {
    console.warn('[SW] trimCache error:', err);
  }
}

// 1. Install: Precache shell & offline fallback
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then(async (cache) => {
      console.log('[SW] Pre-caching core application shell & offline page');
      try {
        await cache.addAll(PRECACHE_ASSETS);
      } catch (err) {
        console.warn('[SW] Some precache assets skipped:', err);
      }
    }).then(() => self.skipWaiting())
  );
});

// 2. Activate: Clean stale caches & claim clients
self.addEventListener('activate', (event) => {
  const currentCaches = [STATIC_CACHE, TILE_CACHE, PAGES_CACHE];
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (!currentCaches.includes(name)) {
            console.log('[SW] Purging outdated cache:', name);
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// 3. Fetch Event Interceptor
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle HTTP/HTTPS GET requests
  if (request.method !== 'GET' || !url.protocol.startsWith('http')) {
    return;
  }

  // A. Backend API Requests (Network-Only with Graceful Offline JSON)
  if (
    url.pathname.startsWith('/api/') ||
    url.hostname.includes('onrender.com') ||
    url.hostname.includes('hf.space') ||
    url.port === '8000'
  ) {
    event.respondWith(
      fetch(request).catch(() => {
        return new Response(
          JSON.stringify({
            status: 'offline',
            offline: true,
            message: 'You are currently offline. Satellite AI inference requires an active connection.'
          }),
          {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
          }
        );
      })
    );
    return;
  }

  // B. Satellite Map Raster Tiles (OpenStreetMap, CartoDB, Bhuvan, Stamen) -> Cache-First
  const isMapTile =
    url.hostname.includes('tile.openstreetmap.org') ||
    url.hostname.includes('basemaps.cartocdn.com') ||
    url.hostname.includes('bhuvan') ||
    url.pathname.includes('/tiles/') ||
    url.pathname.match(/\/\d+\/\d+\/\d+\.(png|jpg|webp|jpeg)/i);

  if (isMapTile) {
    event.respondWith(
      caches.open(TILE_CACHE).then(async (cache) => {
        const cachedResponse = await cache.match(request);
        if (cachedResponse) {
          return cachedResponse;
        }
        try {
          const networkResponse = await fetch(request);
          if (networkResponse && networkResponse.status === 200) {
            cache.put(request, networkResponse.clone());
            trimCache(TILE_CACHE, MAX_TILES);
          }
          return networkResponse;
        } catch (err) {
          // If offline and tile not cached, return empty transparent 256x256 png or fail gracefully
          return new Response('', { status: 408, statusText: 'Tile Unavailable Offline' });
        }
      })
    );
    return;
  }

  // C. HTML Navigation Requests -> Network-First, fallback to Cache, fallback to /offline
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response && response.status === 200) {
            const clone = response.clone();
            caches.open(PAGES_CACHE).then((cache) => cache.put(request, clone));
          }
          return response;
        })
        .catch(async () => {
          const cachedPage = await caches.match(request);
          if (cachedPage) return cachedPage;

          const offlinePage = await caches.match('/offline');
          return offlinePage || new Response('Offline - SatQuery AI', { status: 200, headers: { 'Content-Type': 'text/html' } });
        })
    );
    return;
  }

  // D. Static Next.js Bundles, Fonts, and Media -> Stale-While-Revalidate
  const isStaticAsset =
    url.pathname.startsWith('/_next/static/') ||
    url.pathname.startsWith('/icons/') ||
    url.pathname.match(/\.(js|css|woff2?|png|jpe?g|svg|webp|ico)$/i);

  if (isStaticAsset) {
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        const fetchPromise = fetch(request)
          .then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              caches.open(STATIC_CACHE).then((cache) => {
                cache.put(request, networkResponse.clone());
              });
            }
            return networkResponse;
          })
          .catch(() => cachedResponse);

        return cachedResponse || fetchPromise;
      })
    );
    return;
  }

  // E. Default: Network with Cache Fallback
  event.respondWith(
    fetch(request).catch(() => caches.match(request))
  );
});

// 4. Message Listener (Support skipWaiting updates)
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
