# HTTP caching

- Cache-Control: max-age=N lets a cache reuse a response for N seconds without asking the origin.
- Cache validation: an ETag lets a client send If-None-Match, and the server answers 304 Not Modified when nothing changed.
- A response that varies by a request header (for example Origin or Accept-Language) must list it in the Vary header, or a shared cache can serve it to the wrong client.
- Cache-Control: no-store forbids caching entirely; private allows only the browser cache, not shared caches.
