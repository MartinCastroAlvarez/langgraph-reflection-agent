# Idempotency keys

- An idempotency key is a unique client-generated value sent with a request (often an HTTP header) so the server can recognise retries of the same operation.
- Idempotency: the server stores the key with the first response and returns that stored response for any retry with the same key, instead of performing the operation twice.
- Idempotency keys matter for payments: a timeout followed by a retry must not charge the customer twice.
- Idempotency keys should expire after a retention window (for example 24 hours) and be scoped to one client or account.
- Idempotency: a retry with the same key but a different request body should be rejected as a conflict, not silently accepted.
