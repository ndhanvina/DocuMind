# API Reference

## Authentication

The API uses JWT-based authentication with refresh tokens. All requests must include a valid
Bearer token in the Authorization header.

### Obtaining a Token

Send a POST request to `/auth/login` with your credentials:

```json
{
  "email": "user@example.com",
  "password": "your-password"
}
```

The response includes an `access_token` (valid for 15 minutes) and a `refresh_token` (valid for 7 days).

### Refreshing Tokens

POST to `/auth/refresh` with the refresh token to obtain a new access token without re-authenticating.

## Endpoints

### GET /api/v1/documents

List all indexed documents. Supports pagination via `page` and `per_page` query parameters.

### POST /api/v1/documents

Upload a new document for indexing. Accepts multipart form data with a `file` field.

### POST /api/v1/query

Submit a natural language query against the document index. Returns an answer with citations.

Request body:
```json
{
  "query": "How do I configure the system?",
  "top_k": 5
}
```

### DELETE /api/v1/documents/{id}

Remove a document from the index. Requires admin role.
