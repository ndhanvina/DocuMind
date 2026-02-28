# Configuration Guide

## Environment Variables

All configuration is done through environment variables. Copy `.env.example` to `.env` and adjust values.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///data.db` | Database connection string |
| `SECRET_KEY` | (required) | Secret key for JWT signing |
| `DEBUG` | `false` | Enable debug mode |

### Rate Limiting

Rate limiting is configured via the `RATE_LIMIT_PER_MINUTE` environment variable. The default
is 60 requests per minute per API key. Set to `0` to disable rate limiting.

Rate limits are enforced using a sliding window algorithm. When the limit is exceeded, the API
returns HTTP 429 with a `Retry-After` header.

### Logging

Set `LOG_LEVEL` to one of: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default is `INFO`.
Structured JSON logging is enabled by default. Set `LOG_FORMAT=text` for plain text output.

### CORS

Configure allowed origins with `CORS_ORIGINS` as a comma-separated list:
```
CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

## Docker Deployment

```bash
docker build -t documind .
docker run -p 8000:8000 --env-file .env documind
```
