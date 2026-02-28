# Database Guide

## Supported Databases

The system supports PostgreSQL (recommended for production) and SQLite (for development).

## Migrations

The system supports Alembic migrations for PostgreSQL and SQLite. Run migrations with:

```bash
alembic upgrade head
```

### Creating a New Migration

```bash
alembic revision --autogenerate -m "description of change"
```

Always review auto-generated migrations before applying them.

## Schema Overview

### documents table
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| filename | VARCHAR(255) | Original file name |
| content_hash | VARCHAR(64) | SHA-256 of file content |
| created_at | TIMESTAMP | Upload timestamp |
| chunk_count | INTEGER | Number of chunks generated |

### chunks table
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| document_id | UUID | Foreign key to documents |
| text | TEXT | Chunk content |
| embedding | VECTOR(384) | Dense vector embedding |
| position | INTEGER | Position within document |

## Backup and Recovery

For PostgreSQL, use `pg_dump` for logical backups:
```bash
pg_dump -Fc documind > backup.dump
pg_restore -d documind backup.dump
```
