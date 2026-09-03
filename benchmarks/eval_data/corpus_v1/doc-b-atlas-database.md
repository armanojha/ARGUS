# Atlas Database Engine — Technical Specification

Atlas is a self-hosted SQL database engine optimized for analytical workloads. It is not a NoSQL store and it does not use a document model.

Key technical facts:
- Default storage engine: columnar ("Atlas Columnar Engine", version 4.2)
- Consistency model: strong consistency by default; eventual consistency is available only in the "Atlas Cloud" deployment mode
- Ingestion model: Atlas is pull-based; it reads from source systems on a schedule rather than being written to directly
- Concurrency: Atlas uses optimistic concurrency control (OCC); it does not use two-phase locking under normal operation
- Partitioning: range partitioning, default 64 partitions
- Compression: zstd is used for warm data; LZ4 is used for hot blocks in memory
- Connection limit: the default maximum connections is 400 per node

Important naming note: Atlas "Delta Sync" and Atlas "Incremental Sync" are DIFFERENT features. Delta Sync copies only changed range partitions. Incremental Sync applies append-only log records. They are not interchangeable.

Atlas ships a CLI tool whose command for rebuilding an index is `atlas index rebuild`, not `atlas reindex`. The command name is frequently confused.