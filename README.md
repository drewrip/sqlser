# `sqlser`

`LogicalPlan` → SQL for DataFusion 55. A drop-in replacement for
`datafusion::sql::unparser::Unparser`, which often emits invalid or silently
wrong SQL for optimized plans. `sqlser` returns an `Err` instead of
approximate SQL.

## Usage

```toml
[dependencies]
sqlser = { git = "https://github.com/drewrip/sqlser" }
```

```rust
use sqlser::{Unparser, dialect::DuckDBDialect};

let sql = Unparser::new(&DuckDBDialect::new()).plan_to_sql(&plan)?;
```

Dialects: `Default`, `PostgreSql`, `DuckDB`, `MySql`, `Sqlite`, `BigQuery`,
`Snowflake`, `CustomDialect`.

## Testing

```sh
cargo test
```

Execution tests run against DuckDB (`TPCH_DB`, default `tpch.duckdb`) and
PostgreSQL (`PG_DSN`, `PG_SCHEMA`), and are skipped when an engine is
unreachable.
