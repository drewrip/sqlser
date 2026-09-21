# `sqlser`

`LogicalPlan` → SQL for DataFusion 55, as a drop-in replacement for
`datafusion::sql::unparser::Unparser`.

```rust
use sqlser::{Unparser, dialect::DuckDBDialect};

let sql = Unparser::new(&DuckDBDialect::new()).plan_to_sql(&plan)?;
```

## Why

`BROKEN.md` at the repo root measures the DataFusion unparser on a 210-case
corpus: **100 cases (48%) produce SQL that DuckDB rejects, or that silently
returns the wrong answer**, once the plan has been through the optimizer. Four
of the thirteen failure modes raise no error at all.

The failures share two causes, and this crate is built to make both
unrepresentable rather than to patch them one at a time.

**Scope.** A `LogicalPlan` qualifies a column by the schema of the node that
produced it. SQL scoping is lexical: once a subtree is wrapped in
`FROM (SELECT …)`, the enclosing query can no longer say `customer.c_custkey`,
because `customer` is not a relation out there. The unparser wraps without
rewriting the qualifiers — 48 cases. So here a `Scope` travels with every
lowered relation, saying for each field of the plan's schema *by index* how to
address that value at this point, and the only way a derived table is created
is by **sealing** a builder, which rebuilds the scope against a freshly
generated alias.

**Clause slots.** The unparser assigns clauses. `Sort.fetch` overwrites a
`LIMIT` already written; a semi-join's `EXISTS` replaces the `WHERE` its left
input contributed. So here `WHERE`, `HAVING`, `QUALIFY` and join `ON` are
accumulators **with no setter**, and `LIMIT` is write-once with a `Sort`'s
fetch held in a separate non-clause field. When a write conflicts, the builder
seals instead of overwriting.

## Guarantees

- Every derived table is aliased, and every emitted qualifier names a relation
  that is in scope.
- Output column count, order and names match `plan.schema()`.
- A plan with no faithful SQL rendering is an `Err`, never approximate SQL.
  This matters more than coverage: the costliest failures in `BROKEN.md` are
  the ones that returned `Ok`.

## Results

Measured on the same 156-case corpus (the 22 TPC-H queries, 95 feature-stress
cases, 39 reductions and bisection probes), executed against DuckDB 1.5.0 and a
live PostgreSQL, rows *and* column names *and* column order compared against
the original query:

| | DataFusion `Unparser` | `sqlser` |
|---|---|---|
| broken on optimized plans | 100/210 (48%) | **0** |
| silent wrong answers | 4 modes | **0** |
| cannot re-plan its own output | 24/210 | **0** (see below) |

All thirteen bugs have a named regression test in `tests/regressions.rs`; the
four silent ones are additionally proven by executing both queries and diffing
the results in `tests/execute.rs`.

## Dialects

`Default`, `PostgreSql`, `DuckDB`, `MySql`, `Sqlite`, `BigQuery`, `Snowflake`,
and `CustomDialect` + `CustomDialectBuilder` — the same set `Unparser` has.
Expression rendering is delegated to DataFusion's `expr_to_sql`, which
`BROKEN.md` measured as clean, so the whole literal/cast/interval/function
surface keeps parity for free. Two corrections are applied inside the dialect,
where a caller cannot forget them:

- **`btrim`.** DataFusion's `trim` UDF is *named* `btrim`, and DuckDB has no
  such function. The `DuckDBDialect` rewrites it.
- **Integer division.** DuckDB's `//` truncates toward zero on integers and
  divides normally on decimals, matching DataFusion exactly — so it is the
  default. DataFusion 55 cannot parse `//` back, so callers that re-plan their
  own output want
  `DuckDBDialect::new().with_division_style(DivisionStyle::TruncCast)`, which
  emits `CAST(trunc(a / b) AS BIGINT)` for the divisions the plan's own types
  show to be integral, and `/` for the rest. That takes round-trip failures to
  zero.

## Running the corpus

```sh
# one-time DuckDB fixture; the live Postgres already carries TPC-H
duckdb tpch.duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)"

cargo test -p sqlser                      # unit + regression + execution tests

cargo run -p sqlser --example sqlser_eval -- \
    tools/unparser_eval/schema.json tools/sqlser_eval/cases.json out.json duckdb
TPCH_DB=tpch.duckdb python3 tools/sqlser_eval/check.py \
    --engine duckdb tools/sqlser_eval/cases.json out.json
python3 tools/sqlser_eval/check.py --engine postgres tools/sqlser_eval/cases.json out.json
```

`cargo test` skips the execution tests when an engine is unreachable, so a
checkout without fixtures stays green.

## Known divergences

Three corpus baselines mean something different from the plan DataFusion builds
for them, and are annotated as such in `tools/sqlser_eval/cases.json` rather
than quietly passed:

- `l_orderkey / 2` — DataFusion plans integer `/` as integer division; the
  DuckDB baseline's `/` is float division. `sqlser` renders the plan.
- `date_trunc('month', <date>)` — DataFusion types this as
  timestamp-without-tz and inserts the cast; Postgres resolves the same call to
  `timestamptz`.
- Baselines written in DuckDB-only syntax (`//`, `#`) that DataFusion cannot
  parse, so there is no plan to lower.

## Not implemented

`Explain`, `Analyze`, DDL, DML, `Copy`, `DescribeTable`, `Statement` and
`Extension` return `SqlserError::Unsupported`, as do struct unnesting and a
null-aware anti join with more than one key. These are errors by design: a
clean refusal is what U12 and U13 got right.
