//! Corpus harness: plan each case, lower it with `sqlser`, record the SQL.
//!
//! Usage: sqlser_eval <schema.json> <cases.json> <out.json> [dialect]
//!
//! Emits, per case, the SQL for both the raw plan and the optimized one. The
//! optimized plan is the interesting column: it is the shape `BROKEN.md`
//! measured DataFusion's unparser failing on 48% of, and the shape a pushdown
//! pass actually hands to a serializer.
//!
//! Execution and result comparison are the Python driver's job
//! (`tools/sqlser_eval/check.py`), because the only conclusive test is running
//! both queries on a real engine and diffing the rows.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::empty::EmptyTable;
use datafusion::datasource::view::ViewTable;
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use serde::{Deserialize, Serialize};
use sqlser::Unparser;
use sqlser::dialect::Dialect;

#[derive(Deserialize)]
struct Col {
    table_name: String,
    column_name: String,
    data_type: String,
}

#[derive(Deserialize, Clone)]
struct ViewDef {
    name: String,
    sql: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    sql: String,
    #[serde(default)]
    views: Vec<ViewDef>,
    #[serde(default)]
    opaque: bool,
    #[serde(default)]
    dialect: Option<String>,
}

#[derive(Serialize, Default)]
struct Out {
    name: String,
    status: String,
    error: Option<String>,
    opt_plan: Option<String>,
    raw_sql: Option<String>,
    raw_err: Option<String>,
    opt_sql: Option<String>,
    opt_err: Option<String>,
    /// Can DataFusion re-plan what we wrote, and does the schema still match?
    opt_roundtrip_err: Option<String>,
    /// Field names of the optimized plan, in order — the output contract.
    opt_schema: Vec<String>,
}

fn duck_type(t: &str) -> DataType {
    let t = t.trim().to_uppercase();
    if let Some(rest) = t.strip_prefix("DECIMAL") {
        let inner = rest.trim_start_matches('(').trim_end_matches(')');
        let mut it = inner.split(',');
        let p: u8 = it.next().unwrap_or("18").trim().parse().unwrap_or(18);
        let s: i8 = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
        return DataType::Decimal128(p, s);
    }
    if let Some(inner) = t.strip_suffix("[]") {
        return DataType::List(Arc::new(Field::new("item", duck_type(inner), true)));
    }
    match t.as_str() {
        "BIGINT" | "INT8" | "LONG" => DataType::Int64,
        "INTEGER" | "INT4" | "INT" | "SIGNED" => DataType::Int32,
        "SMALLINT" | "INT2" => DataType::Int16,
        "TINYINT" | "INT1" => DataType::Int8,
        "BOOLEAN" | "BOOL" => DataType::Boolean,
        "DOUBLE" | "FLOAT8" => DataType::Float64,
        "FLOAT" | "REAL" | "FLOAT4" => DataType::Float32,
        "DATE" => DataType::Date32,
        "TIMESTAMP" | "DATETIME" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "TIME" => DataType::Time64(TimeUnit::Microsecond),
        "BLOB" | "BYTEA" => DataType::Binary,
        _ => DataType::Utf8,
    }
}

fn build_schemas(cols: Vec<Col>) -> Vec<(String, SchemaRef)> {
    let mut out: Vec<(String, Vec<Field>)> = Vec::new();
    for c in cols {
        let f = Field::new(&c.column_name, duck_type(&c.data_type), true);
        match out.iter_mut().find(|(t, _)| *t == c.table_name) {
            Some((_, fs)) => fs.push(f),
            None => out.push((c.table_name.clone(), vec![f])),
        }
    }
    out.into_iter()
        .map(|(t, fs)| (t, Arc::new(Schema::new(fs)) as SchemaRef))
        .collect()
}

/// Mirrors `dee`'s `OpaqueScanTable`: exact pushdown for every filter, so the
/// plan carries `TableScan.filters` rather than separate `Filter` nodes.
#[derive(Debug)]
struct OpaqueScanTable {
    schema: SchemaRef,
}

#[async_trait::async_trait]
impl TableProvider for OpaqueScanTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
    }
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        EmptyTable::new(Arc::clone(&self.schema))
            .scan(state, projection, filters, limit)
            .await
    }
}

async fn make_ctx(
    schemas: &[(String, SchemaRef)],
    views: &[ViewDef],
    opaque: bool,
) -> datafusion::error::Result<SessionContext> {
    let ctx = SessionContext::new();
    for (name, schema) in schemas {
        let provider: Arc<dyn TableProvider> = if opaque {
            Arc::new(OpaqueScanTable {
                schema: Arc::clone(schema),
            })
        } else {
            Arc::new(EmptyTable::new(Arc::clone(schema)))
        };
        ctx.register_table(name.as_str(), provider)?;
    }
    for v in views {
        let plan = ctx.state().create_logical_plan(&v.sql).await?;
        ctx.register_table(v.name.as_str(), Arc::new(ViewTable::new(plan, None)))?;
    }
    Ok(ctx)
}

fn lower(dialect: &dyn Dialect, plan: &LogicalPlan) -> Result<String, String> {
    // A panic here would be a bug, but one bad case should not lose the run.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Unparser::new(dialect).plan_to_sql(plan)
    }))
    .map_err(|_| "PANIC".to_string())?
    .map_err(|e| e.to_string())
}

/// Re-plan our own output and check the schema still lines up — names *and*
/// order, since a rotation (U3) round-trips cleanly if you only count columns.
async fn roundtrip(ctx: &SessionContext, sql: &str, expect: &LogicalPlan) -> Option<String> {
    let replanned = match ctx.state().create_logical_plan(sql).await {
        Ok(p) => p,
        Err(e) => return Some(format!("re-plan: {e}")),
    };
    let got: Vec<&String> = replanned
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .collect();
    let want: Vec<&String> = expect.schema().fields().iter().map(|f| f.name()).collect();
    if got.len() != want.len() {
        return Some(format!("schema drift: {got:?} vs {want:?}"));
    }
    // A plan schema may carry the same name twice (a self-join on `n_name`);
    // a select list may not, or DataFusion refuses to re-plan it at all
    // ("Projections require unique expression names").  The `__N` suffix is
    // the minimum change that makes the output re-plannable, so it counts as
    // a match.
    for (g, w) in got.iter().zip(&want) {
        if g == w || g.split("__").next() == Some(w.as_str()) {
            continue;
        }
        return Some(format!("schema drift: {got:?} vs {want:?}"));
    }
    None
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: sqlser_eval <schema.json> <cases.json> <out.json> [dialect]");
        std::process::exit(2);
    }
    let dialect_name = args.get(4).cloned().unwrap_or_else(|| "duckdb".into());
    // `<name>_rt` selects the round-trip-safe integer-division spelling, for
    // callers like `dee`'s pushdown that re-plan their own output. DataFusion
    // 55 cannot parse DuckDB's `//` back (U11).
    let dialect: Box<dyn Dialect> = match dialect_name.as_str() {
        "duckdb_rt" => Box::new(
            sqlser::dialect::DuckDBDialect::new()
                .with_division_style(sqlser::dialect::DivisionStyle::TruncCast),
        ),
        other => sqlser::dialect_for_db(other).unwrap_or_else(|| panic!("unknown dialect {other}")),
    };

    let cols: Vec<Col> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let schemas = build_schemas(cols);
    let cases: Vec<Case> = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;

    let mut results = Vec::new();
    for case in &cases {
        let mut o = Out {
            name: case.name.clone(),
            status: "ok".into(),
            ..Default::default()
        };
        // A per-case dialect override in the corpus is about the *old*
        // harness; the dialect under test is whatever was asked for.
        let _ = &case.dialect;

        let ctx = match make_ctx(&schemas, &case.views, case.opaque).await {
            Ok(c) => c,
            Err(e) => {
                o.status = "setup_err".into();
                o.error = Some(e.to_string());
                results.push(o);
                continue;
            }
        };
        let raw = match ctx.state().create_logical_plan(&case.sql).await {
            Ok(p) => p,
            Err(e) => {
                o.status = "plan_err".into();
                o.error = Some(e.to_string());
                results.push(o);
                continue;
            }
        };

        match lower(dialect.as_ref(), &raw) {
            Ok(s) => o.raw_sql = Some(s),
            Err(e) => o.raw_err = Some(e),
        }

        match ctx.state().optimize(&raw) {
            Ok(opt) => {
                o.opt_plan = Some(opt.display_indent().to_string());
                o.opt_schema = opt
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                match lower(dialect.as_ref(), &opt) {
                    Ok(s) => {
                        o.opt_roundtrip_err = roundtrip(&ctx, &s, &opt).await;
                        o.opt_sql = Some(s);
                    }
                    Err(e) => o.opt_err = Some(e),
                }
            }
            Err(e) => {
                o.status = "optimize_err".into();
                o.error = Some(e.to_string());
            }
        }
        results.push(o);
    }

    std::fs::write(&args[3], serde_json::to_string_pretty(&results)?)?;
    eprintln!("wrote {} results to {}", results.len(), args[3]);
    Ok(())
}
