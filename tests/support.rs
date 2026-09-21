//! Shared test scaffolding: a TPC-H `SessionContext` and plan helpers.
#![allow(dead_code)]

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::empty::EmptyTable;
use datafusion::datasource::view::ViewTable;
use datafusion::logical_expr::{Expr, LogicalPlan, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

/// The TPC-H schema, as both DuckDB and the Postgres fixture carry it.
pub fn tpch_tables() -> Vec<(&'static str, Vec<(&'static str, DataType)>)> {
    use DataType::*;
    let dec = Decimal128(15, 2);
    vec![
        (
            "region",
            vec![
                ("r_regionkey", Int32),
                ("r_name", Utf8),
                ("r_comment", Utf8),
            ],
        ),
        (
            "nation",
            vec![
                ("n_nationkey", Int32),
                ("n_name", Utf8),
                ("n_regionkey", Int32),
                ("n_comment", Utf8),
            ],
        ),
        (
            "supplier",
            vec![
                ("s_suppkey", Int32),
                ("s_name", Utf8),
                ("s_address", Utf8),
                ("s_nationkey", Int32),
                ("s_phone", Utf8),
                ("s_acctbal", dec.clone()),
                ("s_comment", Utf8),
            ],
        ),
        (
            "customer",
            vec![
                ("c_custkey", Int32),
                ("c_name", Utf8),
                ("c_address", Utf8),
                ("c_nationkey", Int32),
                ("c_phone", Utf8),
                ("c_acctbal", dec.clone()),
                ("c_mktsegment", Utf8),
                ("c_comment", Utf8),
            ],
        ),
        (
            "part",
            vec![
                ("p_partkey", Int32),
                ("p_name", Utf8),
                ("p_mfgr", Utf8),
                ("p_brand", Utf8),
                ("p_type", Utf8),
                ("p_size", Int32),
                ("p_container", Utf8),
                ("p_retailprice", dec.clone()),
                ("p_comment", Utf8),
            ],
        ),
        (
            "partsupp",
            vec![
                ("ps_partkey", Int32),
                ("ps_suppkey", Int32),
                ("ps_availqty", Int32),
                ("ps_supplycost", dec.clone()),
                ("ps_comment", Utf8),
            ],
        ),
        (
            "orders",
            vec![
                ("o_orderkey", Int32),
                ("o_custkey", Int32),
                ("o_orderstatus", Utf8),
                ("o_totalprice", dec.clone()),
                ("o_orderdate", Date32),
                ("o_orderpriority", Utf8),
                ("o_clerk", Utf8),
                ("o_shippriority", Int32),
                ("o_comment", Utf8),
            ],
        ),
        (
            "lineitem",
            vec![
                ("l_orderkey", Int32),
                ("l_partkey", Int32),
                ("l_suppkey", Int32),
                ("l_linenumber", Int32),
                ("l_quantity", dec.clone()),
                ("l_extendedprice", dec.clone()),
                ("l_discount", dec.clone()),
                ("l_tax", dec.clone()),
                ("l_returnflag", Utf8),
                ("l_linestatus", Utf8),
                ("l_shipdate", Date32),
                ("l_commitdate", Date32),
                ("l_receiptdate", Date32),
                ("l_shipinstruct", Utf8),
                ("l_shipmode", Utf8),
                ("l_comment", Utf8),
            ],
        ),
    ]
}

/// Mirrors `dee`'s `OpaqueScanTable`: claims exact pushdown for every filter,
/// so plans carry `TableScan.filters` the way `dee`'s do.
#[derive(Debug)]
pub struct OpaqueScanTable {
    pub schema: SchemaRef,
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

pub async fn ctx() -> SessionContext {
    ctx_with(false, &[]).await
}

pub async fn ctx_opaque() -> SessionContext {
    ctx_with(true, &[]).await
}

pub async fn ctx_with(opaque: bool, views: &[(&str, &str)]) -> SessionContext {
    let ctx = SessionContext::new();
    for (name, cols) in tpch_tables() {
        let schema = Arc::new(Schema::new(
            cols.into_iter()
                .map(|(n, t)| Field::new(n, t, true))
                .collect::<Vec<_>>(),
        ));
        let provider: Arc<dyn TableProvider> = if opaque {
            Arc::new(OpaqueScanTable { schema })
        } else {
            Arc::new(EmptyTable::new(schema))
        };
        ctx.register_table(name, provider).unwrap();
    }
    for (name, sql) in views {
        let plan = ctx.state().create_logical_plan(sql).await.unwrap();
        ctx.register_table(*name, Arc::new(ViewTable::new(plan, None)))
            .unwrap();
    }
    ctx
}

/// The raw plan, as written.
pub async fn plan(ctx: &SessionContext, sql: &str) -> LogicalPlan {
    ctx.state().create_logical_plan(sql).await.unwrap()
}

/// The optimized plan — the shape `BROKEN.md` found the old unparser failing
/// on 48% of, and the shape `dee`'s pushdown actually hands to a serializer.
pub async fn opt_plan(ctx: &SessionContext, sql: &str) -> LogicalPlan {
    let p = plan(ctx, sql).await;
    ctx.state().optimize(&p).unwrap()
}
