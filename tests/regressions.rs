//! One test per failure in `BROKEN.md`, using its own minimal repro.
//!
//! These assert on the generated SQL. The execution-level proof — that the
//! text also returns the right rows on DuckDB and Postgres — lives in
//! `tests/execute.rs`; these are the fast local guards that say *why* each
//! shape is correct.

mod support;

use sqlser::Unparser;
use sqlser::dialect::{DuckDBDialect, PostgreSqlDialect};
use support::{ctx, ctx_with, opt_plan, plan};

fn duck(plan: &datafusion::logical_expr::LogicalPlan) -> String {
    Unparser::new(&DuckDBDialect::new())
        .plan_to_sql(plan)
        .expect("lowering must succeed")
}

fn pg(plan: &datafusion::logical_expr::LogicalPlan) -> String {
    Unparser::new(&PostgreSqlDialect::new())
        .plan_to_sql(plan)
        .expect("lowering must succeed")
}

/// No emitted qualifier may name a relation that is not in scope. Every
/// relation this crate emits carries a generated alias, so any bare table name
/// used as a qualifier is a leak.
fn no_stale_qualifiers(sql: &str) {
    for table in [
        "customer", "orders", "lineitem", "nation", "region", "supplier", "part", "partsupp",
    ] {
        let stale = format!(r#""{table}"."#);
        assert!(
            !sql.contains(&stale),
            "leaked a plan qualifier `{table}.` into SQL:\n{sql}"
        );
    }
}

// -- U1 ---------------------------------------------------------------------

#[tokio::test]
async fn u1_derived_table_keeps_its_relation_addressable() {
    // BROKEN.md's minimal repro: ORDER BY on a column that is not selected.
    // The old unparser emitted `FROM (SELECT ...)` with no alias and left the
    // outer SELECT saying `"customer"."c_custkey"`.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT c_custkey FROM customer ORDER BY c_acctbal * 2 DESC LIMIT 5",
    )
    .await;
    let sql = duck(&p);
    no_stale_qualifiers(&sql);
    assert!(sql.contains("ORDER BY"), "{sql}");
    assert!(sql.contains("LIMIT 5"), "{sql}");
}

#[tokio::test]
async fn u1_every_derived_table_is_aliased() {
    // Postgres rejects an unaliased derived table at parse time, so this is
    // the stricter statement of the same guarantee.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT * FROM (SELECT * FROM (SELECT c_custkey, c_acctbal FROM customer \
         WHERE c_acctbal > 0) a WHERE c_custkey < 100) b",
    )
    .await;
    let sql = pg(&p);
    no_stale_qualifiers(&sql);
    // Every `FROM (` must be followed, eventually, by an `AS`.
    let opens = sql.matches("FROM (").count();
    let aliases = sql.matches(") AS ").count();
    assert!(
        aliases >= opens,
        "{opens} derived tables but only {aliases} aliases:\n{sql}"
    );
}

// -- U2 ---------------------------------------------------------------------

#[tokio::test]
async fn u2_an_empty_projection_never_emits_an_empty_select_list() {
    // `optimize_projections` proves no column is needed and leaves a
    // `Projection:` with zero expressions. The old unparser emitted
    // `SELECT FROM "customer"`, a parse error.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT count(*) FROM (SELECT c_custkey FROM customer WHERE c_acctbal > 100) x",
    )
    .await;
    let sql = duck(&p);
    assert!(!sql.contains("SELECT FROM"), "empty select list:\n{sql}");
    assert!(sql.to_lowercase().contains("count"), "{sql}");
}

#[tokio::test]
async fn u2_holds_for_the_semi_join_shape_too() {
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT count(*) FROM orders WHERE o_custkey IN (SELECT c_custkey FROM customer)",
    )
    .await;
    let sql = duck(&p);
    assert!(!sql.contains("SELECT FROM"), "{sql}");
}

// -- U3 ---------------------------------------------------------------------

#[tokio::test]
async fn u3_an_aggregate_at_the_root_keeps_its_schema_column_order() {
    // The old unparser chained `aggr_expr` before `group_expr`, rotating the
    // aggregate from last position to first with no error at all.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT c_nationkey, c_mktsegment, sum(c_acctbal) FROM customer \
         GROUP BY c_nationkey, c_mktsegment",
    )
    .await;
    let sql = duck(&p);

    let nat = sql.find("c_nationkey").expect("nationkey in output");
    let seg = sql.find("c_mktsegment").expect("mktsegment in output");
    let agg = sql.find("sum(").expect("sum in output");
    assert!(
        nat < seg && seg < agg,
        "columns must come out in plan-schema order (group keys, then aggregates):\n{sql}"
    );
}

// -- U4 ---------------------------------------------------------------------

#[tokio::test]
async fn u4_a_sorts_fetch_never_becomes_the_querys_limit() {
    // `push_down_limit` sets `Sort.fetch = skip + fetch` = 15. The query
    // returns 10 rows. The old unparser emitted `LIMIT 15 OFFSET 5` — fifteen
    // rows, no error.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT c_custkey FROM customer ORDER BY c_custkey LIMIT 10 OFFSET 5",
    )
    .await;
    let sql = duck(&p);
    assert!(sql.contains("LIMIT 10"), "must return 10 rows:\n{sql}");
    assert!(sql.contains("OFFSET 5"), "{sql}");
    assert!(
        !sql.contains("LIMIT 15"),
        "the sort's internal row count leaked into the output:\n{sql}"
    );
}

#[tokio::test]
async fn u4_holds_for_nested_limits() {
    // Plan: Limit{skip:5, fetch:5} over Sort{fetch:10}. Five rows, not ten.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT c_custkey FROM (SELECT c_custkey FROM customer ORDER BY c_custkey LIMIT 10) t \
         LIMIT 5 OFFSET 5",
    )
    .await;
    let sql = duck(&p);
    assert!(sql.contains("LIMIT 5"), "{sql}");
}

// -- U5 ---------------------------------------------------------------------

#[tokio::test]
async fn u5_an_anti_join_keeps_the_filter_on_its_driving_input() {
    // Reduced from TPC-H Q22. The old unparser replaced the WHERE with the
    // NOT EXISTS instead of AND-ing, losing the IN list: 25 rows where the
    // original gives 3.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT cc, count(*) FROM ( \
           SELECT substring(c_phone FROM 1 FOR 2) AS cc, c_acctbal FROM customer \
           WHERE substring(c_phone FROM 1 FOR 2) IN ('13','31','23') \
             AND NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey) \
         ) t GROUP BY cc",
    )
    .await;
    let sql = duck(&p);
    assert!(
        sql.contains("NOT EXISTS"),
        "the anti join must render:\n{sql}"
    );
    assert!(
        sql.contains("'13'") && sql.contains("'31'") && sql.contains("'23'"),
        "the IN predicate on the driving input must survive:\n{sql}"
    );
}

// -- U6 ---------------------------------------------------------------------

#[tokio::test]
async fn u6_a_cross_join_never_emits_a_wildcard() {
    // The old unparser emitted the columns *and* a trailing `*`, so a
    // five-column result came back with nine and could not be re-planned.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT * FROM nation n, (SELECT count(*) AS c FROM customer) x",
    )
    .await;
    let sql = duck(&p);
    assert!(!sql.contains(", *"), "wildcard re-expansion:\n{sql}");
    assert!(!sql.contains("SELECT *"), "{sql}");
}

#[tokio::test]
async fn u6_output_arity_matches_the_plan_schema() {
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT * FROM nation n, (SELECT count(*) AS c FROM customer) x",
    )
    .await;
    let query = Unparser::new(&DuckDBDialect::new())
        .plan_to_query(&p)
        .unwrap();
    let n = match query.body.as_ref() {
        sqlparser::ast::SetExpr::Select(s) => s.projection.len(),
        _ => panic!("expected a SELECT"),
    };
    assert_eq!(
        n,
        p.schema().fields().len(),
        "output arity must equal the plan schema's"
    );
}

// -- U7 ---------------------------------------------------------------------

#[tokio::test]
async fn u7_stacked_subquery_aliases_do_not_leak_the_inner_name() {
    // The OMP / view-inlining shape: `SubqueryAlias: s` over
    // `SubqueryAlias: stg_cust` over the scan. The old unparser emitted the
    // inner alias (the view name) while every column said `s.` —
    // `Binder Error: Referenced table "s" not found!`.
    let views = [
        (
            "stg_cust",
            "SELECT c_custkey, c_name, c_nationkey, c_acctbal, c_mktsegment FROM customer WHERE c_acctbal > -999",
        ),
        (
            "stg_ord",
            "SELECT o_orderkey, o_custkey, o_totalprice, o_orderdate, o_orderstatus FROM orders WHERE o_orderstatus <> 'X'",
        ),
    ];
    let ctx = ctx_with(false, &views).await;
    let p = opt_plan(
        &ctx,
        "SELECT s.c_nationkey, count(*) AS n, sum(o.o_totalprice) AS t \
         FROM stg_cust s JOIN stg_ord o ON s.c_custkey = o.o_custkey \
         WHERE s.c_mktsegment = 'BUILDING' GROUP BY s.c_nationkey",
    )
    .await;
    let sql = duck(&p);
    no_stale_qualifiers(&sql);
    for leaked in [r#""s"."#, r#""o"."#, r#""stg_cust"."#, r#""stg_ord"."#] {
        assert!(
            !sql.contains(leaked),
            "plan alias `{leaked}` leaked into SQL:\n{sql}"
        );
    }
}

#[tokio::test]
async fn u7_self_joins_keep_their_legs_distinct() {
    // TPC-H Q7's shape: `n1`/`n2` over the same table.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT n1.n_name, n2.n_name FROM nation n1 JOIN nation n2 \
         ON n1.n_regionkey = n2.n_regionkey AND n1.n_nationkey < n2.n_nationkey",
    )
    .await;
    let sql = duck(&p);
    no_stale_qualifiers(&sql);
    // Two distinct generated aliases for the two legs.
    let aliases: std::collections::HashSet<_> = sql
        .match_indices("__sqlser_r")
        .map(|(i, _)| &sql[i..i + 12])
        .collect();
    assert!(aliases.len() >= 2, "self-join legs must differ:\n{sql}");
}

// -- U8 ---------------------------------------------------------------------

#[tokio::test]
async fn u8_a_window_column_is_never_referenced_by_its_display_name() {
    // DataFusion's internal display name is
    // `row_number() PARTITION BY [...] ORDER BY [...] RANGE BETWEEN ...`.
    // The old unparser quoted that as an identifier.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT * FROM (SELECT o_orderkey, \
           row_number() OVER (PARTITION BY o_custkey ORDER BY o_orderkey) AS rn \
         FROM orders) t WHERE rn = 1",
    )
    .await;
    let sql = duck(&p);
    assert!(
        !sql.contains("PARTITION BY ["),
        "DataFusion's display name leaked into SQL:\n{sql}"
    );
    assert!(
        !sql.contains("RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\""),
        "{sql}"
    );
}

// -- U9 ---------------------------------------------------------------------

#[tokio::test]
async fn u9_a_mark_join_does_not_emit_a_mark_column() {
    // `> ANY` decorrelates into a LeftMark join whose synthetic boolean the
    // old unparser emitted three times, qualified by subquery names.
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT count(*) FROM orders WHERE o_totalprice > ANY \
         (SELECT c_acctbal FROM customer WHERE c_nationkey = 2)",
    )
    .await;
    match Unparser::new(&DuckDBDialect::new()).plan_to_sql(&p) {
        Ok(sql) => {
            // The old output named the mark column three times over, each
            // qualified by `__correlated_sq_N` -- an EXISTS subquery, not a
            // relation in scope.  Neither can happen now: the mark is an
            // inlinable expression, and it only becomes a column at a seal,
            // where it is defined and uniquely named before it is referenced.
            assert!(
                !sql.contains("__correlated_sq"),
                "a subquery name was used as a relation:\n{sql}"
            );
            let defined = sql.matches(r#" AS "mark"#).count();
            let referenced = sql.matches(r#"."mark"#).count();
            assert!(
                referenced <= defined,
                "{referenced} mark references but only {defined} definitions:\n{sql}"
            );
        }
        // Refusing is acceptable; emitting something wrong is not.
        Err(e) => println!("mark join refused cleanly: {e}"),
    }
}

// -- U10 --------------------------------------------------------------------

#[tokio::test]
async fn u10_duckdb_gets_trim_not_btrim() {
    // DataFusion's trim UDF is *named* `btrim`; DuckDB has no such function.
    let ctx = ctx().await;
    let p = opt_plan(&ctx, "SELECT trim(c_name) AS t FROM customer").await;
    let sql = duck(&p);
    assert!(!sql.contains("btrim"), "DuckDB has no btrim:\n{sql}");
    assert!(sql.contains("trim("), "{sql}");

    // Postgres does have btrim, so it is left alone there.
    let sql = pg(&p);
    assert!(sql.contains("trim"), "{sql}");
}

// -- U11 --------------------------------------------------------------------

#[tokio::test]
async fn u11_integer_division_spelling_is_selectable() {
    use sqlser::dialect::{CustomDialectBuilder, DivisionStyle};

    let ctx = ctx().await;
    let p = opt_plan(&ctx, "SELECT l_orderkey / 2 AS d FROM lineitem").await;

    // DuckDB's `//` truncates toward zero, matching DataFusion. Faithful.
    let sql = duck(&p);
    assert!(sql.contains("//"), "{sql}");

    // Callers that re-plan their own output pick a spelling DataFusion parses.
    let rt = CustomDialectBuilder::new()
        .with_df_dialect(std::sync::Arc::new(
            datafusion::sql::unparser::dialect::DuckDBDialect::new(),
        ))
        .with_division_style(DivisionStyle::TruncCast)
        .build();
    let sql = Unparser::new(&rt).plan_to_sql(&p).unwrap();
    assert!(sql.contains("CAST(trunc("), "{sql}");
    assert!(!sql.contains("//"), "{sql}");
}

// -- U12 --------------------------------------------------------------------

#[tokio::test]
async fn u12_values_is_supported() {
    // DataFusion: "This feature is not implemented: Unsupported operator:
    // Values".
    let ctx = ctx().await;
    let p = plan(&ctx, "SELECT * FROM (VALUES (1,'a'),(2,'b')) AS t(x,y)").await;
    let sql = duck(&p);
    assert!(sql.contains("VALUES"), "{sql}");
    assert!(sql.contains("'a'") && sql.contains("'b'"), "{sql}");
}

// -- U13 --------------------------------------------------------------------

#[tokio::test]
async fn u13_recursive_queries_are_supported() {
    // DataFusion: "Unsupported operator: RecursiveQuery".
    let ctx = ctx().await;
    let p = plan(
        &ctx,
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n < 5) \
         SELECT sum(n) FROM t",
    )
    .await;
    let sql = duck(&p);
    assert!(sql.contains("WITH RECURSIVE"), "{sql}");
    assert!(sql.contains("UNION ALL"), "{sql}");
}

// -- U14 --------------------------------------------------------------------

#[tokio::test]
async fn u14_quoted_extract_field_is_unquoted() {
    // DataFusion plans `extract('year' from x)` as `date_part(Utf8("'year'"),
    // x)`, quotes included; rendered as-is that is `date_part('''year''', x)`,
    // which DuckDB rejects with `extract specifier "'year'" not recognized`.
    let ctx = ctx().await;
    let p = plan(
        &ctx,
        "SELECT extract('year' from o_orderdate) AS y, extract(month from o_orderdate) AS m \
         FROM orders",
    )
    .await;
    for sql in [duck(&p), pg(&p)] {
        assert!(!sql.contains("'''"), "quoted field leaked:\n{sql}");
        assert!(sql.contains("'year'"), "{sql}");
        assert!(sql.to_lowercase().contains("'month'"), "{sql}");
    }
}

// -- the standing contract --------------------------------------------------

#[tokio::test]
async fn unsupported_nodes_are_errors_not_approximations() {
    let ctx = ctx().await;
    let p = plan(&ctx, "EXPLAIN SELECT c_custkey FROM customer").await;
    let err = Unparser::new(&DuckDBDialect::new())
        .plan_to_sql(&p)
        .expect_err("EXPLAIN has no faithful rendering");
    assert!(
        matches!(err, sqlser::SqlserError::Unsupported { .. }),
        "{err}"
    );
}

// -- drop-in API ------------------------------------------------------------

#[tokio::test]
async fn expr_to_sql_matches_the_unparser_entry_point_dee_uses() {
    // `dee/src/opt/pushdown.rs:470` and `:579` render extracted predicates
    // with no enclosing plan. `BROKEN.md` found no failures at those two call
    // sites, so the behaviour here is deliberately the same: a column keeps
    // whatever qualifier it carries.
    use datafusion::logical_expr::{col, lit};

    let e = col("customer.c_acctbal").gt(lit(100i64));
    let sql = Unparser::new(&DuckDBDialect::new())
        .expr_to_sql(&e)
        .unwrap()
        .to_string();
    assert_eq!(sql, r#"("customer"."c_acctbal" > 100)"#);

    let sql = Unparser::new(&PostgreSqlDialect::new())
        .expr_to_sql(&col("c_name").eq(lit("x")))
        .unwrap()
        .to_string();
    assert_eq!(sql, r#"("c_name" = 'x')"#);
}

#[tokio::test]
async fn plan_to_statement_is_available_for_callers_that_want_the_ast() {
    let ctx = ctx().await;
    let p = opt_plan(&ctx, "SELECT c_custkey FROM customer").await;
    let stmt = Unparser::new(&DuckDBDialect::new())
        .plan_to_statement(&p)
        .unwrap();
    assert!(matches!(stmt, sqlparser::ast::Statement::Query(_)));
}

#[tokio::test]
async fn every_dialect_can_lower_a_representative_plan() {
    use sqlser::dialect::{
        BigQueryDialect, DefaultDialect, MySqlDialect, SnowflakeDialect, SqliteDialect,
    };
    let ctx = ctx().await;
    let p = opt_plan(
        &ctx,
        "SELECT c_nationkey, count(*) AS n FROM customer \
         WHERE c_acctbal > 0 GROUP BY c_nationkey ORDER BY n DESC LIMIT 5",
    )
    .await;

    let dialects: Vec<(&str, Box<dyn sqlser::Dialect>)> = vec![
        ("default", Box::new(DefaultDialect::new())),
        ("postgres", Box::new(PostgreSqlDialect::new())),
        ("duckdb", Box::new(DuckDBDialect::new())),
        ("mysql", Box::new(MySqlDialect::new())),
        ("sqlite", Box::new(SqliteDialect::new())),
        ("bigquery", Box::new(BigQueryDialect::new())),
        ("snowflake", Box::new(SnowflakeDialect::new())),
    ];
    for (name, d) in dialects {
        let sql = Unparser::new(d.as_ref())
            .plan_to_sql(&p)
            .unwrap_or_else(|e| panic!("{name} failed: {e}"));
        assert!(sql.contains("GROUP BY"), "{name}: {sql}");
        assert!(sql.contains("LIMIT 5"), "{name}: {sql}");
    }
}
