//! Dialects.
//!
//! `sqlser` keeps its own `Dialect` trait rather than reusing DataFusion's,
//! for two reasons.  The plan walk needs knobs DataFusion does not model
//! (whether `DISTINCT ON` exists, whether grouping sets are available), and a
//! few of DataFusion's defaults are wrong in ways `BROKEN.md` documents.
//!
//! Expression rendering is still delegated, so every dialect carries a
//! configured DataFusion dialect via [`Dialect::df`].  That keeps parity with
//! `Unparser` for the whole literal/cast/function surface for free, and means
//! a correction like U10's `btrim` is applied once, here, where a caller
//! cannot forget it.

use std::sync::Arc;

use datafusion::sql::unparser::dialect as df;
// Brought in for its builder methods (`with_custom_scalar_overrides`); our own
// `Dialect` is the one callers see.
use datafusion::sql::unparser::dialect::Dialect as DfDialect;
use sqlparser::ast;

/// How to render integer division.
///
/// DataFusion's `/` on two integers is integer division truncating toward
/// zero.  Engines disagree about the spelling, and one of them disagrees
/// about the meaning:
///
/// - Postgres `/` on integers already truncates toward zero — [`Native`] is
///   faithful.
/// - DuckDB `/` is *float* division (`7 / 2 = 3.5`); its `//` truncates toward
///   zero (measured: `-7 // 2 = -3`, matching Postgres and DataFusion), so
///   [`FloorOperator`] is the faithful spelling despite the name.
///
/// [`TruncCast`] exists for callers that re-plan their own output:
/// DataFusion 55 cannot parse `//` back (`Operator DIV is not yet supported`),
/// which is U11.  It costs a little legibility and is exactly equivalent.
///
/// [`Native`]: DivisionStyle::Native
/// [`FloorOperator`]: DivisionStyle::FloorOperator
/// [`TruncCast`]: DivisionStyle::TruncCast
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DivisionStyle {
    /// Plain `/`.  Correct where `/` on integers already truncates.
    #[default]
    Native,
    /// DuckDB's `//`.
    FloorOperator,
    /// `CAST(trunc(a / b) AS BIGINT)` — re-parseable everywhere.
    TruncCast,
}

/// How a dialect spells `DISTINCT ON`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistinctOnStyle {
    /// Native `SELECT DISTINCT ON (a, b) …`.
    Native,
    /// No native form; rewrite via `row_number()` in a derived table.
    RowNumber,
}

/// Which `GROUP BY` extensions the engine has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupingSupport {
    pub rollup: bool,
    pub cube: bool,
    pub grouping_sets: bool,
}

impl GroupingSupport {
    pub const ALL: Self = Self { rollup: true, cube: true, grouping_sets: true };
    pub const NONE: Self = Self { rollup: false, cube: false, grouping_sets: false };
}

/// Everything the plan walk needs to know about the target engine.
///
/// Every method has a default, so a new dialect is a small impl.  The one
/// method without a default is [`df`](Dialect::df), because expression
/// rendering cannot be guessed.
pub trait Dialect: std::fmt::Debug + Send + Sync {
    /// The DataFusion dialect used for delegated expression rendering.
    fn df(&self) -> &dyn df::Dialect;

    /// Quote character for generated identifiers.
    fn quote(&self) -> Option<char> {
        Some('"')
    }

    /// Whether `SELECT` with no select list parses.  When false, an otherwise
    /// empty list gets a dummy literal — this is the U2 guard.
    fn supports_empty_select_list(&self) -> bool {
        false
    }

    /// Whether `QUALIFY` is available for filtering on window functions.
    fn supports_qualify(&self) -> bool {
        false
    }

    /// Whether `(SELECT …) AS t(a, b)` may name columns in the table alias.
    fn supports_column_alias_in_table_alias(&self) -> bool {
        true
    }

    fn distinct_on_style(&self) -> DistinctOnStyle {
        DistinctOnStyle::RowNumber
    }

    fn grouping_support(&self) -> GroupingSupport {
        GroupingSupport::ALL
    }

    fn division_style(&self) -> DivisionStyle {
        DivisionStyle::Native
    }

    /// Whether `UNNEST` may appear directly as a table factor.
    fn unnest_as_table_factor(&self) -> bool {
        false
    }

    /// Whether a set-returning `unnest(...)` may appear in the select list.
    ///
    /// This is the shape DataFusion's `Unnest` node actually has — it sits
    /// above a projection that computed the list — so it is the rendering that
    /// needs no restructuring.
    fn unnest_in_select_list(&self) -> bool {
        false
    }

    /// Whether a `SELECT` with no `FROM` parses.  MySQL-style engines that
    /// need `FROM DUAL` say false.
    fn supports_select_without_from(&self) -> bool {
        true
    }

    /// `IS NOT DISTINCT FROM` vs an engine-specific spelling, used for
    /// null-equating join keys.
    fn null_safe_equality(&self) -> NullSafeEquality {
        NullSafeEquality::IsNotDistinctFrom
    }
}

/// Spelling of a null-equating comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullSafeEquality {
    /// `a IS NOT DISTINCT FROM b`
    IsNotDistinctFrom,
    /// MySQL's `a <=> b`
    Spaceship,
}

// ---------------------------------------------------------------------------
// Scalar-function corrections applied to the delegated renderer
// ---------------------------------------------------------------------------

/// Render `f(args…)` as a plain function call.
fn call(name: &str, args: Vec<ast::Expr>) -> ast::Expr {
    ast::Expr::Function(ast::Function {
        name: ast::ObjectName(vec![ast::ObjectNamePart::Identifier(ast::Ident::new(name))]),
        uses_odbc_syntax: false,
        parameters: ast::FunctionArguments::None,
        args: ast::FunctionArguments::List(ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: args
                .into_iter()
                .map(|a| ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(a)))
                .collect(),
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    })
}

/// U10: DataFusion's `trim` UDF is *named* `btrim`, and the expression
/// renderer emits functions by their DataFusion name.  Postgres has `btrim`;
/// DuckDB does not (`Catalog Error: Scalar Function with name btrim does not
/// exist!`).  Rewriting it here means every DuckDB dialect instance gets the
/// fix, rather than each caller remembering `with_custom_scalar_overrides`.
fn duckdb_df_dialect() -> df::DuckDBDialect {
    use datafusion::sql::unparser::Unparser;
    df::DuckDBDialect::new().with_custom_scalar_overrides(vec![(
        "btrim",
        Box::new(|unparser: &Unparser, args: &[datafusion::logical_expr::Expr]| {
            let rendered = args
                .iter()
                .map(|a| unparser.expr_to_sql(a))
                .collect::<datafusion::error::Result<Vec<_>>>()?;
            Ok(Some(call("trim", rendered)))
        }) as df::ScalarFnToSqlHandler,
    )])
}

// ---------------------------------------------------------------------------
// The dialects, matching `Unparser`'s set one for one
// ---------------------------------------------------------------------------

macro_rules! simple_dialect {
    ($name:ident, $df:expr, { $($body:tt)* }) => {
        pub struct $name {
            df: Box<dyn df::Dialect>,
        }
        // `df::Dialect` is not `Debug`, so derive is unavailable; the name is
        // the only interesting part anyway.
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(stringify!($name))
            }
        }
        impl $name {
            #[allow(clippy::new_without_default)]
            pub fn new() -> Self {
                Self { df: Box::new($df) }
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl Dialect for $name {
            fn df(&self) -> &dyn df::Dialect {
                self.df.as_ref()
            }
            $($body)*
        }
    };
}

simple_dialect!(DefaultDialect, df::DefaultDialect {}, {
    fn quote(&self) -> Option<char> { None }
    fn supports_empty_select_list(&self) -> bool { false }
});

simple_dialect!(PostgreSqlDialect, df::PostgreSqlDialect {}, {
    fn supports_empty_select_list(&self) -> bool { true }
    fn supports_qualify(&self) -> bool { false }
    fn distinct_on_style(&self) -> DistinctOnStyle { DistinctOnStyle::Native }
    fn division_style(&self) -> DivisionStyle { DivisionStyle::Native }
    fn unnest_in_select_list(&self) -> bool { true }
});

/// DuckDB.
///
/// Carries the `btrim` -> `trim` correction (U10) on the delegate, so a caller
/// cannot lose it by reaching for [`CustomDialectBuilder`] to change something
/// unrelated — which is how that bug survives in practice.
pub struct DuckDBDialect {
    df: Box<dyn df::Dialect>,
    division_style: DivisionStyle,
}

impl DuckDBDialect {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            df: Box::new(duckdb_df_dialect()),
            division_style: DivisionStyle::FloorOperator,
        }
    }

    /// Choose how integer division is spelled.
    ///
    /// The default, [`DivisionStyle::FloorOperator`], emits `//`, which is
    /// what DuckDB means and what DataFusion means. Callers that re-plan their
    /// own output want [`DivisionStyle::TruncCast`] instead, because
    /// DataFusion 55 cannot parse `//` back (U11).
    pub fn with_division_style(mut self, style: DivisionStyle) -> Self {
        self.division_style = style;
        self
    }
}

impl Default for DuckDBDialect {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DuckDBDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDBDialect")
            .field("division_style", &self.division_style)
            .finish()
    }
}

impl Dialect for DuckDBDialect {
    fn df(&self) -> &dyn df::Dialect { self.df.as_ref() }
    fn supports_qualify(&self) -> bool { true }
    fn supports_empty_select_list(&self) -> bool { false }
    fn distinct_on_style(&self) -> DistinctOnStyle { DistinctOnStyle::Native }
    fn division_style(&self) -> DivisionStyle { self.division_style }
    fn unnest_as_table_factor(&self) -> bool { true }
    fn unnest_in_select_list(&self) -> bool { true }
}

simple_dialect!(MySqlDialect, df::MySqlDialect {}, {
    fn quote(&self) -> Option<char> { Some('`') }
    fn supports_qualify(&self) -> bool { false }
    fn grouping_support(&self) -> GroupingSupport {
        GroupingSupport { rollup: true, cube: false, grouping_sets: false }
    }
    fn supports_column_alias_in_table_alias(&self) -> bool { false }
    fn supports_select_without_from(&self) -> bool { false }
    fn null_safe_equality(&self) -> NullSafeEquality { NullSafeEquality::Spaceship }
});

simple_dialect!(SqliteDialect, df::SqliteDialect {}, {
    fn supports_qualify(&self) -> bool { false }
    fn grouping_support(&self) -> GroupingSupport { GroupingSupport::NONE }
    fn supports_column_alias_in_table_alias(&self) -> bool { false }
});

simple_dialect!(BigQueryDialect, df::BigQueryDialect {}, {
    fn quote(&self) -> Option<char> { Some('`') }
    fn supports_qualify(&self) -> bool { true }
    fn unnest_as_table_factor(&self) -> bool { true }
    fn unnest_in_select_list(&self) -> bool { true }
    fn supports_column_alias_in_table_alias(&self) -> bool { false }
});

simple_dialect!(SnowflakeDialect, df::SnowflakeDialect {}, {
    fn supports_qualify(&self) -> bool { true }
    fn unnest_as_table_factor(&self) -> bool { true }
    fn unnest_in_select_list(&self) -> bool { true }
});

/// A dialect assembled at runtime, for engines with no built-in impl.
pub struct CustomDialect {
    df: Arc<dyn df::Dialect>,
    quote: Option<char>,
    supports_empty_select_list: bool,
    supports_qualify: bool,
    supports_column_alias_in_table_alias: bool,
    supports_select_without_from: bool,
    unnest_in_select_list: bool,
    distinct_on_style: DistinctOnStyle,
    grouping_support: GroupingSupport,
    division_style: DivisionStyle,
    unnest_as_table_factor: bool,
    null_safe_equality: NullSafeEquality,
}

impl std::fmt::Debug for CustomDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomDialect")
            .field("quote", &self.quote)
            .field("division_style", &self.division_style)
            .finish_non_exhaustive()
    }
}

impl Dialect for CustomDialect {
    fn df(&self) -> &dyn df::Dialect { self.df.as_ref() }
    fn quote(&self) -> Option<char> { self.quote }
    fn supports_empty_select_list(&self) -> bool { self.supports_empty_select_list }
    fn supports_qualify(&self) -> bool { self.supports_qualify }
    fn supports_column_alias_in_table_alias(&self) -> bool { self.supports_column_alias_in_table_alias }
    fn supports_select_without_from(&self) -> bool { self.supports_select_without_from }
    fn distinct_on_style(&self) -> DistinctOnStyle { self.distinct_on_style }
    fn grouping_support(&self) -> GroupingSupport { self.grouping_support }
    fn division_style(&self) -> DivisionStyle { self.division_style }
    fn unnest_as_table_factor(&self) -> bool { self.unnest_as_table_factor }
    fn unnest_in_select_list(&self) -> bool { self.unnest_in_select_list }
    fn null_safe_equality(&self) -> NullSafeEquality { self.null_safe_equality }
}

/// Builder for [`CustomDialect`].  Starts from conservative defaults: no
/// engine extensions assumed, no empty select lists, native division.
#[derive(Debug)]
pub struct CustomDialectBuilder {
    inner: CustomDialect,
}

impl Default for CustomDialectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomDialectBuilder {
    pub fn new() -> Self {
        Self {
            inner: CustomDialect {
                df: Arc::new(df::DefaultDialect {}),
                quote: Some('"'),
                supports_empty_select_list: false,
                supports_qualify: false,
                supports_column_alias_in_table_alias: true,
                supports_select_without_from: true,
                unnest_in_select_list: false,
                distinct_on_style: DistinctOnStyle::RowNumber,
                grouping_support: GroupingSupport::NONE,
                division_style: DivisionStyle::Native,
                unnest_as_table_factor: false,
                null_safe_equality: NullSafeEquality::IsNotDistinctFrom,
            },
        }
    }

    pub fn with_df_dialect(mut self, d: Arc<dyn df::Dialect>) -> Self {
        self.inner.df = d;
        self
    }
    pub fn with_quote(mut self, q: Option<char>) -> Self {
        self.inner.quote = q;
        self
    }
    pub fn with_supports_empty_select_list(mut self, v: bool) -> Self {
        self.inner.supports_empty_select_list = v;
        self
    }
    pub fn with_supports_qualify(mut self, v: bool) -> Self {
        self.inner.supports_qualify = v;
        self
    }
    pub fn with_supports_column_alias_in_table_alias(mut self, v: bool) -> Self {
        self.inner.supports_column_alias_in_table_alias = v;
        self
    }
    pub fn with_supports_select_without_from(mut self, v: bool) -> Self {
        self.inner.supports_select_without_from = v;
        self
    }
    pub fn with_distinct_on_style(mut self, v: DistinctOnStyle) -> Self {
        self.inner.distinct_on_style = v;
        self
    }
    pub fn with_grouping_support(mut self, v: GroupingSupport) -> Self {
        self.inner.grouping_support = v;
        self
    }
    pub fn with_division_style(mut self, v: DivisionStyle) -> Self {
        self.inner.division_style = v;
        self
    }
    pub fn with_unnest_as_table_factor(mut self, v: bool) -> Self {
        self.inner.unnest_as_table_factor = v;
        self
    }
    pub fn with_unnest_in_select_list(mut self, v: bool) -> Self {
        self.inner.unnest_in_select_list = v;
        self
    }
    pub fn with_null_safe_equality(mut self, v: NullSafeEquality) -> Self {
        self.inner.null_safe_equality = v;
        self
    }

    pub fn build(self) -> CustomDialect {
        self.inner
    }
}

/// Resolve a dialect by engine name, matching `dee`'s `dialect_for_db`.
///
/// Unlike that function this returns `None` for an unknown name rather than
/// silently falling back to DuckDB: guessing the engine is how a correct
/// serializer produces SQL the engine rejects.
pub fn dialect_for_db(db: &str) -> Option<Box<dyn Dialect>> {
    let d: Box<dyn Dialect> = match db.to_lowercase().as_str() {
        "duckdb" => Box::new(DuckDBDialect::new()),
        "postgresql" | "postgres" => Box::new(PostgreSqlDialect::new()),
        "mysql" => Box::new(MySqlDialect::new()),
        "sqlite" => Box::new(SqliteDialect::new()),
        "bigquery" => Box::new(BigQueryDialect::new()),
        "snowflake" => Box::new(SnowflakeDialect::new()),
        "default" | "generic" => Box::new(DefaultDialect::new()),
        _ => return None,
    };
    Some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duckdb_defaults_differ_from_postgres_where_it_matters() {
        let duck = DuckDBDialect::new();
        let pg = PostgreSqlDialect::new();

        // U2: DuckDB rejects `SELECT FROM t`, Postgres accepts `SELECT`.
        assert!(!duck.supports_empty_select_list());
        assert!(pg.supports_empty_select_list());

        // U11: `/` means float division in DuckDB, integer division in Postgres.
        assert_eq!(duck.division_style(), DivisionStyle::FloorOperator);
        assert_eq!(pg.division_style(), DivisionStyle::Native);

        assert!(duck.supports_qualify());
        assert!(!pg.supports_qualify());
    }

    #[test]
    fn unknown_engine_is_none_not_a_guess() {
        assert!(dialect_for_db("duckdb").is_some());
        assert!(dialect_for_db("PostGreSQL").is_some());
        assert!(dialect_for_db("snowflake").is_some());
        assert!(dialect_for_db("clickhouse").is_none());
        assert!(dialect_for_db("").is_none());
    }

    #[test]
    fn custom_builder_round_trips_its_knobs() {
        let d = CustomDialectBuilder::new()
            .with_supports_qualify(true)
            .with_division_style(DivisionStyle::TruncCast)
            .with_quote(Some('`'))
            .build();
        assert!(d.supports_qualify());
        assert_eq!(d.division_style(), DivisionStyle::TruncCast);
        assert_eq!(d.quote(), Some('`'));
    }
}
