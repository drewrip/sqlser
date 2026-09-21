//! `sqlser` — `LogicalPlan` to SQL.
//!
//! A replacement for `datafusion::sql::unparser` built around one idea: the
//! serializer carries an explicit **scope** saying how to address each column
//! of the plan it is lowering, and the only way it produces a derived table is
//! by *sealing* a builder, which rebuilds that scope. DataFusion's unparser
//! wraps subtrees without rewriting qualifiers, which is why `BROKEN.md`
//! measures it failing on 48% of optimized plans, four of those modes with no
//! error at all.
//!
//! ```no_run
//! use sqlser::{Unparser, dialect::DuckDBDialect};
//! # fn demo(plan: &datafusion::logical_expr::LogicalPlan) -> sqlser::Result<()> {
//! let sql = Unparser::new(&DuckDBDialect::new()).plan_to_sql(plan)?;
//! println!("{sql}");
//! # Ok(()) }
//! ```
//!
//! ## What is guaranteed
//!
//! - Every derived table is aliased, and every qualifier in the enclosing
//!   scope names a relation that is actually in scope.
//! - Output column *count, order and names* match `plan.schema()`.
//! - A plan with no faithful SQL rendering is an [`Err`], never approximate
//!   SQL. That matters more than coverage: the failure modes that cost the
//!   most in `BROKEN.md` are the ones that returned `Ok`.

pub mod builder;
pub mod dialect;
pub mod error;
pub mod expr;
mod lower;
mod names;
pub mod scope;
pub mod ser;

use datafusion::logical_expr::{Expr, LogicalPlan};
use sqlparser::ast;

pub use dialect::{Dialect, dialect_for_db};
pub use error::{Result, SqlserError};
pub use ser::Config;

/// Serializes plans and expressions for one dialect.
///
/// Shaped like `datafusion::sql::unparser::Unparser` so swapping is
/// mechanical, but `plan_to_sql` returns the SQL text directly — the statement
/// AST is available through [`Unparser::plan_to_statement`] for callers that
/// want it.
pub struct Unparser<'a> {
    dialect: &'a dyn Dialect,
    config: Config,
}

impl<'a> Unparser<'a> {
    pub fn new(dialect: &'a dyn Dialect) -> Self {
        Self {
            dialect,
            config: Config::default(),
        }
    }

    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Lower a plan to SQL text.
    pub fn plan_to_sql(&self, plan: &LogicalPlan) -> Result<String> {
        Ok(self.plan_to_query(plan)?.to_string())
    }

    /// Lower a plan to a `Query` AST.
    pub fn plan_to_query(&self, plan: &LogicalPlan) -> Result<ast::Query> {
        ser::Serializer::new(self.dialect)
            .with_config(self.config.clone())
            .plan_to_query(plan)
    }

    /// Lower a plan to a `Statement`, for callers that hand the AST onward.
    pub fn plan_to_statement(&self, plan: &LogicalPlan) -> Result<ast::Statement> {
        Ok(ast::Statement::Query(Box::new(self.plan_to_query(plan)?)))
    }

    /// Render a standalone expression.
    ///
    /// No plan, so no scope: column references keep whatever qualifier they
    /// carry. This mirrors `Unparser::expr_to_sql`, and is the one entry point
    /// where the caller is responsible for the names being meaningful.
    pub fn expr_to_sql(&self, expr: &Expr) -> Result<ast::Expr> {
        let mut ser = ser::Serializer::new(self.dialect);
        ser.render_bare(expr)
    }
}
