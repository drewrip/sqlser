//! Expression rendering.
//!
//! `BROKEN.md` measured DataFusion's `expr_to_sql` as the one sound part of
//! the unparser — zero failures across the 210-case corpus — so this crate
//! delegates to it rather than re-deriving several thousand lines of literal,
//! cast, interval and scalar-function handling.  What it does *not* delegate
//! is anything that needs to know where in the query it is:
//!
//! - **Columns.** Every `Expr::Column` and `Expr::OuterReferenceColumn` is
//!   resolved through the current [`Scope`] and replaced by a hole *before*
//!   the delegate sees it.  The delegate therefore never emits a qualifier of
//!   its own, which is the mechanism by which U1, U7 and U8 stop being
//!   possible rather than merely being fixed.
//! - **Subqueries.** `Unparser::expr_to_sql` renders `ScalarSubquery`,
//!   `InSubquery` and `Exists` by calling its own `plan_to_sql` — the buggy
//!   plan walk.  Handing it one would reintroduce every bug this crate
//!   exists to remove, so those arms are lowered here and spliced in.
//!
//! The splice is structural.  Each intercepted subtree becomes a uniquely
//! named placeholder column, the delegate renders the rest, and the resulting
//! `ast::Expr` tree is walked to swap each placeholder identifier for the AST
//! we built.  Nothing is ever substituted by string surgery.

use std::collections::HashMap;
use std::ops::ControlFlow;

use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::Expr;
use datafusion::sql::unparser::Unparser;
use sqlparser::ast;
use sqlparser::ast::visit_expressions_mut;

use crate::dialect::{Dialect, DivisionStyle};
use crate::error::{Result, SqlserError};
use crate::names::{NameGen, RESERVED};

/// The outcome of trying to render an expression at a particular point.
// The `Ast` variant is much larger than the others; boxing it would put an
// allocation on the hot path of every rendered expression to save a few bytes
// on the rare one.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Rendered {
    Ast(ast::Expr),
    /// One of the columns is not addressable in this clause yet.  The caller
    /// seals and tries again; on a freshly sealed builder every address is a
    /// plain column, so the retry always succeeds.
    NeedsSeal(&'static str),
}

/// A set of placeholder substitutions to apply after delegation.
#[derive(Debug, Default)]
pub struct Holes {
    map: HashMap<String, ast::Expr>,
}

impl Holes {
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn insert(&mut self, key: String, value: ast::Expr) {
        self.map.insert(key, value);
    }

    /// Replace every placeholder identifier in `expr` with its AST.
    ///
    /// Runs to a fixed point over the tree; placeholders are leaves, so one
    /// pass of the visitor suffices, but the loop guards against a delegate
    /// that wraps a placeholder in another placeholder-bearing rewrite.
    pub fn apply(&self, mut expr: ast::Expr) -> Result<ast::Expr> {
        if self.map.is_empty() {
            return Ok(expr);
        }
        for _ in 0..8 {
            let mut hit = false;
            let _: ControlFlow<()> = visit_expressions_mut(&mut expr, |e| {
                if let Some(name) = placeholder_name(e)
                    && let Some(replacement) = self.map.get(&name)
                {
                    *e = replacement.clone();
                    hit = true;
                }
                ControlFlow::Continue(())
            });
            if !hit {
                return Ok(expr);
            }
        }
        Err(SqlserError::invariant(
            "expression placeholders did not settle",
        ))
    }
}

/// The placeholder name an identifier carries, if it is one.
fn placeholder_name(e: &ast::Expr) -> Option<String> {
    let ident = match e {
        ast::Expr::Identifier(i) => i,
        // A dialect that qualifies bare columns may render the placeholder as
        // a compound identifier; the last part is still the name.
        ast::Expr::CompoundIdentifier(parts) => parts.last()?,
        _ => return None,
    };
    ident
        .value
        .starts_with(&format!("{RESERVED}_hole"))
        .then(|| ident.value.clone())
}

/// A hole standing in for an already-rendered subtree.
fn hole_expr(name: &str) -> Expr {
    Expr::Column(Column::new_unqualified(name))
}

/// Everything the expression layer needs from its caller, so that this module
/// stays free of a dependency on the plan walk (which in turn depends on it).
pub trait ExprCtx {
    /// Resolve a plan column into SQL at the current clause.
    fn resolve_column(&self, col: &Column) -> Resolved;
    /// Resolve an explicit outer reference.
    fn resolve_outer(&mut self, col: &Column) -> Option<ast::Expr>;
    /// Lower a subquery plan into a complete `ast::Query`.
    fn lower_subquery(
        &mut self,
        plan: &datafusion::logical_expr::LogicalPlan,
    ) -> Result<ast::Query>;
    fn names(&mut self) -> &mut NameGen;
    /// The type of an expression in the current plan node's input schema.
    /// `None` when it cannot be determined, which callers must treat as
    /// "assume nothing".
    fn expr_type(&self, e: &Expr) -> Option<datafusion::arrow::datatypes::DataType>;
    /// Render a sub-expression in the same context.
    fn render_sub(&mut self, e: &Expr) -> Result<Rendered>;
}

/// What [`ExprCtx::resolve_column`] can say.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Resolved {
    Ast(ast::Expr),
    NeedsSeal(&'static str),
    NotFound,
}

/// Render `expr` to SQL.
///
/// Two phases, and the split matters: phase one resolves every column and can
/// still back out with [`Rendered::NeedsSeal`]; phase two renders, and by then
/// nothing can fail on scope grounds.  A rendered AST is therefore never held
/// across a seal, so it can never carry a qualifier from a scope that has
/// since been replaced.
pub fn render<C: ExprCtx>(expr: &Expr, ctx: &mut C, dialect: &dyn Dialect) -> Result<Rendered> {
    // Aliases are stripped up front, not inside the walk below.  Unwrapping
    // one mid-`transform_down` would hand the walk a node it has already
    // passed, so the column underneath would never be resolved and would reach
    // the delegate with its plan qualifier intact — the exact leak this layer
    // exists to prevent.
    let expr = expr.clone().unalias_nested().data;

    // Phase one: intercept everything context-dependent.
    let mut holes = Holes::default();
    let mut blocked: Option<&'static str> = None;

    let holed = expr
        .transform_down(|e| {
            if blocked.is_some() {
                return Ok(Transformed::new(e, false, TreeNodeRecursion::Stop));
            }
            match &e {
                Expr::Column(col) => match ctx.resolve_column(col) {
                    Resolved::Ast(ast) => Ok(punch(&mut holes, ctx, ast)),
                    Resolved::NeedsSeal(why) => {
                        blocked = Some(why);
                        Ok(Transformed::new(e, false, TreeNodeRecursion::Stop))
                    }
                    // Not in this scope: a correlation written as a plain
                    // column, which DataFusion does emit.
                    Resolved::NotFound => match ctx.resolve_outer(col) {
                        Some(ast) => Ok(punch(&mut holes, ctx, ast)),
                        None => Err(datafusion::error::DataFusionError::Plan(format!(
                            "sqlser: unresolved column {col}"
                        ))),
                    },
                },
                Expr::OuterReferenceColumn(_, col) => match ctx.resolve_outer(col) {
                    Some(ast) => Ok(punch(&mut holes, ctx, ast)),
                    None => Err(datafusion::error::DataFusionError::Plan(format!(
                        "sqlser: unresolved outer reference {col}"
                    ))),
                },
                // Integer division, rewritten here rather than over the
                // rendered AST because only here are the operand types still
                // known.  DuckDB's `//` is type-dependent — it truncates on
                // integers (`7 // 2 = 3`) and divides normally on decimals
                // (`7.5 // 2 = 3.75`) — so a text-level rewrite cannot tell
                // the two apart, and rewriting both is a silent wrong answer.
                Expr::BinaryExpr(be)
                    if be.op == datafusion::logical_expr::Operator::Divide
                        && dialect.division_style() == DivisionStyle::TruncCast
                        && ctx.expr_type(&e).is_some_and(|t| t.is_integer()) =>
                {
                    let left = match ctx.render_sub(&be.left).map_err(to_df)? {
                        Rendered::Ast(a) => a,
                        Rendered::NeedsSeal(why) => {
                            blocked = Some(why);
                            return Ok(Transformed::new(e, false, TreeNodeRecursion::Stop));
                        }
                    };
                    let right = match ctx.render_sub(&be.right).map_err(to_df)? {
                        Rendered::Ast(a) => a,
                        Rendered::NeedsSeal(why) => {
                            blocked = Some(why);
                            return Ok(Transformed::new(e, false, TreeNodeRecursion::Stop));
                        }
                    };
                    Ok(punch(&mut holes, ctx, trunc_div(left, right)))
                }
                // DataFusion plans `extract('year' from x)` — a quoted field —
                // as `date_part(Utf8("'year'"), x)`, keeping the quotes in the
                // string. Its own `date_part` strips them when it executes, so
                // the plan means `year`; rendered as-is it becomes
                // `date_part('''year''', x)`, which DuckDB and Postgres reject.
                // Every dialect gets the fix, so it lives here, not in one.
                Expr::ScalarFunction(f) if f.func.name() == "date_part" => {
                    match unquoted_date_part_field(&f.args) {
                        Some(args) => Ok(Transformed::yes(Expr::ScalarFunction(
                            datafusion::logical_expr::expr::ScalarFunction::new_udf(
                                std::sync::Arc::clone(&f.func),
                                args,
                            ),
                        ))),
                        None => Ok(Transformed::no(e)),
                    }
                }
                Expr::ScalarSubquery(sq) => {
                    let q = ctx.lower_subquery(&sq.subquery).map_err(to_df)?;
                    Ok(punch(&mut holes, ctx, ast::Expr::Subquery(Box::new(q))))
                }
                Expr::Exists(ex) => {
                    let q = ctx.lower_subquery(&ex.subquery.subquery).map_err(to_df)?;
                    Ok(punch(
                        &mut holes,
                        ctx,
                        ast::Expr::Exists {
                            subquery: Box::new(q),
                            negated: ex.negated,
                        },
                    ))
                }
                Expr::InSubquery(insq) => {
                    // The probe expression is rendered by this same function,
                    // so it gets the same column treatment.
                    let probe = match render(&insq.expr, ctx, dialect).map_err(to_df)? {
                        Rendered::Ast(a) => a,
                        Rendered::NeedsSeal(why) => {
                            blocked = Some(why);
                            return Ok(Transformed::new(e, false, TreeNodeRecursion::Stop));
                        }
                    };
                    let q = ctx.lower_subquery(&insq.subquery.subquery).map_err(to_df)?;
                    Ok(punch(
                        &mut holes,
                        ctx,
                        ast::Expr::InSubquery {
                            expr: Box::new(probe),
                            subquery: Box::new(q),
                            negated: insq.negated,
                        },
                    ))
                }
                _ => Ok(Transformed::no(e)),
            }
        })
        .map_err(SqlserError::DataFusion)?
        .data;

    if let Some(why) = blocked {
        return Ok(Rendered::NeedsSeal(why));
    }

    // Phase two: delegate the dialect-heavy remainder.
    let rendered = Unparser::new(dialect.df())
        .expr_to_sql(&holed)
        .map_err(SqlserError::DataFusion)?;

    let spliced = holes.apply(rendered)?;
    Ok(Rendered::Ast(apply_division_style(spliced, dialect)?))
}

fn to_df(e: SqlserError) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::External(Box::new(e))
}

/// Record `ast` under a fresh placeholder and return the placeholder node.
fn punch<C: ExprCtx>(holes: &mut Holes, ctx: &mut C, ast: ast::Expr) -> Transformed<Expr> {
    let name = ctx.names().fresh_hole();
    holes.insert(name.clone(), ast);
    Transformed::new(hole_expr(&name), true, TreeNodeRecursion::Jump)
}

// ---------------------------------------------------------------------------
// Integer division (U11)
// ---------------------------------------------------------------------------

/// Rewrite integer division to the requested spelling.
///
/// DataFusion's expression renderer emits `BinaryOperator::DuckIntegerDivide`
/// (`//`) wherever the target dialect's `division_operator()` says so, which
/// for DuckDB is everywhere.  That is the *correct* result — measured, `-7 //
/// 2 = -3`, matching Postgres's `-7 / 2` and DataFusion's own integer division
/// — but DataFusion 55 cannot parse `//` back, which is U11 and which matters
/// for any caller that re-plans its own output.
///
/// So the operator is the one node rewritten here, and only into forms that
/// mean exactly the same thing.
fn apply_division_style(mut expr: ast::Expr, dialect: &dyn Dialect) -> Result<ast::Expr> {
    if dialect.division_style() == DivisionStyle::FloorOperator {
        return Ok(expr);
    }
    let _: ControlFlow<()> = visit_expressions_mut(&mut expr, |e| {
        if let ast::Expr::BinaryOp { left, op, right } = e
            && *op == ast::BinaryOperator::DuckIntegerDivide
        {
            // Whatever `//` is left is a division the type check did not
            // find integral, and for those `//` and `/` mean the same thing
            // on every engine that spells it this way.  `/` is the one both
            // the engine and DataFusion's own parser accept.
            *e = ast::Expr::BinaryOp {
                left: left.clone(),
                op: ast::BinaryOperator::Divide,
                right: right.clone(),
            };
        }
        ControlFlow::Continue(())
    });
    Ok(expr)
}

/// `date_part`'s arguments with a quoted field literal (`'year'`, as planned
/// from `extract('year' from x)`) unquoted, or `None` when there is nothing to
/// fix.
fn unquoted_date_part_field(args: &[Expr]) -> Option<Vec<Expr>> {
    use datafusion::scalar::ScalarValue;

    let (Expr::Literal(value, metadata), rest) = args.split_first()? else {
        return None;
    };
    let field = match value {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => s,
        _ => return None,
    };
    let inner = field.strip_prefix('\'')?.strip_suffix('\'')?;

    let unquoted = match value {
        ScalarValue::LargeUtf8(_) => ScalarValue::LargeUtf8(Some(inner.to_string())),
        ScalarValue::Utf8View(_) => ScalarValue::Utf8View(Some(inner.to_string())),
        _ => ScalarValue::Utf8(Some(inner.to_string())),
    };
    let mut fixed = vec![Expr::Literal(unquoted, metadata.clone())];
    fixed.extend(rest.iter().cloned());
    Some(fixed)
}

/// `CAST(trunc(a / b) AS BIGINT)` — truncation toward zero, which is what both
/// DataFusion's `/` and DuckDB's `//` do, spelled so every parser accepts it.
fn trunc_div(left: ast::Expr, right: ast::Expr) -> ast::Expr {
    let quotient = ast::Expr::Nested(Box::new(ast::Expr::BinaryOp {
        left: Box::new(left),
        op: ast::BinaryOperator::Divide,
        right: Box::new(right),
    }));
    ast::Expr::Cast {
        kind: ast::CastKind::Cast,
        expr: Box::new(call("trunc", vec![quotient])),
        data_type: ast::DataType::BigInt(None),
        format: None,
        array: false,
    }
}

pub(crate) fn call(name: &str, args: Vec<ast::Expr>) -> ast::Expr {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holes_are_substituted_structurally() {
        let mut holes = Holes::default();
        holes.insert(
            "__sqlser_hole1".into(),
            ast::Expr::CompoundIdentifier(vec![
                ast::Ident::with_quote('"', "r1"),
                ast::Ident::with_quote('"', "c_custkey"),
            ]),
        );
        let rendered = ast::Expr::BinaryOp {
            left: Box::new(ast::Expr::Identifier(ast::Ident::new("__sqlser_hole1"))),
            op: ast::BinaryOperator::Gt,
            right: Box::new(ast::Expr::Value(
                ast::Value::Number("1".into(), false).into(),
            )),
        };
        let out = holes.apply(rendered).unwrap();
        assert_eq!(out.to_string(), r#""r1"."c_custkey" > 1"#);
    }

    #[test]
    fn a_quoted_placeholder_is_still_recognised() {
        // DuckDB's dialect quotes every identifier, so the delegate renders
        // the hole as `"__sqlser_hole1"`.
        let e = ast::Expr::Identifier(ast::Ident::with_quote('"', "__sqlser_hole1"));
        assert_eq!(placeholder_name(&e).as_deref(), Some("__sqlser_hole1"));
        let e = ast::Expr::Identifier(ast::Ident::new("c_custkey"));
        assert!(placeholder_name(&e).is_none());
    }

    #[test]
    fn trunc_cast_division_is_re_parseable_and_truncates() {
        let e = trunc_div(
            ast::Expr::Identifier(ast::Ident::new("a")),
            ast::Expr::Value(ast::Value::Number("2".into(), false).into()),
        );
        assert_eq!(e.to_string(), "CAST(trunc((a / 2)) AS BIGINT)");
    }

    #[test]
    fn division_style_rewrites_only_the_integer_operator() {
        let duck_divide = ast::Expr::BinaryOp {
            left: Box::new(ast::Expr::Identifier(ast::Ident::new("a"))),
            op: ast::BinaryOperator::DuckIntegerDivide,
            right: Box::new(ast::Expr::Identifier(ast::Ident::new("b"))),
        };
        let float_divide = ast::Expr::BinaryOp {
            left: Box::new(ast::Expr::Identifier(ast::Ident::new("a"))),
            op: ast::BinaryOperator::Divide,
            right: Box::new(ast::Expr::Identifier(ast::Ident::new("b"))),
        };

        let duck = crate::dialect::DuckDBDialect::new();
        assert_eq!(
            apply_division_style(duck_divide.clone(), &duck)
                .unwrap()
                .to_string(),
            "a // b",
            "DuckDB's // is the faithful spelling and is left alone"
        );

        let pg = crate::dialect::PostgreSqlDialect::new();
        assert_eq!(
            apply_division_style(duck_divide.clone(), &pg)
                .unwrap()
                .to_string(),
            "a / b"
        );
        assert_eq!(
            apply_division_style(float_divide.clone(), &pg)
                .unwrap()
                .to_string(),
            "a / b",
            "plain division is never touched"
        );

        // Under TruncCast the integral cases are rewritten earlier, where
        // the operand types are still known; whatever `//` reaches this pass
        // was *not* integral, and for those `//` and `/` agree — so `/`, the
        // spelling DataFusion can also parse back, is the right output.
        let rt = crate::dialect::CustomDialectBuilder::new()
            .with_division_style(DivisionStyle::TruncCast)
            .build();
        assert_eq!(
            apply_division_style(duck_divide, &rt).unwrap().to_string(),
            "a / b"
        );
    }
}
