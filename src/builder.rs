//! The `SELECT` builder and its seal rule.
//!
//! A builder holds SQL's clause slots in evaluation order.  Writing a clause
//! that conflicts with what is already there does not overwrite it — the
//! builder *seals* into an aliased derived table and a fresh builder starts
//! over it.  Sealing is the only way this crate produces a derived table, and
//! it always rebuilds the scope, which is what keeps every qualifier honest.
//!
//! The slot discipline is where four of `BROKEN.md`'s bugs go to die:
//!
//! - `wheres` / `havings` / `qualifies` / join `ON` are **accumulators with no
//!   setter**, so a semi-join's `EXISTS` cannot displace the filter its left
//!   input contributed (U5).
//! - `limit` is **write-once**, and a `Sort`'s `fetch` goes into
//!   [`fetch_hint`](SelectBuilder::set_fetch_hint) — a non-clause field — so it
//!   can never clobber a `LIMIT` that is already there (U4).
//! - the select list is assembled in exactly one place, which is also the only
//!   place that can notice it came out empty (U2).
//! - that same place writes the list from the scope, in schema order, and
//!   never emits `*` (U3, U6).

use sqlparser::ast;

use crate::dialect::Dialect;
use crate::error::{Result, SqlserError};
use crate::names::{NameGen, unique_name};
use crate::scope::{ColumnAddr, Scope, ScopeEntry};

/// How far through the clause pipeline a builder has been written.
///
/// A write to rank `r` is legal only when `stage <= r`; on success the stage
/// advances to `r`.  The ordering is SQL's own evaluation order, so the rule
/// reads directly as "you cannot go back and add an earlier clause".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Stage {
    #[default]
    From = 0,
    Where = 1,
    GroupBy = 2,
    Having = 3,
    Window = 4,
    Qualify = 5,
    Select = 6,
    Distinct = 7,
    OrderBy = 8,
    Limit = 9,
}

/// A relation that can sit in a `FROM`.  Always aliased.
#[derive(Debug, Clone)]
pub struct Relation {
    pub factor: ast::TableFactor,
    pub alias: ast::Ident,
}

/// A `SELECT` under construction.
#[derive(Debug, Default)]
pub struct SelectBuilder {
    from: Option<ast::TableWithJoins>,
    wheres: Vec<ast::Expr>,
    group_by: Option<Vec<ast::Expr>>,
    group_modifier: Option<ast::GroupByWithModifier>,
    havings: Vec<ast::Expr>,
    qualifies: Vec<ast::Expr>,
    select: Option<Vec<ast::SelectItem>>,
    /// Extra select items appended at seal — `ORDER BY` keys and columns an
    /// inner correlated subquery needs — which are never exported to the scope
    /// and so can never appear in the caller's result set.
    hidden: Vec<ast::SelectItem>,
    distinct: Option<ast::Distinct>,
    order_by: Option<Vec<ast::OrderByExpr>>,
    limit: Option<ast::Expr>,
    offset: Option<ast::Expr>,

    stage: Stage,
    /// An `Aggregate` has been folded in, even one with no grouping columns.
    /// Tells a `Filter` above whether it is a `WHERE` or a `HAVING`.
    has_agg: bool,
    /// A `Sort.fetch` or `TableScan.fetch`: a row cap the optimizer attached
    /// as a hint, never written straight into `limit`.
    fetch_hint: Option<u64>,
}

impl SelectBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder whose `FROM` is `rel`.
    pub fn over(rel: Relation) -> Self {
        Self {
            from: Some(ast::TableWithJoins {
                relation: rel.factor,
                joins: vec![],
            }),
            ..Self::default()
        }
    }

    /// The alias of the relation this builder was created over, when it has
    /// exactly one and no joins yet.
    pub fn builder_alias(&self) -> ast::Ident {
        match &self.from {
            Some(ast::TableWithJoins {
                relation: ast::TableFactor::Table { alias: Some(a), .. },
                ..
            }) => a.name.clone(),
            Some(ast::TableWithJoins {
                relation: ast::TableFactor::Derived { alias: Some(a), .. },
                ..
            }) => a.name.clone(),
            _ => ast::Ident::new(""),
        }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn has_agg(&self) -> bool {
        self.has_agg
    }

    pub fn has_from(&self) -> bool {
        self.from.is_some()
    }

    pub fn select_is_set(&self) -> bool {
        self.select.is_some()
    }

    pub fn distinct_is_set(&self) -> bool {
        self.distinct.is_some()
    }

    pub fn order_by_is_set(&self) -> bool {
        self.order_by.is_some()
    }

    pub fn limit_is_set(&self) -> bool {
        self.limit.is_some() || self.offset.is_some()
    }

    pub fn group_by_is_set(&self) -> bool {
        self.group_by.is_some()
    }

    fn advance(&mut self, to: Stage) {
        if to > self.stage {
            self.stage = to;
        }
    }

    // -- writes -------------------------------------------------------------

    /// Set the driving relation.  Only legal on a fresh builder.
    pub fn set_from(&mut self, rel: Relation) -> Result<()> {
        if self.from.is_some() {
            return Err(SqlserError::invariant("FROM written twice"));
        }
        self.from = Some(ast::TableWithJoins {
            relation: rel.factor,
            joins: vec![],
        });
        Ok(())
    }

    /// A `SELECT` with no `FROM` at all (`EmptyRelation` with one row).
    pub fn set_no_from(&mut self) {
        self.from = None;
    }

    pub fn push_join(&mut self, join: ast::Join) -> Result<()> {
        let Some(from) = self.from.as_mut() else {
            return Err(SqlserError::invariant("join with no left relation"));
        };
        from.joins.push(join);
        Ok(())
    }

    /// AND a predicate into `WHERE`.  There is deliberately no `set_where`.
    pub fn and_where(&mut self, e: ast::Expr) {
        self.wheres.push(e);
        self.advance(Stage::Where);
    }

    /// AND a predicate into `HAVING`.
    pub fn and_having(&mut self, e: ast::Expr) {
        self.havings.push(e);
        self.advance(Stage::Having);
    }

    /// AND a predicate into `QUALIFY`.
    pub fn and_qualify(&mut self, e: ast::Expr) {
        self.qualifies.push(e);
        self.advance(Stage::Qualify);
    }

    pub fn set_group_by(
        &mut self,
        exprs: Vec<ast::Expr>,
        modifier: Option<ast::GroupByWithModifier>,
    ) -> Result<()> {
        if self.group_by.is_some() {
            return Err(SqlserError::invariant("GROUP BY written twice"));
        }
        self.group_by = Some(exprs);
        self.group_modifier = modifier;
        self.has_agg = true;
        self.advance(Stage::GroupBy);
        Ok(())
    }

    /// Record that an aggregate with no grouping columns was folded in.
    pub fn mark_aggregated(&mut self) {
        self.has_agg = true;
        self.advance(Stage::GroupBy);
    }

    pub fn set_select(&mut self, items: Vec<ast::SelectItem>) -> Result<()> {
        if self.select.is_some() {
            return Err(SqlserError::invariant("select list written twice"));
        }
        self.select = Some(items);
        self.advance(Stage::Select);
        Ok(())
    }

    pub fn push_hidden(&mut self, item: ast::SelectItem) {
        self.hidden.push(item);
    }

    pub fn set_distinct(&mut self, d: ast::Distinct) -> Result<()> {
        if self.distinct.is_some() {
            return Err(SqlserError::invariant("DISTINCT written twice"));
        }
        self.distinct = Some(d);
        self.advance(Stage::Distinct);
        Ok(())
    }

    pub fn set_order_by(&mut self, keys: Vec<ast::OrderByExpr>) -> Result<()> {
        if self.order_by.is_some() {
            return Err(SqlserError::invariant("ORDER BY written twice"));
        }
        self.order_by = Some(keys);
        self.advance(Stage::OrderBy);
        Ok(())
    }

    /// Write the real `LIMIT`/`OFFSET`.
    ///
    /// Discards any `fetch_hint`: the optimizer sets `Sort.fetch = skip +
    /// fetch` so the sort keeps enough rows, but that count is an internal
    /// detail of the sort, not the number of rows the query returns.  Letting
    /// it survive to the output is U4 — `LIMIT 10 OFFSET 5` coming back as
    /// `LIMIT 15 OFFSET 5`, fifteen rows instead of ten, with no error.
    pub fn set_limit(&mut self, limit: Option<ast::Expr>, offset: Option<ast::Expr>) -> Result<()> {
        if self.limit.is_some() || self.offset.is_some() {
            return Err(SqlserError::invariant("LIMIT written twice"));
        }
        self.limit = limit;
        self.offset = offset;
        self.fetch_hint = None;
        self.advance(Stage::Limit);
        Ok(())
    }

    /// Record a `fetch` that came attached to a `Sort` or `TableScan`.
    ///
    /// Deliberately not a clause write: it lands in the `LIMIT` slot only at
    /// seal, and only if nothing else has claimed it.
    pub fn set_fetch_hint(&mut self, n: u64) {
        self.fetch_hint = Some(match self.fetch_hint {
            Some(prev) => prev.max(n),
            None => n,
        });
    }

    pub fn fetch_hint(&self) -> Option<u64> {
        self.fetch_hint
    }

    // -- seal ---------------------------------------------------------------

    /// Turn this builder into an aliased derived table, and return the scope
    /// that addresses its columns.
    ///
    /// The returned scope keeps the *keys* of `scope` — the plan-schema
    /// `(qualifier, name)` pairs parent nodes still use — while every address
    /// becomes a plain column of the new alias.  That substitution is the
    /// whole of U1: after a seal there is nothing left in scope that could be
    /// named by a stale qualifier.
    pub fn seal(
        mut self,
        scope: &Scope,
        names: &mut NameGen,
        dialect: &dyn Dialect,
    ) -> Result<(Relation, Scope)> {
        // The select list names each exported column; record them so the new
        // scope can address them, and so nothing is exported twice.
        let mut exported: Vec<ast::Ident> = Vec::with_capacity(scope.len());
        match &self.select {
            None => {
                let (items, idents) = identity_projection(scope, names)?;
                self.select = Some(items);
                exported = idents;
            }
            // A select list was written from this scope, so item i is field i.
            Some(items) => {
                for item in items.iter().take(scope.len()) {
                    exported.push(select_item_ident(item)?);
                }
            }
        }

        let query = self.finish(dialect)?;
        let alias = names.fresh_rel();
        let factor = ast::TableFactor::Derived {
            lateral: false,
            subquery: Box::new(query),
            alias: Some(ast::TableAlias {
                explicit: true,
                name: alias.clone(),
                columns: vec![],
                at: None,
            }),
            sample: None,
        };

        let entries = scope
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let col = exported.get(i).cloned().ok_or_else(|| {
                    SqlserError::invariant(format!(
                        "sealed relation exports {} columns but its scope has {}",
                        exported.len(),
                        scope.len()
                    ))
                })?;
                Ok(ScopeEntry {
                    qualifier: e.qualifier.clone(),
                    name: e.name.clone(),
                    addr: ColumnAddr::Column {
                        rel: alias.clone(),
                        col,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((Relation { factor, alias }, Scope::from_entries(entries)))
    }

    /// Render the builder as a complete `Query`.
    ///
    /// This is the single choke point for the select list, so it is also the
    /// only place that can observe an empty one — which is why U2 cannot
    /// recur here the way it does in DataFusion, where the fallback exists but
    /// is wired to one call site in the `TableScan` handler and never reached
    /// by the `Projection` node the optimizer actually produces.
    pub fn finish(mut self, dialect: &dyn Dialect) -> Result<ast::Query> {
        let mut projection = self.select.take().unwrap_or_default();
        projection.append(&mut self.hidden);

        if projection.is_empty() && !dialect.supports_empty_select_list() {
            projection.push(ast::SelectItem::ExprWithAlias {
                expr: ast::Expr::Value(ast::Value::Number("1".to_string(), false).into()),
                alias: ast::Ident::with_quote('"', "__sqlser_empty"),
            });
        }

        // A fetch that no explicit LIMIT claimed becomes the LIMIT now.
        if self.limit.is_none()
            && let Some(n) = self.fetch_hint
        {
            self.limit = Some(number(n));
        }

        let select = ast::Select {
            select_token: ast::helpers::attached_token::AttachedToken::empty(),
            optimizer_hints: vec![],
            distinct: self.distinct,
            select_modifiers: None,
            top: None,
            top_before_distinct: false,
            projection,
            exclude: None,
            into: None,
            from: self.from.into_iter().collect(),
            lateral_views: vec![],
            prewhere: None,
            selection: conjoin(self.wheres),
            connect_by: vec![],
            group_by: match self.group_by {
                Some(exprs) => {
                    ast::GroupByExpr::Expressions(exprs, self.group_modifier.into_iter().collect())
                }
                None => ast::GroupByExpr::Expressions(vec![], vec![]),
            },
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            having: conjoin(self.havings),
            named_window: vec![],
            qualify: conjoin(self.qualifies),
            window_before_qualify: false,
            value_table_mode: None,
            flavor: ast::SelectFlavor::Standard,
        };

        Ok(ast::Query {
            with: None,
            body: Box::new(ast::SetExpr::Select(Box::new(select))),
            order_by: self.order_by.map(|keys| ast::OrderBy {
                kind: ast::OrderByKind::Expressions(keys),
                interpolate: None,
            }),
            limit_clause: limit_clause(self.limit, self.offset),
            fetch: None,
            locks: vec![],
            for_clause: None,
            settings: None,
            format_clause: None,
            pipe_operators: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// The select list that re-exports a scope unchanged, in schema order.
///
/// Every item is explicit and aliased.  `*` is never emitted anywhere in this
/// crate, which is U6: DataFusion's cross-join path writes the columns *and*
/// a trailing `*`, so a five-column result comes back with nine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Naming {
    /// A seal boundary.  Uses the alias each entry was given when it was
    /// created, which for a window column is a short generated name rather
    /// than DataFusion's display name (`row_number() PARTITION BY [...] RANGE
    /// BETWEEN ...`).  Keeping that display name out of identifiers is U8.
    Internal,
    /// The statement's own output.  Uses the plan schema's field names, so the
    /// result set a caller sees matches `plan.schema()` name for name.
    Output,
}

pub(crate) fn identity_projection(
    scope: &Scope,
    names: &mut NameGen,
) -> Result<(Vec<ast::SelectItem>, Vec<ast::Ident>)> {
    identity_projection_named(scope, names, Naming::Internal)
}

pub(crate) fn identity_projection_named(
    scope: &Scope,
    names: &mut NameGen,
    naming: Naming,
) -> Result<(Vec<ast::SelectItem>, Vec<ast::Ident>)> {
    let mut used: Vec<String> = Vec::new();
    let mut items = Vec::with_capacity(scope.len());
    let mut idents = Vec::with_capacity(scope.len());

    for entry in scope.iter() {
        let preferred = match (naming, &entry.addr) {
            (Naming::Internal, ColumnAddr::Projected { alias, .. }) => alias.value.clone(),
            // A mark join's `mark`, or a constant: give it a generated name
            // rather than the synthetic one the plan carries (U9).
            (Naming::Internal, ColumnAddr::Inline { .. }) => names.fresh_col("v").value,
            (Naming::Internal, ColumnAddr::Column { .. }) => safe_name(&entry.name, names),
            _ => entry.name.clone(),
        };
        let alias_text = unique_name(&preferred, &mut used);
        let alias = ast::Ident::with_quote('"', &alias_text);
        let expr = match &entry.addr {
            ColumnAddr::Column { rel, col } => {
                ast::Expr::CompoundIdentifier(vec![rel.clone(), col.clone()])
            }
            // A mark-join EXISTS or a constant becomes a real column here, and
            // gets a generated name rather than the synthetic `mark` the plan
            // carries (U9).
            ColumnAddr::Inline { ast } => (**ast).clone(),
            ColumnAddr::Projected { ast, .. } => (**ast).clone(),
        };
        items.push(ast::SelectItem::ExprWithAlias {
            expr,
            alias: alias.clone(),
        });
        idents.push(alias);
    }
    Ok((items, idents))
}

/// A name safe to emit as an identifier at an internal boundary.
///
/// DataFusion names some schema fields after the expression that produced
/// them — a window column's name is the whole `row_number() PARTITION BY
/// [orders.o_custkey] ORDER BY [...] RANGE BETWEEN ...` display string. Those
/// are fine as scope *keys* but make poor identifiers: they carry brackets and
/// run past several engines' identifier length limits. Internal boundaries
/// substitute a generated name; the scope key is untouched, so resolution is
/// unaffected, and the statement's own output names are re-imposed at the root.
pub(crate) fn safe_name(name: &str, names: &mut NameGen) -> String {
    let synthetic = name.contains('[') || name.contains('(') || name.len() > 63;
    if synthetic {
        names.fresh_col("c").value
    } else {
        name.to_string()
    }
}

fn select_item_ident(item: &ast::SelectItem) -> Result<ast::Ident> {
    match item {
        ast::SelectItem::ExprWithAlias { alias, .. } => Ok(alias.clone()),
        ast::SelectItem::UnnamedExpr(ast::Expr::Identifier(i)) => Ok(i.clone()),
        ast::SelectItem::UnnamedExpr(ast::Expr::CompoundIdentifier(parts)) => parts
            .last()
            .cloned()
            .ok_or_else(|| SqlserError::invariant("empty compound identifier")),
        other => Err(SqlserError::invariant(format!(
            "select item without a stable name: {other}"
        ))),
    }
}

/// AND a list of predicates into one, or `None` when empty.
pub(crate) fn conjoin(mut preds: Vec<ast::Expr>) -> Option<ast::Expr> {
    if preds.is_empty() {
        return None;
    }
    let mut acc = preds.remove(0);
    for p in preds {
        acc = ast::Expr::BinaryOp {
            left: Box::new(acc),
            op: ast::BinaryOperator::And,
            right: Box::new(p),
        };
    }
    Some(acc)
}

pub(crate) fn number(n: u64) -> ast::Expr {
    ast::Expr::Value(ast::Value::Number(n.to_string(), false).into())
}

fn limit_clause(limit: Option<ast::Expr>, offset: Option<ast::Expr>) -> Option<ast::LimitClause> {
    if limit.is_none() && offset.is_none() {
        return None;
    }
    Some(ast::LimitClause::LimitOffset {
        limit,
        offset: offset.map(|value| ast::Offset {
            value,
            rows: ast::OffsetRows::None,
        }),
        limit_by: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::DuckDBDialect;
    use crate::scope::{Scope, ScopeEntry};
    use datafusion::common::TableReference;

    fn scope_of(names: &[&str], rel: &str) -> Scope {
        Scope::from_entries(
            names
                .iter()
                .map(|n| ScopeEntry {
                    qualifier: Some(TableReference::bare("t")),
                    name: (*n).to_string(),
                    addr: ColumnAddr::Column {
                        rel: ast::Ident::with_quote('"', rel),
                        col: ast::Ident::with_quote('"', *n),
                    },
                })
                .collect(),
        )
    }

    fn table(name: &str, alias: &str) -> Relation {
        Relation {
            factor: ast::TableFactor::Table {
                name: ast::ObjectName(vec![ast::ObjectNamePart::Identifier(
                    ast::Ident::with_quote('"', name),
                )]),
                alias: Some(ast::TableAlias {
                    explicit: true,
                    name: ast::Ident::with_quote('"', alias),
                    columns: vec![],
                    at: None,
                }),
                args: None,
                with_hints: vec![],
                version: None,
                with_ordinality: false,
                partitions: vec![],
                json_path: None,
                sample: None,
                index_hints: vec![],
            },
            alias: ast::Ident::with_quote('"', alias),
        }
    }

    #[test]
    fn where_accumulates_and_has_no_setter() {
        // U5: a semi-join's EXISTS is pushed, not assigned, so the filter the
        // left input contributed survives.
        let mut b = SelectBuilder::over(table("customer", "r1"));
        b.and_where(ast::Expr::Identifier(ast::Ident::new("a")));
        b.and_where(ast::Expr::Identifier(ast::Ident::new("b")));
        let q = b.finish(&DuckDBDialect::new()).unwrap().to_string();
        assert!(q.contains("WHERE a AND b"), "{q}");
    }

    #[test]
    fn a_sort_fetch_never_overwrites_a_limit() {
        // U4: plan is Limit{skip:5, fetch:10} over Sort{fetch:15}.  The sort's
        // 15 is how many rows the sort must keep; the query returns 10.
        let mut b = SelectBuilder::over(table("customer", "r1"));
        b.set_fetch_hint(15);
        b.set_limit(Some(number(10)), Some(number(5))).unwrap();
        let q = b.finish(&DuckDBDialect::new()).unwrap().to_string();
        assert!(q.contains("LIMIT 10 OFFSET 5"), "{q}");
        assert!(
            !q.contains("15"),
            "the sort's internal count must not reach the output: {q}"
        );
    }

    #[test]
    fn a_standalone_fetch_hint_becomes_the_limit() {
        let mut b = SelectBuilder::over(table("customer", "r1"));
        b.set_fetch_hint(7);
        let q = b.finish(&DuckDBDialect::new()).unwrap().to_string();
        assert!(q.contains("LIMIT 7"), "{q}");
    }

    #[test]
    fn an_empty_select_list_gets_a_dummy_where_the_dialect_needs_one() {
        // U2: `SELECT FROM "customer"` is a parse error in DuckDB.
        let b = SelectBuilder::over(table("customer", "r1"));
        let q = b.finish(&DuckDBDialect::new()).unwrap().to_string();
        assert!(q.starts_with("SELECT 1 AS "), "{q}");

        let b = SelectBuilder::over(table("customer", "r1"));
        let q = b
            .finish(&crate::dialect::PostgreSqlDialect::new())
            .unwrap()
            .to_string();
        assert!(q.starts_with("SELECT FROM"), "postgres allows it: {q}");
    }

    #[test]
    fn sealing_rebinds_every_address_to_the_new_alias() {
        // U1: after a seal nothing is addressable by the old relation name.
        let scope = scope_of(&["c_custkey", "c_acctbal"], "r1");
        let b = SelectBuilder::over(table("customer", "r1"));
        let mut names = NameGen::new();
        let (rel, new_scope) = b.seal(&scope, &mut names, &DuckDBDialect::new()).unwrap();

        assert_eq!(new_scope.len(), 2);
        for entry in new_scope.iter() {
            match &entry.addr {
                ColumnAddr::Column { rel: r, .. } => assert_eq!(r.value, rel.alias.value),
                other => panic!("seal must produce plain columns, got {other:?}"),
            }
        }
        // The plan-schema keys survive, so parent nodes still resolve.
        assert!(
            new_scope
                .index_of(&datafusion::common::Column::new(Some("t"), "c_custkey"))
                .is_some()
        );
        // And the derived table is aliased, unconditionally.
        assert!(rel.alias.value.starts_with("__sqlser"));
    }

    #[test]
    fn identity_projection_is_explicit_and_deduplicated() {
        // U6: never `*`; duplicate names are disambiguated so the output can
        // be re-planned.
        let scope = scope_of(&["n_name", "n_name"], "r1");
        let mut names = NameGen::new();
        let (items, idents) = identity_projection(&scope, &mut names).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(idents[0].value, "n_name");
        assert_eq!(idents[1].value, "n_name__1");
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, ast::SelectItem::Wildcard(_)))
        );
    }

    #[test]
    fn stage_ordering_matches_sql_evaluation_order() {
        assert!(Stage::From < Stage::Where);
        assert!(Stage::Where < Stage::GroupBy);
        assert!(Stage::GroupBy < Stage::Having);
        assert!(Stage::Having < Stage::Select);
        assert!(Stage::Select < Stage::OrderBy);
        assert!(Stage::OrderBy < Stage::Limit);
    }

    #[test]
    fn writing_a_write_once_slot_twice_is_an_error_not_a_clobber() {
        let mut b = SelectBuilder::over(table("customer", "r1"));
        b.set_limit(Some(number(1)), None).unwrap();
        assert!(b.set_limit(Some(number(2)), None).is_err());
    }
}
