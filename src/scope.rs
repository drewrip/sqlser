//! Scope: the mapping from plan columns to SQL names.
//!
//! This is the module `BROKEN.md` is really about.  DataFusion's `LogicalPlan`
//! qualifies a column by the schema of the node that produced it, but SQL
//! scoping is lexical: the moment a subtree is wrapped in `FROM (SELECT …)`,
//! the enclosing query can no longer say `customer.c_custkey`, because
//! `customer` is not a relation out there any more.  The existing unparser
//! wraps without rewriting, which is U1 (48 cases), U7 and U8.
//!
//! So a `Scope` travels with every lowered relation, and says — for each field
//! of the plan's schema, *by index* — how to address that value right here.
//! Sealing a builder rebuilds the scope against the new alias, and there is no
//! code path that renders a column without consulting one.

use std::collections::HashMap;

use datafusion::common::{Column, TableReference};
use sqlparser::ast;

/// Where a plan column lives, in SQL terms, at this point in the query.
#[derive(Debug, Clone)]
pub enum ColumnAddr {
    /// A real column of a relation in the current `FROM`.  Legal in every
    /// clause, which is what makes it the address every seal produces.
    Column { rel: ast::Ident, col: ast::Ident },

    /// An expression that the current builder's `SELECT` computes.  `alias`
    /// is how `ORDER BY` refers to it; `ast` is what has to be inlined
    /// anywhere the alias is not yet visible (`HAVING`, `QUALIFY`).  Illegal
    /// in `WHERE` and in a join `ON` at the same level, because those are
    /// evaluated before the select list.
    Projected {
        alias: ast::Ident,
        ast: Box<ast::Expr>,
        kind: ProjKind,
    },

    /// A self-contained expression with no dependency on the current `FROM`:
    /// a constant, or a mark-join `EXISTS`.  Inlinable anywhere, and
    /// materialized into the select list on seal.  Keeping the mark join in
    /// this form is what stops its synthetic `mark` column reaching the
    /// output (U9).
    Inline { ast: Box<ast::Expr> },
}

/// What kind of expression a [`ColumnAddr::Projected`] holds.  Determines
/// which clauses may inline it: a window function may not appear in `HAVING`,
/// an aggregate may not appear in `WHERE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjKind {
    Plain,
    Aggregate,
    Window,
}

/// A SQL clause, used to decide whether an address is legal where it is about
/// to be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Clause {
    /// `FROM` / join `ON`.
    From,
    Where,
    GroupBy,
    Having,
    /// `WINDOW` definitions and `QUALIFY`.
    Window,
    Select,
    OrderBy,
}

/// The answer to "can I write this column into this clause".
// See `expr::Rendered`: boxing the common variant would cost an allocation per
// resolved column to shrink the rare one.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Resolution {
    /// Use this AST.
    Ast(ast::Expr),
    /// Legal only after the current builder is sealed.  Carries the reason,
    /// which surfaces in errors and in `--explain`-style diagnostics.
    NeedsSeal(&'static str),
    /// Not in this scope at all.  The caller tries enclosing scopes next.
    NotFound,
}

/// One field of a plan schema, and how to say it in SQL.
#[derive(Debug, Clone)]
pub struct ScopeEntry {
    pub qualifier: Option<TableReference>,
    pub name: String,
    pub addr: ColumnAddr,
}

/// The output columns of a lowered plan node, index-aligned with its schema.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    entries: Vec<ScopeEntry>,
    by_key: HashMap<(Option<TableReference>, String), Vec<usize>>,
    by_name: HashMap<String, Vec<usize>>,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_entries(entries: Vec<ScopeEntry>) -> Self {
        let mut s = Self {
            entries,
            by_key: HashMap::new(),
            by_name: HashMap::new(),
        };
        s.reindex();
        s
    }

    fn reindex(&mut self) {
        self.by_key.clear();
        self.by_name.clear();
        for (i, e) in self.entries.iter().enumerate() {
            self.by_key
                .entry((e.qualifier.clone(), e.name.clone()))
                .or_default()
                .push(i);
            self.by_name.entry(e.name.clone()).or_default().push(i);
        }
    }

    pub fn push(&mut self, entry: ScopeEntry) {
        let i = self.entries.len();
        self.by_key
            .entry((entry.qualifier.clone(), entry.name.clone()))
            .or_default()
            .push(i);
        self.by_name.entry(entry.name.clone()).or_default().push(i);
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ScopeEntry] {
        &self.entries
    }

    pub fn entry(&self, i: usize) -> Option<&ScopeEntry> {
        self.entries.get(i)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ScopeEntry> {
        self.entries.iter()
    }

    /// Concatenate two scopes, as a join's output schema concatenates its
    /// inputs'.
    pub fn concat(left: &Scope, right: &Scope) -> Scope {
        let mut entries = left.entries.clone();
        entries.extend(right.entries.iter().cloned());
        Scope::from_entries(entries)
    }

    /// Rebind every entry's *qualifier* to `alias`, leaving addresses alone.
    ///
    /// This is all `SubqueryAlias` does, and it is the whole of U7.  The alias
    /// is a naming fact about the plan's schema, not a relation to emit: the
    /// relation already has a generated alias of its own.  Stacked aliases
    /// rewrite twice and the outermost — the one the query actually wrote, and
    /// the one every column reference above is qualified by — survives.
    pub fn requalify(&self, alias: &TableReference) -> Scope {
        Scope::from_entries(
            self.entries
                .iter()
                .map(|e| ScopeEntry {
                    qualifier: Some(alias.clone()),
                    name: e.name.clone(),
                    addr: e.addr.clone(),
                })
                .collect(),
        )
    }

    /// Find the entry index a plan `Column` refers to.
    ///
    /// A **qualified** column must match a qualified key exactly, or match an
    /// entry that carries no qualifier at all. It deliberately does *not* fall
    /// back to a bare-name search: inside a correlated subquery that would let
    /// an outer reference such as `l1.l_orderkey` bind to the *inner*
    /// relation's own `l_orderkey`, turning TPC-H Q21's anti-join condition
    /// into `x = x` — which executes happily and returns nothing. Missing here
    /// is the right answer; the caller then searches the enclosing scopes,
    /// which is where the column actually lives.
    ///
    /// An **unqualified** column may match by name, since that is the only
    /// information it carries.
    ///
    /// Duplicate keys are possible — join outputs and renamed union branches
    /// both produce them — so this returns the first match. Callers that
    /// cannot tolerate that (joins, set operations) address by index instead.
    pub fn index_of(&self, col: &Column) -> Option<usize> {
        if let Some(i) = self
            .by_key
            .get(&(col.relation.clone(), col.name.clone()))
            .and_then(|v| v.first())
        {
            return Some(*i);
        }
        match col.relation {
            // Unqualified: name is all there is.
            None => self.by_name.get(&col.name).and_then(|v| v.first()).copied(),
            // Qualified: an entry that lost its qualifier (a projection
            // output, say) is still a legitimate match; another relation's
            // same-named column is not.
            Some(_) => self
                .by_key
                .get(&(None, col.name.clone()))
                .and_then(|v| v.first())
                .copied(),
        }
    }

    /// Resolve a plan `Column` for use in `clause`.
    pub fn resolve(&self, col: &Column, clause: Clause) -> Resolution {
        match self.index_of(col) {
            None => Resolution::NotFound,
            Some(i) => self.resolve_index(i, clause),
        }
    }

    /// Resolve by position.  Used wherever names cannot be trusted to line up
    /// across the boundary — union branches especially, whose schemas keep
    /// their own qualifiers while the union output drops them.
    pub fn resolve_index(&self, i: usize, clause: Clause) -> Resolution {
        let Some(entry) = self.entries.get(i) else {
            return Resolution::NotFound;
        };
        match &entry.addr {
            ColumnAddr::Column { rel, col } => {
                Resolution::Ast(ast::Expr::CompoundIdentifier(vec![
                    rel.clone(),
                    col.clone(),
                ]))
            }
            ColumnAddr::Inline { ast } => Resolution::Ast((**ast).clone()),
            ColumnAddr::Projected { alias, ast, kind } => match (clause, kind) {
                // Evaluated before the select list exists.
                (Clause::From, _) => Resolution::NeedsSeal("join ON cannot see the select list"),
                (Clause::Where, _) => Resolution::NeedsSeal("WHERE cannot see the select list"),
                (Clause::GroupBy, _) => {
                    Resolution::NeedsSeal("GROUP BY cannot see the select list")
                }
                // A window function is evaluated after HAVING, so it cannot be
                // inlined there; anything else can.
                (Clause::Having, ProjKind::Window) => {
                    Resolution::NeedsSeal("HAVING cannot see a window function")
                }
                (Clause::Having, _) | (Clause::Window, _) | (Clause::Select, _) => {
                    Resolution::Ast((**ast).clone())
                }
                // ORDER BY is the one clause that can use the output alias.
                (Clause::OrderBy, _) => Resolution::Ast(ast::Expr::Identifier(alias.clone())),
            },
        }
    }
}

/// The stack of enclosing scopes, innermost last.
///
/// A correlated subquery is lowered with its parent's scope pushed here, so
/// `Expr::OuterReferenceColumn` — and the bare `Expr::Column`s DataFusion also
/// emits for correlations — resolve against the outer relation's *generated*
/// alias.  Because every relation alias is generated and unique, an outer
/// reference can never be captured by a relation introduced inside the
/// subquery; that is the structural reason correlation is safe here.
#[derive(Debug, Default)]
pub struct OuterScopes {
    stack: Vec<Scope>,
}

impl OuterScopes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, s: Scope) {
        self.stack.push(s);
    }

    pub fn pop(&mut self) {
        self.stack.pop();
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Search outward for `col`.  An outer reference is always a plain column
    /// of an enclosing `FROM`, so only `Column` addresses count: an enclosing
    /// select-list alias is not visible inside a subquery in any dialect.
    pub fn resolve(&self, col: &Column) -> Option<ast::Expr> {
        for scope in self.stack.iter().rev() {
            let Some(i) = scope.index_of(col) else {
                continue;
            };
            let Some(entry) = scope.entry(i) else {
                continue;
            };
            if let ColumnAddr::Column { rel, col } = &entry.addr {
                return Some(ast::Expr::CompoundIdentifier(vec![
                    rel.clone(),
                    col.clone(),
                ]));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col_entry(q: &str, n: &str, rel: &str, c: &str) -> ScopeEntry {
        ScopeEntry {
            qualifier: Some(TableReference::bare(q)),
            name: n.to_string(),
            addr: ColumnAddr::Column {
                rel: ast::Ident::with_quote('"', rel),
                col: ast::Ident::with_quote('"', c),
            },
        }
    }

    fn projected(n: &str, alias: &str, kind: ProjKind) -> ScopeEntry {
        ScopeEntry {
            qualifier: None,
            name: n.to_string(),
            addr: ColumnAddr::Projected {
                alias: ast::Ident::with_quote('"', alias),
                ast: Box::new(ast::Expr::Identifier(ast::Ident::new("inner"))),
                kind,
            },
        }
    }

    #[test]
    fn a_column_resolves_to_its_relation_alias_not_its_plan_qualifier() {
        // This is the U1 shape: the plan says `customer.c_custkey`, but the
        // relation in scope is a generated alias.
        let s = Scope::from_entries(vec![col_entry(
            "customer",
            "c_custkey",
            "__sqlser_r1",
            "c_custkey",
        )]);
        let got = s.resolve(&Column::new(Some("customer"), "c_custkey"), Clause::Select);
        match got {
            Resolution::Ast(ast::Expr::CompoundIdentifier(parts)) => {
                assert_eq!(parts[0].value, "__sqlser_r1");
                assert_eq!(parts[1].value, "c_custkey");
            }
            other => panic!("expected a compound identifier, got {other:?}"),
        }
    }

    #[test]
    fn requalify_keeps_addresses_and_swaps_only_the_plan_qualifier() {
        // U7: `SubqueryAlias: s` over `SubqueryAlias: stg_cust` over a scan.
        // Both rewrites keep pointing at the scan's generated alias, and the
        // outermost name — the one the query used — is what resolves.
        let scan = Scope::from_entries(vec![col_entry(
            "customer",
            "c_custkey",
            "__sqlser_r1",
            "c_custkey",
        )]);
        let inner = scan.requalify(&TableReference::bare("stg_cust"));
        let outer = inner.requalify(&TableReference::bare("s"));

        let got = outer.resolve(&Column::new(Some("s"), "c_custkey"), Clause::Select);
        match got {
            Resolution::Ast(ast::Expr::CompoundIdentifier(parts)) => {
                assert_eq!(
                    parts[0].value, "__sqlser_r1",
                    "must address the real relation"
                );
            }
            other => panic!("expected a compound identifier, got {other:?}"),
        }
        // The stale inner name is gone from the keys.
        assert!(
            outer
                .by_key
                .contains_key(&(Some(TableReference::bare("s")), "c_custkey".into()))
        );
        assert!(
            !outer
                .by_key
                .contains_key(&(Some(TableReference::bare("stg_cust")), "c_custkey".into()))
        );
    }

    #[test]
    fn a_projected_column_is_illegal_before_the_select_list_exists() {
        let s = Scope::from_entries(vec![projected("rn", "__sqlser_win1", ProjKind::Window)]);
        let c = Column::new_unqualified("rn");

        // U8: filtering on a window column cannot happen in WHERE.
        assert!(matches!(
            s.resolve(&c, Clause::Where),
            Resolution::NeedsSeal(_)
        ));
        assert!(matches!(
            s.resolve(&c, Clause::From),
            Resolution::NeedsSeal(_)
        ));
        assert!(matches!(
            s.resolve(&c, Clause::Having),
            Resolution::NeedsSeal(_)
        ));
        // But QUALIFY and SELECT may inline it, and ORDER BY may use the alias.
        assert!(matches!(s.resolve(&c, Clause::Window), Resolution::Ast(_)));
        assert!(matches!(s.resolve(&c, Clause::Select), Resolution::Ast(_)));
        match s.resolve(&c, Clause::OrderBy) {
            Resolution::Ast(ast::Expr::Identifier(i)) => assert_eq!(i.value, "__sqlser_win1"),
            other => panic!("ORDER BY should use the alias, got {other:?}"),
        }
    }

    #[test]
    fn an_aggregate_may_be_inlined_in_having_but_not_in_where() {
        let s = Scope::from_entries(vec![projected("n", "n", ProjKind::Aggregate)]);
        let c = Column::new_unqualified("n");
        assert!(matches!(s.resolve(&c, Clause::Having), Resolution::Ast(_)));
        assert!(matches!(
            s.resolve(&c, Clause::Where),
            Resolution::NeedsSeal(_)
        ));
    }

    #[test]
    fn concat_lines_up_with_a_join_schema() {
        let l = Scope::from_entries(vec![col_entry("customer", "c_custkey", "r1", "c_custkey")]);
        let r = Scope::from_entries(vec![col_entry("orders", "o_orderkey", "r2", "o_orderkey")]);
        let j = Scope::concat(&l, &r);
        assert_eq!(j.len(), 2);
        assert_eq!(j.entry(0).unwrap().name, "c_custkey");
        assert_eq!(j.entry(1).unwrap().name, "o_orderkey");
    }

    #[test]
    fn outer_scopes_resolve_correlations_to_the_generated_alias() {
        let mut outers = OuterScopes::new();
        outers.push(Scope::from_entries(vec![col_entry(
            "customer",
            "c_custkey",
            "__sqlser_r1",
            "c_custkey",
        )]));
        let got = outers.resolve(&Column::new(Some("customer"), "c_custkey"));
        match got {
            Some(ast::Expr::CompoundIdentifier(parts)) => assert_eq!(parts[0].value, "__sqlser_r1"),
            other => panic!("expected an outer resolution, got {other:?}"),
        }
        assert!(outers.resolve(&Column::new_unqualified("nope")).is_none());
    }
}
