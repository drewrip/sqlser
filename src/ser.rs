//! The plan walk.
//!
//! Lowering is bottom-up.  Each node returns a [`Lowered`] — a builder under
//! construction plus the [`Scope`] describing its output — and the single
//! invariant every arm maintains is:
//!
//! > the scope has exactly one entry per field of `plan.schema()`, in that
//! > order, and each entry says how to address that value here.
//!
//! Column *order* therefore comes from the plan's own schema rather than from
//! the order an arm happens to iterate its expressions, which is why the
//! aggregate arm cannot reproduce U3, and the select list is always explicit,
//! which is why no arm can reproduce U6.

use std::sync::Arc;

use datafusion::common::{Column, DFSchema, DFSchemaRef, TableReference};
use datafusion::logical_expr::ExprSchemable;
use datafusion::logical_expr::{Expr, LogicalPlan, Subquery};
use sqlparser::ast;

use crate::builder::{Relation, SelectBuilder, identity_projection};
use crate::dialect::Dialect;
use crate::error::{Result, SqlserError};
use crate::expr::{ExprCtx, Rendered, Resolved};
use crate::names::NameGen;
use crate::scope::{Clause, ColumnAddr, OuterScopes, Resolution, Scope, ScopeEntry};

/// A plan node lowered to a builder plus the scope that addresses its output.
#[derive(Debug)]
pub struct Lowered {
    pub builder: SelectBuilder,
    pub scope: Scope,
    /// The schema of the plan node this was lowered from.
    ///
    /// Expressions belonging to the node *above* are typed against it, which
    /// is what lets the integer-division rewrite tell `7 / 2` (truncating)
    /// from `7.5 / 2` (not) without guessing from the SQL text.
    pub schema: DFSchemaRef,
}

impl Default for Lowered {
    fn default() -> Self {
        Self {
            builder: SelectBuilder::default(),
            scope: Scope::new(),
            schema: Arc::new(DFSchema::empty()),
        }
    }
}

/// Knobs that change the shape of the output without changing its meaning.
#[derive(Debug, Clone)]
pub struct Config {
    /// Guard against pathological nesting rather than blowing the stack.
    pub max_derived_depth: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_derived_depth: 64,
        }
    }
}

/// Serializes one plan.  Holds the name generator, so every alias it produces
/// is unique across the whole statement.
pub struct Serializer<'a> {
    pub(crate) dialect: &'a dyn Dialect,
    pub(crate) names: NameGen,
    pub(crate) outers: OuterScopes,
    pub(crate) config: Config,
    pub(crate) depth: usize,
    /// `WITH RECURSIVE` bodies collected while lowering, attached to the root.
    pub(crate) ctes: Vec<ast::Cte>,
    pub(crate) ctes_recursive: bool,
    /// Names currently resolvable as CTE relations rather than base tables.
    pub(crate) cte_names: Vec<String>,
}

impl<'a> Serializer<'a> {
    pub fn new(dialect: &'a dyn Dialect) -> Self {
        Self {
            dialect,
            names: NameGen::new(),
            outers: OuterScopes::new(),
            config: Config::default(),
            depth: 0,
            ctes: Vec::new(),
            ctes_recursive: false,
            cte_names: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Lower a whole plan to a `Query`.
    pub fn plan_to_query(&mut self, plan: &LogicalPlan) -> Result<ast::Query> {
        let lowered = self.lower(plan)?;
        self.finish_root(lowered, plan.schema())
    }

    /// Finish the outermost builder, forcing its select list to match the
    /// plan's schema exactly.
    ///
    /// This is the output contract: field *i* of the result is field *i* of
    /// `plan.schema()`, under that name.  A serializer that got the order
    /// wrong cannot reach here, which is the standing guard against U3 and
    /// U6 recurring through some future arm.
    fn finish_root(&mut self, mut lowered: Lowered, schema: &DFSchema) -> Result<ast::Query> {
        if !lowered.builder.select_is_set() && !lowered.scope.is_empty() {
            let (items, _) = crate::builder::identity_projection_named(
                &lowered.scope,
                &mut self.names,
                crate::builder::Naming::Output,
            )?;
            lowered.builder.set_select(items)?;
        }
        let mut query = lowered.builder.finish(self.dialect)?;

        // The output contract: field `i` of the result is field `i` of the
        // plan's schema, under that name. Internal boundaries are free to
        // rename for legibility; the statement's own result set is not.
        rename_output(&mut query, schema)?;

        if !self.ctes.is_empty() {
            query.with = Some(ast::With {
                with_token: ast::helpers::attached_token::AttachedToken::empty(),
                recursive: self.ctes_recursive,
                cte_tables: std::mem::take(&mut self.ctes),
            });
        }

        debug_assert_eq!(
            projection_len(&query),
            schema.fields().len(),
            "output arity must equal the plan schema's"
        );
        Ok(query)
    }

    // -- scaffolding --------------------------------------------------------

    pub(crate) fn seal_in_place_pub(&mut self, l: &mut Lowered) -> Result<()> {
        self.depth += 1;
        if self.depth > self.config.max_derived_depth {
            return Err(SqlserError::DepthExceeded {
                limit: self.config.max_derived_depth,
            });
        }
        let builder = std::mem::take(&mut l.builder);
        let (rel, scope) = builder.seal(&l.scope, &mut self.names, self.dialect)?;
        l.builder = SelectBuilder::over(rel);
        l.scope = scope;
        Ok(())
    }

    pub(crate) fn seal(&mut self, mut l: Lowered) -> Result<(Relation, Scope)> {
        self.depth += 1;
        if self.depth > self.config.max_derived_depth {
            return Err(SqlserError::DepthExceeded {
                limit: self.config.max_derived_depth,
            });
        }
        let builder = std::mem::take(&mut l.builder);
        builder.seal(&l.scope, &mut self.names, self.dialect)
    }

    /// Render a batch of expressions, sealing once if any of them is not yet
    /// addressable.
    ///
    /// Batched deliberately: rendering nothing until every expression in the
    /// clause is known to resolve is what keeps a rendered AST from ever
    /// outliving the scope it was built against.
    pub(crate) fn render_all(
        &mut self,
        l: &mut Lowered,
        exprs: &[Expr],
        clause: Clause,
        node: &str,
    ) -> Result<Vec<ast::Expr>> {
        for attempt in 0..2 {
            let mut out = Vec::with_capacity(exprs.len());
            let mut blocked = None;
            for e in exprs {
                match self.render_one(&l.scope, &l.schema, e, clause)? {
                    Rendered::Ast(a) => out.push(a),
                    Rendered::NeedsSeal(why) => {
                        blocked = Some(why);
                        break;
                    }
                }
            }
            match blocked {
                None => return Ok(out),
                Some(why) => {
                    if attempt == 1 {
                        return Err(SqlserError::invariant(format!(
                            "{node}: expression still unaddressable after sealing ({why})"
                        )));
                    }
                    self.seal_in_place_pub(l)?;
                }
            }
        }
        unreachable!("the loop returns on both attempts")
    }

    fn render_one(
        &mut self,
        scope: &Scope,
        schema: &DFSchemaRef,
        e: &Expr,
        clause: Clause,
    ) -> Result<Rendered> {
        let dialect = self.dialect;
        let mut ctx = Ctx {
            scope: scope.clone(),
            schema: Arc::clone(schema),
            clause,
            ser: self,
        };
        crate::expr::render(e, &mut ctx, dialect)
    }

    // -- dispatch -----------------------------------------------------------

    pub fn lower(&mut self, plan: &LogicalPlan) -> Result<Lowered> {
        let mut lowered = self.lower_node(plan)?;
        // Recorded centrally so no arm can forget it.
        lowered.schema = plan.schema().clone();
        Ok(lowered)
    }

    fn lower_node(&mut self, plan: &LogicalPlan) -> Result<Lowered> {
        match plan {
            LogicalPlan::TableScan(scan) => self.lower_scan(scan),
            LogicalPlan::Projection(p) => self.lower_projection(p),
            LogicalPlan::Filter(f) => self.lower_filter(f),
            LogicalPlan::Aggregate(a) => self.lower_aggregate(a),
            LogicalPlan::Sort(s) => self.lower_sort(s),
            LogicalPlan::Limit(l) => self.lower_limit(l),
            LogicalPlan::Distinct(d) => self.lower_distinct(d),
            LogicalPlan::Join(j) => self.lower_join(j),
            LogicalPlan::Union(u) => self.lower_union(u),
            LogicalPlan::SubqueryAlias(a) => self.lower_subquery_alias(a),
            LogicalPlan::Window(w) => self.lower_window(w),
            LogicalPlan::Values(v) => self.lower_values(v),
            LogicalPlan::EmptyRelation(e) => self.lower_empty(e),
            LogicalPlan::RecursiveQuery(r) => self.lower_recursive(r),
            LogicalPlan::Unnest(u) => self.lower_unnest(u),
            // Repartition is a physical hint; it cannot change the result set,
            // so it is transparent.
            LogicalPlan::Repartition(r) => self.lower(&r.input),
            LogicalPlan::Subquery(Subquery { subquery, .. }) => {
                let inner = self.lower(subquery)?;
                let (rel, scope) = self.seal(inner)?;
                Ok(Lowered {
                    builder: SelectBuilder::over(rel),
                    scope,
                    ..Default::default()
                })
            }
            other => Err(SqlserError::unsupported(
                plan_name(other),
                "no faithful SQL rendering; refusing rather than guessing",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// The expression context
// ---------------------------------------------------------------------------

struct Ctx<'s, 'a> {
    ser: &'s mut Serializer<'a>,
    scope: Scope,
    schema: DFSchemaRef,
    clause: Clause,
}

impl ExprCtx for Ctx<'_, '_> {
    fn resolve_column(&self, col: &Column) -> Resolved {
        match self.scope.resolve(col, self.clause) {
            Resolution::Ast(a) => Resolved::Ast(a),
            Resolution::NeedsSeal(w) => Resolved::NeedsSeal(w),
            Resolution::NotFound => Resolved::NotFound,
        }
    }

    fn resolve_outer(&mut self, col: &Column) -> Option<ast::Expr> {
        self.ser.outers.resolve(col)
    }

    fn lower_subquery(&mut self, plan: &LogicalPlan) -> Result<ast::Query> {
        // The subquery is lowered with this scope pushed as its parent, so an
        // outer reference resolves to the enclosing relation's generated
        // alias.  Because aliases are generated, a relation introduced inside
        // the subquery can never capture it.
        self.ser.outers.push(self.scope.clone());
        let result = (|| {
            let mut lowered = self.ser.lower(plan)?;
            debug_assert_eq!(
                lowered.scope.len(),
                plan.schema().fields().len(),
                "a lowered subquery must expose exactly its schema"
            );
            if !lowered.builder.select_is_set() {
                let (items, _) = identity_projection(&lowered.scope, &mut self.ser.names)?;
                lowered.builder.set_select(items)?;
            }
            lowered.builder.finish(self.ser.dialect)
        })();
        self.ser.outers.pop();
        result
    }

    fn names(&mut self) -> &mut NameGen {
        &mut self.ser.names
    }

    fn expr_type(&self, e: &Expr) -> Option<datafusion::arrow::datatypes::DataType> {
        e.get_type(self.schema.as_ref()).ok()
    }

    fn render_sub(&mut self, e: &Expr) -> Result<Rendered> {
        let dialect = self.ser.dialect;
        let mut inner = Ctx {
            scope: self.scope.clone(),
            schema: Arc::clone(&self.schema),
            clause: self.clause,
            ser: self.ser,
        };
        crate::expr::render(e, &mut inner, dialect)
    }
}

// ---------------------------------------------------------------------------
// helpers shared by the arms
// ---------------------------------------------------------------------------

pub(crate) fn plan_name(plan: &LogicalPlan) -> &'static str {
    match plan {
        LogicalPlan::Explain(_) => "Explain",
        LogicalPlan::Analyze(_) => "Analyze",
        LogicalPlan::Ddl(_) => "Ddl",
        LogicalPlan::Dml(_) => "Dml",
        LogicalPlan::Copy(_) => "Copy",
        LogicalPlan::DescribeTable(_) => "DescribeTable",
        LogicalPlan::Statement(_) => "Statement",
        LogicalPlan::Extension(_) => "Extension",
        LogicalPlan::Unnest(_) => "Unnest",
        _ => "plan node",
    }
}

/// Force the top-level select list's aliases to the plan schema's field names.
fn rename_output(q: &mut ast::Query, schema: &DFSchema) -> Result<()> {
    let ast::SetExpr::Select(select) = q.body.as_mut() else {
        // A bare set operation already carries the union schema's names.
        return Ok(());
    };
    if select.projection.len() != schema.fields().len() {
        return Err(SqlserError::invariant(format!(
            "output has {} columns, the plan schema has {}",
            select.projection.len(),
            schema.fields().len()
        )));
    }
    let mut used = Vec::new();
    for (item, f) in select.projection.iter_mut().zip(schema.fields()) {
        let alias = ast::Ident::with_quote('"', crate::names::unique_name(f.name(), &mut used));
        let expr = match item {
            ast::SelectItem::ExprWithAlias { expr, .. } => expr.clone(),
            ast::SelectItem::UnnamedExpr(e) => e.clone(),
            other => {
                return Err(SqlserError::invariant(format!(
                    "unexpected select item at the root: {other}"
                )));
            }
        };
        *item = ast::SelectItem::ExprWithAlias { expr, alias };
    }
    Ok(())
}

fn projection_len(q: &ast::Query) -> usize {
    match q.body.as_ref() {
        ast::SetExpr::Select(s) => s.projection.len(),
        _ => usize::MAX,
    }
}

/// Build the scope for a relation whose columns are exactly `schema`'s fields,
/// all addressable as plain columns of `alias`.
pub(crate) fn scope_over_relation(schema: &DFSchema, alias: &ast::Ident) -> Scope {
    Scope::from_entries(
        schema
            .iter()
            .map(|(q, f)| ScopeEntry {
                qualifier: q.cloned(),
                name: f.name().clone(),
                addr: ColumnAddr::Column {
                    rel: alias.clone(),
                    col: ast::Ident::with_quote('"', f.name()),
                },
            })
            .collect(),
    )
}

pub(crate) fn object_name(name: &TableReference) -> ast::ObjectName {
    let parts: Vec<ast::ObjectNamePart> = match name {
        TableReference::Bare { table } => vec![ident_part(table)],
        TableReference::Partial { schema, table } => vec![ident_part(schema), ident_part(table)],
        TableReference::Full {
            catalog,
            schema,
            table,
        } => vec![ident_part(catalog), ident_part(schema), ident_part(table)],
    };
    ast::ObjectName(parts)
}

fn ident_part(s: &str) -> ast::ObjectNamePart {
    ast::ObjectNamePart::Identifier(ast::Ident::with_quote('"', s))
}

pub(crate) fn table_factor(name: ast::ObjectName, alias: ast::Ident) -> ast::TableFactor {
    ast::TableFactor::Table {
        name,
        alias: Some(ast::TableAlias {
            explicit: true,
            name: alias,
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
    }
}

impl Serializer<'_> {
    /// Render an expression with no enclosing plan.
    ///
    /// Columns resolve to themselves, qualifier and all. That is the right
    /// behaviour for the one caller shape this serves — a predicate lifted out
    /// of a plan and pasted into SQL that still mentions the same tables — and
    /// it is why `dee`'s two `expr_to_sql` call sites were never implicated in
    /// `BROKEN.md`'s census.
    pub(crate) fn render_bare(&mut self, expr: &Expr) -> Result<ast::Expr> {
        let dialect = self.dialect;
        let mut ctx = BareCtx { ser: self };
        match crate::expr::render(expr, &mut ctx, dialect)? {
            Rendered::Ast(a) => Ok(a),
            Rendered::NeedsSeal(why) => Err(SqlserError::invariant(format!(
                "bare expression needs a relation scope ({why})"
            ))),
        }
    }
}

struct BareCtx<'s, 'a> {
    ser: &'s mut Serializer<'a>,
}

impl ExprCtx for BareCtx<'_, '_> {
    fn expr_type(&self, _e: &Expr) -> Option<datafusion::arrow::datatypes::DataType> {
        None
    }

    fn render_sub(&mut self, e: &Expr) -> Result<Rendered> {
        let dialect = self.ser.dialect;
        let mut inner = BareCtx { ser: self.ser };
        crate::expr::render(e, &mut inner, dialect)
    }

    fn resolve_column(&self, col: &Column) -> Resolved {
        let ident = ast::Ident::with_quote('"', &col.name);
        Resolved::Ast(match &col.relation {
            Some(r) => {
                ast::Expr::CompoundIdentifier(vec![ast::Ident::with_quote('"', r.table()), ident])
            }
            None => ast::Expr::Identifier(ident),
        })
    }

    fn resolve_outer(&mut self, col: &Column) -> Option<ast::Expr> {
        match self.resolve_column(col) {
            Resolved::Ast(a) => Some(a),
            _ => None,
        }
    }

    fn lower_subquery(&mut self, plan: &LogicalPlan) -> Result<ast::Query> {
        self.ser.plan_to_query(plan)
    }

    fn names(&mut self) -> &mut NameGen {
        &mut self.ser.names
    }
}
