//! One arm per `LogicalPlan` variant.
//!
//! Each arm follows the same shape: decide whether the builder must seal
//! *before* rendering anything, seal if so, then write. The seal decision is
//! stated explicitly in each arm rather than buried in a shared "can I fold
//! this in" heuristic, because that heuristic is exactly what drifted out of
//! sync in the implementation `BROKEN.md` measures.

use std::sync::Arc;

use datafusion::common::{DFSchema, NullEquality};
use datafusion::logical_expr::{
    Aggregate, Distinct, EmptyRelation, Expr, Filter, Join, JoinType, Limit, LogicalPlan,
    Projection, RecursiveQuery, Sort, SubqueryAlias, TableScan, Union, Values, Window,
    expr::GroupingSet,
};
use sqlparser::ast;

use crate::builder::{Relation, SelectBuilder, Stage, identity_projection, number};
use crate::dialect::{DistinctOnStyle, NullSafeEquality};
use crate::error::{Result, SqlserError};
use crate::names::unique_name;
use crate::scope::{Clause, ColumnAddr, ProjKind, Scope, ScopeEntry};
use crate::ser::{Lowered, Serializer, object_name, scope_over_relation, table_factor};

impl Serializer<'_> {
    // -- leaves -------------------------------------------------------------

    /// `TableScan` — the base case.
    ///
    /// The relation always gets a generated alias, even though the table has
    /// a perfectly good name.  That is what makes a self-join's two legs
    /// distinguishable without relying on the plan's qualifiers, and what
    /// makes an outer reference impossible to capture.
    pub(crate) fn lower_scan(&mut self, scan: &TableScan) -> Result<Lowered> {
        let alias = self.names.fresh_rel();
        let factor = if self.cte_names.iter().any(|n| *n == scan.table_name.table()) {
            table_factor(
                ast::ObjectName(vec![ast::ObjectNamePart::Identifier(ast::Ident::with_quote(
                    '"',
                    scan.table_name.table(),
                ))]),
                alias.clone(),
            )
        } else {
            table_factor(object_name(&scan.table_name), alias.clone())
        };

        let scope = scope_over_relation(&scan.projected_schema, &alias);
        let mut l = Lowered {
            builder: SelectBuilder::over(Relation { factor, alias }),
            scope,
            schema: Arc::clone(&scan.projected_schema),
        };

        // Filters the provider accepted. Re-emitting one the plan also keeps
        // as a `Filter` node is harmless — these predicates are idempotent —
        // and omitting one would silently widen the result.
        //
        // They are rendered against the table's *full* schema, not against
        // `projected_schema`. A provider that claims exact pushdown — which is
        // the shape `dee`'s `OpaqueScanTable` produces — lets the optimizer
        // narrow the projection to what survives downstream while the filters
        // still mention columns that did not, and those columns are perfectly
        // addressable in the scan's own WHERE.
        if !scan.filters.is_empty() {
            let full = scan_scope(scan, &l.builder.builder_alias());
            let full_schema = DFSchema::try_from_qualified_schema(
                scan.table_name.clone(),
                scan.source.schema().as_ref(),
            )
            .map(Arc::new)
            .unwrap_or_else(|_| Arc::clone(&scan.projected_schema));
            let mut probe = Lowered {
                builder: std::mem::take(&mut l.builder),
                scope: full,
                schema: full_schema,
            };
            let preds = self.render_all(&mut probe, &scan.filters, Clause::Where, "TableScan")?;
            l.builder = probe.builder;
            for p in preds {
                l.builder.and_where(p);
            }
        }

        // A scan-level row cap must be applied before anything above it can
        // filter, so it seals immediately rather than sharing the LIMIT slot
        // with whatever comes next.
        if let Some(n) = scan.fetch {
            l.builder.set_limit(Some(number(n as u64)), None)?;
            let (rel, scope) = self.seal(l)?;
            l = Lowered {
                builder: SelectBuilder::over(rel),
                scope,
                schema: Arc::clone(&scan.projected_schema),
            };
        }
        Ok(l)
    }

    /// `Values` — U12.  DataFusion's unparser refuses these outright.
    pub(crate) fn lower_values(&mut self, values: &Values) -> Result<Lowered> {
        let mut l = Lowered {
            builder: SelectBuilder::new(),
            scope: Scope::new(),
            schema: Arc::clone(&values.schema),
        };
        let mut rows = Vec::with_capacity(values.values.len());
        for row in &values.values {
            rows.push(self.render_all(&mut l, row, Clause::Select, "Values")?);
        }

        let alias = self.names.fresh_rel();
        let cols: Vec<ast::TableAliasColumnDef> = values
            .schema
            .fields()
            .iter()
            .map(|f| ast::TableAliasColumnDef {
                name: ast::Ident::with_quote('"', f.name()),
                data_type: None,
            })
            .collect();

        if !self.dialect.supports_column_alias_in_table_alias() {
            return Err(SqlserError::unsupported(
                "Values",
                "dialect cannot name columns in a table alias",
            ));
        }

        let query = ast::Query {
            with: None,
            body: Box::new(ast::SetExpr::Values(ast::Values {
                explicit_row: false,
                value_keyword: false,
                rows: rows
                    .into_iter()
                    .map(|content| sqlparser::ast::Parens {
                        opening_token: ast::helpers::attached_token::AttachedToken::empty(),
                        content,
                        closing_token: ast::helpers::attached_token::AttachedToken::empty(),
                    })
                    .collect(),
            })),
            order_by: None,
            limit_clause: None,
            fetch: None,
            locks: vec![],
            for_clause: None,
            settings: None,
            format_clause: None,
            pipe_operators: vec![],
        };

        let factor = ast::TableFactor::Derived {
            lateral: false,
            subquery: Box::new(query),
            alias: Some(ast::TableAlias {
                explicit: true,
                name: alias.clone(),
                columns: cols,
                at: None,
            }),
            sample: None,
        };

        Ok(Lowered {
            builder: SelectBuilder::over(Relation { factor, alias: alias.clone() }),
            scope: scope_over_relation(&values.schema, &alias),
            schema: Arc::clone(&values.schema),
        })
    }

    pub(crate) fn lower_empty(&mut self, e: &EmptyRelation) -> Result<Lowered> {
        let mut builder = SelectBuilder::new();
        builder.set_no_from();
        if !e.produce_one_row {
            // A relation with no rows: select the right column types, and a
            // predicate that is never true.
            builder.and_where(ast::Expr::Value(ast::Value::Boolean(false).into()));
        }
        let entries = e
            .schema
            .iter()
            .map(|(q, f)| ScopeEntry {
                qualifier: q.cloned(),
                name: f.name().clone(),
                addr: ColumnAddr::Inline {
                    ast: Box::new(ast::Expr::Value(ast::Value::Null.into())),
                },
            })
            .collect();
        Ok(Lowered {
            builder,
            scope: Scope::from_entries(entries),
            schema: Arc::clone(&e.schema),
        })
    }

    // -- the common chain ---------------------------------------------------

    pub(crate) fn lower_projection(&mut self, p: &Projection) -> Result<Lowered> {
        let mut l = self.lower(&p.input)?;

        // A select list can only be written once, and never after DISTINCT has
        // narrowed the rows.
        if l.builder.select_is_set() || l.builder.distinct_is_set() {
            self.seal_into(&mut l)?;
        }

        let asts = self.render_all(&mut l, &p.expr, Clause::Select, "Projection")?;
        let (items, scope) = self.project(&asts, &p.schema)?;
        l.builder.set_select(items)?;
        l.scope = scope;
        Ok(l)
    }

    /// Build a select list and the matching scope from already-rendered
    /// expressions.
    ///
    /// Item *i* is field *i* of the schema, under field *i*'s name. Ordering
    /// is taken from the schema, never from the expression list's own order,
    /// so an arm that iterates its expressions in some other order cannot
    /// silently rotate the output — which is U3.
    fn project(
        &mut self,
        asts: &[ast::Expr],
        schema: &DFSchema,
    ) -> Result<(Vec<ast::SelectItem>, Scope)> {
        if asts.len() != schema.fields().len() {
            return Err(SqlserError::invariant(format!(
                "projection has {} expressions for {} schema fields",
                asts.len(),
                schema.fields().len()
            )));
        }
        let mut used = Vec::new();
        let mut items = Vec::with_capacity(asts.len());
        let mut entries = Vec::with_capacity(asts.len());
        for (i, (q, f)) in schema.iter().enumerate() {
            let alias_text = unique_name(
                &crate::builder::safe_name(f.name(), &mut self.names),
                &mut used,
            );
            let alias = ast::Ident::with_quote('"', &alias_text);
            items.push(ast::SelectItem::ExprWithAlias {
                expr: asts[i].clone(),
                alias: alias.clone(),
            });
            entries.push(ScopeEntry {
                qualifier: q.cloned(),
                name: f.name().clone(),
                addr: ColumnAddr::Projected {
                    alias,
                    ast: Box::new(asts[i].clone()),
                    kind: ProjKind::Plain,
                },
            });
        }
        Ok((items, Scope::from_entries(entries)))
    }

    pub(crate) fn lower_filter(&mut self, f: &Filter) -> Result<Lowered> {
        let mut l = self.lower(&f.input)?;

        // Which clause a predicate belongs in is decided by what the builder
        // already holds, not by guessing from the predicate's shape.
        let aggregated = l.builder.has_agg();
        let windowed = scope_has_window(&l.scope);

        if windowed && self.dialect.supports_qualify() && l.builder.stage() <= Stage::Qualify {
            let preds = self.render_all(
                &mut l,
                std::slice::from_ref(&f.predicate),
                Clause::Window,
                "Filter",
            )?;
            for p in preds {
                l.builder.and_qualify(p);
            }
            return Ok(l);
        }

        let clause = if aggregated { Clause::Having } else { Clause::Where };
        if !aggregated && l.builder.stage() > Stage::Where {
            self.seal_into(&mut l)?;
        }

        let clause = if l.builder.has_agg() { clause } else { Clause::Where };
        let preds = self.render_all(
            &mut l,
            std::slice::from_ref(&f.predicate),
            clause,
            "Filter",
        )?;
        for p in preds {
            // Accumulate. There is no setter, so whatever the input
            // contributed survives — U5 is unrepresentable here.
            match clause {
                Clause::Having => l.builder.and_having(p),
                _ => l.builder.and_where(p),
            }
        }
        Ok(l)
    }

    pub(crate) fn lower_aggregate(&mut self, a: &Aggregate) -> Result<Lowered> {
        let mut l = self.lower(&a.input)?;
        if l.builder.group_by_is_set() || l.builder.stage() > Stage::GroupBy {
            self.seal_into(&mut l)?;
        }

        let grouping = Grouping::of(&a.group_expr);
        let support = self.dialect.grouping_support();
        let ok = match grouping.kind {
            GroupingKind::Plain => true,
            GroupingKind::Rollup => support.rollup,
            GroupingKind::Cube => support.cube,
            GroupingKind::Sets(_) => support.grouping_sets,
        };
        if !ok {
            return Err(SqlserError::unsupported(
                "Aggregate",
                "dialect has no GROUP BY extension for this grouping set",
            ));
        }
        let group_exprs = grouping.exprs.clone();
        let uses_grouping_set = grouping.kind != GroupingKind::Plain;

        // Group keys and aggregate arguments alike must address real columns,
        // so both are resolved at the strictest clause.
        let group_asts = self.render_all(&mut l, &group_exprs, Clause::GroupBy, "Aggregate")?;
        let aggr_asts = self.render_all(&mut l, &a.aggr_expr, Clause::GroupBy, "Aggregate")?;

        if group_asts.is_empty() {
            l.builder.mark_aggregated();
        } else {
            l.builder
                .set_group_by(grouping.to_sql(&group_asts), None)?;
        }

        // `Aggregate::try_new` builds its schema as group_expr, then — for a
        // grouping set only — an internal `__grouping_id`, then aggr_expr.
        // Matching that order here is the whole of U3: DataFusion's unparser
        // chains them the other way round and silently rotates the result.
        let mut entries = Vec::with_capacity(a.schema.fields().len());
        let grouping_id: Vec<(&ast::Expr, ProjKind)> = Vec::new();
        let grouping_id_ast;
        let all: Box<dyn Iterator<Item = (&ast::Expr, ProjKind)>> = if uses_grouping_set {
            // `__grouping_id` is DataFusion's own bookkeeping column. SQL's
            // equivalent is `GROUPING(...)`. It stays an inlinable expression,
            // so unless something above actually reads it — nothing in normal
            // SQL does — it is never emitted at all.
            grouping_id_ast = crate::expr::call("GROUPING", group_asts.clone());
            let _ = &grouping_id;
            Box::new(
                group_asts
                    .iter()
                    .map(|e| (e, ProjKind::Plain))
                    .chain(std::iter::once((&grouping_id_ast, ProjKind::Plain)))
                    .chain(aggr_asts.iter().map(|e| (e, ProjKind::Aggregate))),
            )
        } else {
            Box::new(
                group_asts
                    .iter()
                    .map(|e| (e, ProjKind::Plain))
                    .chain(aggr_asts.iter().map(|e| (e, ProjKind::Aggregate))),
            )
        };
        for ((q, f), (ast_expr, kind)) in a.schema.iter().zip(all) {
            entries.push(ScopeEntry {
                qualifier: q.cloned(),
                name: f.name().clone(),
                addr: ColumnAddr::Projected {
                    alias: ast::Ident::with_quote('"', f.name()),
                    ast: Box::new(ast_expr.clone()),
                    kind,
                },
            });
        }
        if entries.len() != a.schema.fields().len() {
            return Err(SqlserError::invariant(format!(
                "aggregate produced {} scope entries for {} schema fields",
                entries.len(),
                a.schema.fields().len()
            )));
        }
        l.scope = Scope::from_entries(entries);
        Ok(l)
    }

    pub(crate) fn lower_sort(&mut self, s: &Sort) -> Result<Lowered> {
        let mut l = self.lower(&s.input)?;
        // A sort below an already-written LIMIT would change which rows that
        // LIMIT selected, so it seals.
        if l.builder.order_by_is_set() || l.builder.limit_is_set() {
            self.seal_into(&mut l)?;
        }

        let exprs: Vec<Expr> = s.expr.iter().map(|se| se.expr.clone()).collect();
        let asts = self.render_all(&mut l, &exprs, Clause::OrderBy, "Sort")?;
        let keys = s
            .expr
            .iter()
            .zip(asts)
            .map(|(se, expr)| ast::OrderByExpr {
                expr,
                options: ast::OrderByOptions {
                    asc: Some(se.asc),
                    nulls_first: Some(se.nulls_first),
                },
                with_fill: None,
            })
            .collect();
        l.builder.set_order_by(keys)?;

        // `push_down_limit` sets `Sort.fetch = skip + fetch` so the sort keeps
        // enough rows for the offset above it. That number is the sort's own
        // business; it becomes the query's LIMIT only if nothing else claims
        // the slot. Treating it as a LIMIT unconditionally is U4.
        if let Some(n) = s.fetch {
            l.builder.set_fetch_hint(n as u64);
        }
        Ok(l)
    }

    pub(crate) fn lower_limit(&mut self, lim: &Limit) -> Result<Lowered> {
        let mut l = self.lower(&lim.input)?;
        if l.builder.limit_is_set() {
            self.seal_into(&mut l)?;
        }

        let render_opt = |ser: &mut Self, e: &Option<Box<Expr>>, l: &mut Lowered| -> Result<Option<ast::Expr>> {
            match e {
                None => Ok(None),
                Some(e) => {
                    let v = ser.render_all(l, std::slice::from_ref(e), Clause::Select, "Limit")?;
                    Ok(v.into_iter().next())
                }
            }
        };
        let fetch = render_opt(self, &lim.fetch, &mut l)?;
        let skip = render_opt(self, &lim.skip, &mut l)?;
        let skip = skip.filter(|s| s.to_string() != "0");
        l.builder.set_limit(fetch, skip)?;
        Ok(l)
    }

    pub(crate) fn lower_distinct(&mut self, d: &Distinct) -> Result<Lowered> {
        match d {
            Distinct::All(input) => {
                // `SELECT DISTINCT` over a UNION is spelled `UNION` — folding
                // it in avoids a pointless wrapper.
                if let LogicalPlan::Union(u) = input.as_ref() {
                    return self.lower_union_with(u, ast::SetQuantifier::None);
                }
                let mut l = self.lower(input)?;
                if l.builder.distinct_is_set()
                    || l.builder.select_is_set()
                    || l.builder.order_by_is_set()
                {
                    self.seal_into(&mut l)?;
                }
                let (items, _) = identity_projection(&l.scope, &mut self.names)?;
                l.builder.set_select(items)?;
                l.builder.set_distinct(ast::Distinct::Distinct)?;
                Ok(l)
            }
            Distinct::On(on) => {
                if self.dialect.distinct_on_style() != DistinctOnStyle::Native {
                    return Err(SqlserError::unsupported(
                        "Distinct::On",
                        "dialect has no DISTINCT ON; the row_number() rewrite is not implemented",
                    ));
                }
                let mut l = self.lower(&on.input)?;
                if l.builder.distinct_is_set()
                    || l.builder.select_is_set()
                    || l.builder.order_by_is_set()
                {
                    self.seal_into(&mut l)?;
                }
                let on_asts = self.render_all(&mut l, &on.on_expr, Clause::Select, "DistinctOn")?;
                let sel_asts =
                    self.render_all(&mut l, &on.select_expr, Clause::Select, "DistinctOn")?;
                let (items, scope) = self.project(&sel_asts, &on.schema)?;
                l.builder.set_select(items)?;
                l.scope = scope;
                l.builder.set_distinct(ast::Distinct::On(on_asts))?;
                if let Some(sort) = &on.sort_expr {
                    let exprs: Vec<Expr> = sort.iter().map(|s| s.expr.clone()).collect();
                    let asts = self.render_all(&mut l, &exprs, Clause::OrderBy, "DistinctOn")?;
                    let keys = sort
                        .iter()
                        .zip(asts)
                        .map(|(se, expr)| ast::OrderByExpr {
                            expr,
                            options: ast::OrderByOptions {
                                asc: Some(se.asc),
                                nulls_first: Some(se.nulls_first),
                            },
                            with_fill: None,
                        })
                        .collect();
                    l.builder.set_order_by(keys)?;
                }
                Ok(l)
            }
        }
    }

    /// `SubqueryAlias` — U7.
    ///
    /// The alias names the plan's schema, not a relation to emit: the relation
    /// underneath already carries a generated alias of its own. Emitting the
    /// plan alias instead, as DataFusion does, produces SQL that names the
    /// *inner* alias of a stacked pair while every column reference above is
    /// qualified by the outer one — which is the OMP / view-inlining failure.
    pub(crate) fn lower_subquery_alias(&mut self, a: &SubqueryAlias) -> Result<Lowered> {
        let mut l = self.lower(&a.input)?;
        l.scope = l.scope.requalify(&a.alias);
        Ok(l)
    }

    pub(crate) fn lower_window(&mut self, w: &Window) -> Result<Lowered> {
        let mut l = self.lower(&w.input)?;
        if l.builder.select_is_set() || l.builder.stage() > Stage::Window {
            self.seal_into(&mut l)?;
        }
        let asts = self.render_all(&mut l, &w.window_expr, Clause::Window, "Window")?;

        // `Window.schema` is the input's fields followed by the window ones.
        let mut entries: Vec<ScopeEntry> = l.scope.iter().cloned().collect();
        let n_input = entries.len();
        for (i, ast_expr) in asts.iter().enumerate() {
            let (q, f) = w.schema.qualified_field(n_input + i);
            // The alias is fixed here, at creation. DataFusion's long display
            // name ("row_number() PARTITION BY [...] ...") stays a scope key
            // and never becomes an identifier, which is U8.
            entries.push(ScopeEntry {
                qualifier: q.cloned(),
                name: f.name().clone(),
                addr: ColumnAddr::Projected {
                    alias: self.names.fresh_col("win"),
                    ast: Box::new(ast_expr.clone()),
                    kind: ProjKind::Window,
                },
            });
        }
        l.scope = Scope::from_entries(entries);
        Ok(l)
    }

    pub(crate) fn lower_recursive(&mut self, r: &RecursiveQuery) -> Result<Lowered> {
        let cte_name = ast::Ident::with_quote('"', &r.name);

        let static_q = self.lower_to_query(&r.static_term)?;
        // The recursive term references the CTE by name, so it must be in
        // scope as a relation before that side is lowered.
        self.cte_names.push(r.name.clone());
        let recursive_q = self.lower_to_query(&r.recursive_term);
        self.cte_names.pop();
        let recursive_q = recursive_q?;

        let body = ast::SetExpr::SetOperation {
            op: ast::SetOperator::Union,
            set_quantifier: if r.is_distinct {
                ast::SetQuantifier::None
            } else {
                ast::SetQuantifier::All
            },
            left: Box::new(ast::SetExpr::Query(Box::new(static_q))),
            right: Box::new(ast::SetExpr::Query(Box::new(recursive_q))),
        };

        let cols: Vec<ast::TableAliasColumnDef> = r
            .schema
            .fields()
            .iter()
            .map(|f| ast::TableAliasColumnDef {
                name: ast::Ident::with_quote('"', f.name()),
                data_type: None,
            })
            .collect();

        self.ctes_recursive = true;
        self.ctes.push(ast::Cte {
            alias: ast::TableAlias {
                // A CTE is `WITH name (cols) AS (...)`, not `WITH AS name`.
                explicit: false,
                name: cte_name.clone(),
                columns: cols,
                at: None,
            },
            query: Box::new(ast::Query {
                with: None,
                body: Box::new(body),
                order_by: None,
                limit_clause: None,
                fetch: None,
                locks: vec![],
                for_clause: None,
                settings: None,
                format_clause: None,
                pipe_operators: vec![],
            }),
            from: None,
            materialized: None,
            closing_paren_token: ast::helpers::attached_token::AttachedToken::empty(),
        });

        let alias = self.names.fresh_rel();
        let factor = table_factor(
            ast::ObjectName(vec![ast::ObjectNamePart::Identifier(cte_name)]),
            alias.clone(),
        );
        Ok(Lowered {
            builder: SelectBuilder::over(Relation { factor, alias: alias.clone() }),
            scope: scope_over_relation(&r.schema, &alias),
            schema: Arc::clone(&r.schema),
        })
    }

    /// `Unnest` — one of the three operators DataFusion's unparser refuses
    /// outright.
    ///
    /// The plan shape is already the SQL shape: a projection computes the list
    /// column, and `Unnest` expands it. So the input is sealed and each output
    /// column is either `unnest(...)` over its source or a plain passthrough,
    /// chosen by `dependency_indices`, which is the plan's own statement of
    /// which input column each output column came from.
    pub(crate) fn lower_unnest(&mut self, u: &datafusion::logical_expr::Unnest) -> Result<Lowered> {
        if !self.dialect.unnest_in_select_list() {
            return Err(SqlserError::unsupported(
                "Unnest",
                "dialect has no set-returning unnest in the select list",
            ));
        }
        if !u.struct_type_columns.is_empty() {
            return Err(SqlserError::unsupported(
                "Unnest",
                "struct unnesting is not implemented; refusing rather than guessing",
            ));
        }
        if u.dependency_indices.len() != u.schema.fields().len() {
            return Err(SqlserError::invariant(
                "unnest dependency_indices does not cover its schema",
            ));
        }

        let inner = self.lower(&u.input)?;
        let (rel, in_scope) = self.seal(inner)?;
        let mut builder = SelectBuilder::over(rel);

        let mut used = Vec::new();
        let mut items = Vec::with_capacity(u.schema.fields().len());
        let mut entries = Vec::with_capacity(u.schema.fields().len());

        for (i, (q, f)) in u.schema.iter().enumerate() {
            let src = u.dependency_indices[i];
            let base = match in_scope.resolve_index(src, Clause::Select) {
                crate::scope::Resolution::Ast(a) => a,
                _ => {
                    return Err(SqlserError::invariant(
                        "unnest source column is not addressable after sealing",
                    ));
                }
            };
            // A list column is wrapped once per level of nesting; anything
            // else rides through untouched.
            let depth = u
                .list_type_columns
                .iter()
                .find(|(idx, _)| *idx == src)
                .map(|(_, l)| l.depth)
                .unwrap_or(0);
            let mut expr = base;
            for _ in 0..depth {
                expr = crate::expr::call("unnest", vec![expr]);
            }

            let alias_text = unique_name(
                &crate::builder::safe_name(f.name(), &mut self.names),
                &mut used,
            );
            let alias = ast::Ident::with_quote('"', &alias_text);
            items.push(ast::SelectItem::ExprWithAlias {
                expr: expr.clone(),
                alias: alias.clone(),
            });
            entries.push(ScopeEntry {
                qualifier: q.cloned(),
                name: f.name().clone(),
                addr: ColumnAddr::Projected {
                    alias,
                    ast: Box::new(expr),
                    kind: ProjKind::Plain,
                },
            });
        }

        builder.set_select(items)?;
        Ok(Lowered {
            builder,
            scope: Scope::from_entries(entries),
            schema: Arc::clone(&u.schema),
        })
    }

    // -- set operations -----------------------------------------------------

    pub(crate) fn lower_union(&mut self, u: &Union) -> Result<Lowered> {
        self.lower_union_with(u, ast::SetQuantifier::All)
    }

    /// `Union` is always `UNION ALL` in DataFusion; deduplication is modelled
    /// as `Distinct(Union)`, which folds into the quantifier here.
    ///
    /// Branch select lists are built **by position**, not by name: the union's
    /// own schema drops every qualifier and takes names from the first branch,
    /// so branch two's field two may be called something else entirely. Lining
    /// the branches up by name would quietly permute the columns.
    fn lower_union_with(&mut self, u: &Union, quantifier: ast::SetQuantifier) -> Result<Lowered> {
        let mut branches = Vec::with_capacity(u.inputs.len());
        for input in &u.inputs {
            branches.push(self.branch_query(input, &u.schema)?);
        }

        let mut iter = branches.into_iter();
        let first = iter
            .next()
            .ok_or_else(|| SqlserError::invariant("union with no inputs"))?;
        let mut body = ast::SetExpr::Query(Box::new(first));
        for next in iter {
            body = ast::SetExpr::SetOperation {
                op: ast::SetOperator::Union,
                set_quantifier: quantifier,
                left: Box::new(body),
                right: Box::new(ast::SetExpr::Query(Box::new(next))),
            };
        }

        let alias = self.names.fresh_rel();
        let factor = ast::TableFactor::Derived {
            lateral: false,
            subquery: Box::new(ast::Query {
                with: None,
                body: Box::new(body),
                order_by: None,
                limit_clause: None,
                fetch: None,
                locks: vec![],
                for_clause: None,
                settings: None,
                format_clause: None,
                pipe_operators: vec![],
            }),
            alias: Some(ast::TableAlias {
                explicit: true,
                name: alias.clone(),
                columns: vec![],
                at: None,
            }),
            sample: None,
        };

        Ok(Lowered {
            builder: SelectBuilder::over(Relation { factor, alias: alias.clone() }),
            scope: scope_over_relation(&u.schema, &alias),
            schema: Arc::clone(&u.schema),
        })
    }

    /// Lower one union branch and force its select list to the union's schema,
    /// position by position.
    fn branch_query(&mut self, input: &LogicalPlan, out: &DFSchema) -> Result<ast::Query> {
        let mut l = self.lower(input)?;
        if l.builder.select_is_set() {
            self.seal_into(&mut l)?;
        }
        if l.scope.len() != out.fields().len() {
            return Err(SqlserError::invariant(format!(
                "union branch exposes {} columns, the union schema has {}",
                l.scope.len(),
                out.fields().len()
            )));
        }
        let mut used = Vec::new();
        let mut items = Vec::with_capacity(out.fields().len());
        for i in 0..out.fields().len() {
            let expr = match l.scope.resolve_index(i, Clause::Select) {
                crate::scope::Resolution::Ast(a) => a,
                _ => {
                    self.seal_into(&mut l)?;
                    match l.scope.resolve_index(i, Clause::Select) {
                        crate::scope::Resolution::Ast(a) => a,
                        _ => {
                            return Err(SqlserError::invariant("union branch column unaddressable"));
                        }
                    }
                }
            };
            let alias_text = unique_name(out.field(i).name(), &mut used);
            items.push(ast::SelectItem::ExprWithAlias {
                expr,
                alias: ast::Ident::with_quote('"', &alias_text),
            });
        }
        l.builder.set_select(items)?;
        l.builder.finish(self.dialect)
    }

    // -- joins --------------------------------------------------------------

    pub(crate) fn lower_join(&mut self, j: &Join) -> Result<Lowered> {
        match j.join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => self.lower_semi_join(j, true),
            JoinType::RightSemi | JoinType::RightAnti => self.lower_semi_join(j, false),
            JoinType::LeftMark | JoinType::RightMark => self.lower_mark_join(j),
            _ => self.lower_plain_join(j),
        }
    }

    fn lower_plain_join(&mut self, j: &Join) -> Result<Lowered> {
        let mut left = self.lower(&j.left)?;

        // An outer join that does not preserve the left side cannot have the
        // left's WHERE applied after it, so that side is sealed first.
        let preserves_left = matches!(j.join_type, JoinType::Inner | JoinType::Left);
        let must_seal_left = if preserves_left {
            left.builder.stage() > Stage::Where
        } else {
            left.builder.stage() > Stage::From
        };
        if must_seal_left {
            self.seal_into(&mut left)?;
        }

        let right = self.lower(&j.right)?;
        let (right_rel, right_scope) = self.seal(right)?;

        // The ON predicate sees both sides, so it is rendered against the
        // concatenated scope — the same scope the join's own schema describes.
        let joined = Scope::concat(&left.scope, &right_scope);
        let mut probe = Lowered {
            builder: std::mem::take(&mut left.builder),
            scope: joined,
            schema: Arc::clone(&j.schema),
        };

        let mut on_exprs: Vec<Expr> = Vec::new();
        for (l, r) in &j.on {
            on_exprs.push(l.clone());
            on_exprs.push(r.clone());
        }
        let on_asts = self.render_all(&mut probe, &on_exprs, Clause::From, "Join")?;
        let mut conds: Vec<ast::Expr> = Vec::new();
        for pair in on_asts.chunks(2) {
            conds.push(ast::Expr::BinaryOp {
                left: Box::new(pair[0].clone()),
                op: match j.null_equality {
                    NullEquality::NullEqualsNull => match self.dialect.null_safe_equality() {
                        NullSafeEquality::Spaceship => ast::BinaryOperator::Spaceship,
                        NullSafeEquality::IsNotDistinctFrom => {
                            // Rendered as an operator below; the AST has a
                            // dedicated node for it.
                            ast::BinaryOperator::Eq
                        }
                    },
                    NullEquality::NullEqualsNothing => ast::BinaryOperator::Eq,
                },
                right: Box::new(pair[1].clone()),
            });
            if j.null_equality == NullEquality::NullEqualsNull
                && self.dialect.null_safe_equality() == NullSafeEquality::IsNotDistinctFrom
            {
                let last = conds.pop().expect("just pushed");
                if let ast::Expr::BinaryOp { left, right, .. } = last {
                    conds.push(ast::Expr::IsNotDistinctFrom(left, right));
                }
            }
        }
        if let Some(filter) = &j.filter {
            let f = self.render_all(
                &mut probe,
                std::slice::from_ref(filter),
                Clause::From,
                "Join",
            )?;
            conds.extend(f);
        }

        let constraint = match crate::builder::conjoin(conds) {
            Some(e) => ast::JoinConstraint::On(e),
            None => ast::JoinConstraint::None,
        };
        let operator = match (j.join_type, &constraint) {
            (JoinType::Inner, ast::JoinConstraint::None) => {
                ast::JoinOperator::CrossJoin(ast::JoinConstraint::None)
            }
            (JoinType::Inner, _) => ast::JoinOperator::Inner(constraint),
            (JoinType::Left, _) => ast::JoinOperator::LeftOuter(constraint),
            (JoinType::Right, _) => ast::JoinOperator::RightOuter(constraint),
            (JoinType::Full, _) => ast::JoinOperator::FullOuter(constraint),
            (other, _) => {
                return Err(SqlserError::unsupported(
                    "Join",
                    format!("unexpected join type {other:?} on the plain path"),
                ));
            }
        };

        probe.builder.push_join(ast::Join {
            relation: right_rel.factor,
            global: false,
            join_operator: operator,
        })?;

        // A join's schema is its left fields followed by its right fields.
        debug_assert_eq!(probe.scope.len(), j.schema.fields().len());
        Ok(probe)
    }

    /// Semi and anti joins become `EXISTS` / `NOT EXISTS` in the driving
    /// side's `WHERE`.
    ///
    /// The predicate is **AND-ed in**, which is the whole of U5: DataFusion
    /// assigns it instead, so a `Filter` sitting on the driving input is
    /// dropped on the floor and the query silently returns too many rows.
    fn lower_semi_join(&mut self, j: &Join, left_drives: bool) -> Result<Lowered> {
        let negated = matches!(j.join_type, JoinType::LeftAnti | JoinType::RightAnti);

        let (driving, inner) = if left_drives {
            (&j.left, &j.right)
        } else {
            (&j.right, &j.left)
        };

        let mut l = self.lower(driving)?;
        if l.builder.stage() > Stage::Where {
            self.seal_into(&mut l)?;
        }

        // `NOT EXISTS` is not equivalent to `NOT IN` once the probe side can
        // be NULL, and this join carries the flag saying it came from `NOT
        // IN`. Rendering the shape it actually means is the only honest
        // option; emitting `NOT EXISTS` anyway would be a silent wrong answer
        // of exactly the kind this crate exists to rule out.
        let predicate = if negated && j.null_aware {
            self.not_in_predicate(j, &l.scope, inner, left_drives)?
        } else {
            self.exists_predicate(j, &l.scope, inner, negated)?
        };
        l.builder.and_where(predicate);
        debug_assert_eq!(l.scope.len(), j.schema.fields().len());
        Ok(l)
    }

    /// Mark joins carry a synthetic boolean column. Keeping it as an inlinable
    /// expression rather than a named column is U9: DataFusion emits the
    /// literal name `mark`, qualified by a subquery that is not a relation,
    /// several times over.
    fn lower_mark_join(&mut self, j: &Join) -> Result<Lowered> {
        let left_drives = j.join_type == JoinType::LeftMark;
        let (driving, inner) = if left_drives {
            (&j.left, &j.right)
        } else {
            (&j.right, &j.left)
        };

        let mut l = self.lower(driving)?;
        if l.builder.stage() > Stage::Where {
            self.seal_into(&mut l)?;
        }
        let exists = self.exists_predicate(j, &l.scope, inner, false)?;

        let idx = l.scope.len();
        let (q, f) = j.schema.qualified_field(idx);
        let mut entries: Vec<ScopeEntry> = l.scope.iter().cloned().collect();
        entries.push(ScopeEntry {
            qualifier: q.cloned(),
            name: f.name().clone(),
            addr: ColumnAddr::Inline {
                ast: Box::new(exists),
            },
        });
        l.scope = Scope::from_entries(entries);
        Ok(l)
    }

    /// Build `<probe> NOT IN (SELECT <key> FROM <inner>)`, which — unlike
    /// `NOT EXISTS` — yields NULL rather than true when the inner side
    /// contains a NULL key.
    fn not_in_predicate(
        &mut self,
        j: &Join,
        outer_scope: &Scope,
        inner: &Arc<LogicalPlan>,
        left_drives: bool,
    ) -> Result<ast::Expr> {
        if j.on.len() != 1 || j.filter.is_some() {
            return Err(SqlserError::unsupported(
                "null-aware anti join",
                "only a single un-filtered key can be rendered as NOT IN",
            ));
        }
        let (probe_expr, key_expr) = if left_drives {
            (&j.on[0].0, &j.on[0].1)
        } else {
            (&j.on[0].1, &j.on[0].0)
        };

        let mut probe_l = Lowered {
            builder: SelectBuilder::new(),
            scope: outer_scope.clone(),
            schema: j.left.schema().clone(),
        };
        let probe = self
            .render_all(&mut probe_l, std::slice::from_ref(probe_expr), Clause::Where, "NotIn")?
            .remove(0);

        let mut inner_l = self.lower(inner)?;
        // Seal first, then render: a key rendered against the pre-seal scope
        // would name a relation the sealed query no longer exposes.
        if inner_l.builder.select_is_set() {
            self.seal_into(&mut inner_l)?;
        }
        let key = self
            .render_all(&mut inner_l, std::slice::from_ref(key_expr), Clause::Select, "NotIn")?
            .remove(0);
        inner_l
            .builder
            .set_select(vec![ast::SelectItem::UnnamedExpr(key)])?;
        let subquery = inner_l.builder.finish(self.dialect)?;

        Ok(ast::Expr::InSubquery {
            expr: Box::new(probe),
            subquery: Box::new(subquery),
            negated: true,
        })
    }

    /// Build `[NOT] EXISTS (SELECT 1 FROM <inner> WHERE <on ∧ filter>)`,
    /// correlated against `outer_scope`.
    fn exists_predicate(
        &mut self,
        j: &Join,
        outer_scope: &Scope,
        inner: &Arc<LogicalPlan>,
        negated: bool,
    ) -> Result<ast::Expr> {
        self.outers.push(outer_scope.clone());
        let built = (|| -> Result<ast::Query> {
            let mut inner_l = self.lower(inner)?;

            let mut on_exprs: Vec<Expr> = Vec::new();
            for (l, r) in &j.on {
                on_exprs.push(l.clone());
                on_exprs.push(r.clone());
            }
            if let Some(f) = &j.filter {
                on_exprs.push(f.clone());
            }

            // Rendered against the inner scope; anything it cannot find falls
            // through to the outer stack, which is the correlation.
            let asts = self.render_all(&mut inner_l, &on_exprs, Clause::Where, "SemiJoin")?;
            let (pairs, extra) = asts.split_at(j.on.len() * 2);
            let mut conds: Vec<ast::Expr> = pairs
                .chunks(2)
                .map(|p| ast::Expr::BinaryOp {
                    left: Box::new(p[0].clone()),
                    op: ast::BinaryOperator::Eq,
                    right: Box::new(p[1].clone()),
                })
                .collect();
            conds.extend(extra.iter().cloned());
            for c in conds {
                inner_l.builder.and_where(c);
            }

            if !inner_l.builder.select_is_set() {
                inner_l.builder.set_select(vec![ast::SelectItem::UnnamedExpr(
                    ast::Expr::Value(ast::Value::Number("1".into(), false).into()),
                )])?;
            }
            inner_l.builder.finish(self.dialect)
        })();
        self.outers.pop();

        Ok(ast::Expr::Exists {
            subquery: Box::new(built?),
            negated,
        })
    }

    // -- shared -------------------------------------------------------------

    fn seal_into(&mut self, l: &mut Lowered) -> Result<()> {
        self.seal_in_place_pub(l)
    }

    /// Lower a plan to a standalone `Query` with an explicit select list.
    fn lower_to_query(&mut self, plan: &LogicalPlan) -> Result<ast::Query> {
        let mut l = self.lower(plan)?;
        if !l.builder.select_is_set() {
            let (items, _) = identity_projection(&l.scope, &mut self.names)?;
            l.builder.set_select(items)?;
        }
        l.builder.finish(self.dialect)
    }
}

/// Every column of the scanned table, addressable on the scan's own relation.
fn scan_scope(scan: &TableScan, alias: &ast::Ident) -> Scope {
    Scope::from_entries(
        scan.source
            .schema()
            .fields()
            .iter()
            .map(|f| ScopeEntry {
                qualifier: Some(scan.table_name.clone()),
                name: f.name().clone(),
                addr: ColumnAddr::Column {
                    rel: alias.clone(),
                    col: ast::Ident::with_quote('"', f.name()),
                },
            })
            .collect(),
    )
}

fn scope_has_window(scope: &Scope) -> bool {
    scope.iter().any(|e| {
        matches!(
            &e.addr,
            ColumnAddr::Projected {
                kind: ProjKind::Window,
                ..
            }
        )
    })
}

/// Split a `GroupingSet` into plain expressions plus the SQL modifier.
#[derive(Debug, PartialEq)]
enum GroupingKind {
    Plain,
    Rollup,
    Cube,
    /// Each set given as indices into the flattened expression list, so the
    /// sets survive rendering without being matched up by name afterwards.
    Sets(Vec<Vec<usize>>),
}

/// A `GROUP BY` clause, flattened into one expression list plus the shape that
/// list should be rendered in.
///
/// The flattening matters: expressions are rendered once, in a single batch,
/// and the sets are rebuilt from positions. Rendering each set separately
/// would mean rendering the same expression more than once, and there is no
/// guarantee two renderings of the same expression agree once a seal has
/// happened in between.
#[derive(Debug)]
struct Grouping {
    exprs: Vec<Expr>,
    kind: GroupingKind,
}

impl Grouping {
    fn of(group_expr: &[Expr]) -> Self {
        let [Expr::GroupingSet(gs)] = group_expr else {
            return Self {
                exprs: group_expr.to_vec(),
                kind: GroupingKind::Plain,
            };
        };
        match gs {
            GroupingSet::Rollup(e) => Self {
                exprs: e.clone(),
                kind: GroupingKind::Rollup,
            },
            GroupingSet::Cube(e) => Self {
                exprs: e.clone(),
                kind: GroupingKind::Cube,
            },
            GroupingSet::GroupingSets(sets) => {
                let mut exprs: Vec<Expr> = Vec::new();
                let mut idx_sets = Vec::with_capacity(sets.len());
                for set in sets {
                    let mut ids = Vec::with_capacity(set.len());
                    for e in set {
                        let pos = exprs.iter().position(|x| x == e).unwrap_or_else(|| {
                            exprs.push(e.clone());
                            exprs.len() - 1
                        });
                        ids.push(pos);
                    }
                    idx_sets.push(ids);
                }
                Self {
                    exprs,
                    kind: GroupingKind::Sets(idx_sets),
                }
            }
        }
    }

    fn to_sql(&self, rendered: &[ast::Expr]) -> Vec<ast::Expr> {
        match &self.kind {
            GroupingKind::Plain => rendered.to_vec(),
            GroupingKind::Rollup => vec![ast::Expr::Rollup(
                rendered.iter().map(|e| vec![e.clone()]).collect(),
            )],
            GroupingKind::Cube => vec![ast::Expr::Cube(
                rendered.iter().map(|e| vec![e.clone()]).collect(),
            )],
            GroupingKind::Sets(sets) => vec![ast::Expr::GroupingSets(
                sets.iter()
                    .map(|ids| ids.iter().map(|i| rendered[*i].clone()).collect())
                    .collect(),
            )],
        }
    }
}
