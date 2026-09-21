//! Generated identifiers.
//!
//! Every relation `sqlser` emits carries a generated alias, and every derived
//! table is sealed under one.  Name generation is therefore load-bearing
//! rather than cosmetic: it is what makes an outer reference impossible to
//! capture by a relation introduced further in (U1, U7), and what keeps a
//! self-join's two legs distinguishable.

use sqlparser::ast::Ident;

/// Identifiers starting with this prefix are reserved for generated names.
/// Nothing derived from user input is allowed to collide with them.
pub(crate) const RESERVED: &str = "__sqlser";

/// Monotonic source of unique identifiers, one per serialization.
#[derive(Debug, Default)]
pub struct NameGen {
    rel: usize,
    col: usize,
    cte: usize,
    hole: usize,
}

impl NameGen {
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh relation alias, used for sealed derived tables and for every
    /// base-table reference.
    pub fn fresh_rel(&mut self) -> Ident {
        self.rel += 1;
        Ident::with_quote('"', format!("{RESERVED}_r{}", self.rel))
    }

    /// A fresh column alias.  `hint` is a human-readable tag that survives
    /// into the generated SQL to keep output legible.
    pub fn fresh_col(&mut self, hint: &str) -> Ident {
        self.col += 1;
        Ident::with_quote('"', format!("{RESERVED}_{hint}{}", self.col))
    }

    /// A fresh CTE name.
    pub fn fresh_cte(&mut self, hint: &str) -> Ident {
        self.cte += 1;
        Ident::with_quote('"', format!("{RESERVED}_{hint}{}", self.cte))
    }

    /// A fresh placeholder name for the expression splice mechanism.
    /// Unquoted, because it has to round-trip through DataFusion's expression
    /// renderer as a plain column reference.
    pub fn fresh_hole(&mut self) -> String {
        self.hole += 1;
        format!("{RESERVED}_hole{}", self.hole)
    }
}

/// Disambiguate `name` against the identifiers already used in a select list,
/// so a projection can never emit two columns with the same name.  This is
/// what stops U6's duplicate-`n_nationkey` round-trip failure.
pub fn unique_name(name: &str, used: &mut Vec<String>) -> String {
    if !used.iter().any(|u| u == name) {
        used.push(name.to_string());
        return name.to_string();
    }
    for n in 1.. {
        let candidate = format!("{name}__{n}");
        if !used.contains(&candidate) {
            used.push(candidate.clone());
            return candidate;
        }
    }
    unreachable!("infinite range yields a free name")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_names_are_unique_per_kind() {
        let mut g = NameGen::new();
        let a = g.fresh_rel();
        let b = g.fresh_rel();
        assert_ne!(a.value, b.value);
        assert!(a.value.starts_with(RESERVED));
        assert_ne!(g.fresh_col("win").value, g.fresh_col("win").value);
        assert_ne!(g.fresh_hole(), g.fresh_hole());
    }

    #[test]
    fn unique_name_disambiguates_repeats() {
        let mut used = Vec::new();
        assert_eq!(unique_name("n_name", &mut used), "n_name");
        assert_eq!(unique_name("n_name", &mut used), "n_name__1");
        assert_eq!(unique_name("n_name", &mut used), "n_name__2");
        assert_eq!(unique_name("other", &mut used), "other");
    }
}
