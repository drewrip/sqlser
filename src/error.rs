//! Errors.
//!
//! Every failure mode is a typed error.  `sqlser` never degrades to
//! best-effort SQL: a caller that gets `Ok` can rely on the text, and a caller
//! that gets `Err` knows to fall back.  This is the signal `BROKEN.md` says is
//! missing today for the silent-wrong-answer bugs (U3-U6).

use std::fmt;

/// The result type used throughout the crate.
pub type Result<T> = std::result::Result<T, SqlserError>;

#[derive(Debug, thiserror::Error)]
pub enum SqlserError {
    /// The plan contains an operator with no faithful SQL rendering.
    #[error("unsupported {node}: {reason}")]
    Unsupported { node: String, reason: String },

    /// An internal invariant was violated.  A bug in this crate, or a plan
    /// whose schema does not describe its own output.
    #[error("invariant violated: {0}")]
    Invariant(String),

    /// A column could not be resolved in the current scope or any enclosing
    /// one.  Carries enough context to locate the offending node.
    #[error("cannot resolve column `{column}` while lowering {node}")]
    UnresolvedColumn { column: String, node: String },

    /// Derived-table nesting exceeded `Config::max_derived_depth`.
    #[error("derived table nesting exceeded {limit}")]
    DepthExceeded { limit: usize },

    /// The delegated expression renderer failed.
    #[error("expression rendering failed: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
}

impl SqlserError {
    pub(crate) fn unsupported(node: impl fmt::Display, reason: impl fmt::Display) -> Self {
        Self::Unsupported {
            node: node.to_string(),
            reason: reason.to_string(),
        }
    }

    pub(crate) fn invariant(msg: impl fmt::Display) -> Self {
        Self::Invariant(msg.to_string())
    }
}
