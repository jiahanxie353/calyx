//! Frontend parsing and AST representation.
//!
//! Defines the frontend AST and the parser.
//! The frontend representation is transformed into the representation defined
//! in the `ir` module.

pub mod ast;
pub mod parser;
pub mod source_info;

mod attribute;
mod attributes;
mod common;
mod lib_sig;
mod workspace;

use attribute::InlineAttributes;

pub use ast::NamespaceDef;
pub use attribute::{
    Attribute, BoolAttr, DEPRECATED_ATTRIBUTES, InternalAttr, NumAttr, SetAttr,
    SetAttribute,
};
pub use attributes::{Attributes, GetAttributes};
pub use common::{Direction, PortDef, Primitive, Width};
pub use lib_sig::{LibrarySignatures, PrimitiveInfo};
pub use workspace::Workspace;
