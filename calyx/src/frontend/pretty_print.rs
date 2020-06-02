use super::colors::ColorHelper;
use crate::lang::ast::*;
use atty::Stream;
use pretty::termcolor::{ColorChoice, ColorSpec, StandardStream};
use pretty::RcDoc;
use std::fmt;
use std::fmt::Display;
use std::io;
use std::io::Write;
use std::vec::IntoIter;

pub trait PrettyHelper<'a>: Sized {
    fn surround(self, pre: &'a str, post: &'a str) -> Self;
    fn parens(self) -> Self {
        self.surround("(", ")")
    }
    fn brackets(self) -> Self {
        self.surround("[", "]")
    }
    fn braces(self) -> Self {
        self.surround("{", "}")
    }
}

impl<'a, A> PrettyHelper<'a> for RcDoc<'a, A> {
    fn surround(self, l: &'a str, r: &'a str) -> Self {
        RcDoc::text(l).append(self).append(RcDoc::text(r))
    }
}

fn block<'a>(
    name: RcDoc<'a, ColorSpec>,
    doc: RcDoc<'a, ColorSpec>,
) -> RcDoc<'a, ColorSpec> {
    name.append(RcDoc::space()).append(
        RcDoc::nil()
            .append(RcDoc::line())
            .append(doc)
            .nest(2)
            .append(RcDoc::line())
            .braces(),
    )
}

fn stmt_vec<'a>(
    vec: impl Iterator<Item = RcDoc<'a, ColorSpec>>,
) -> RcDoc<'a, ColorSpec> {
    let docs = vec.map(|s| s.append(RcDoc::text(";")));
    RcDoc::intersperse(docs, RcDoc::line())
}

pub fn display<W: Write>(doc: RcDoc<ColorSpec>, write: Option<W>) {
    if atty::is(Stream::Stdout) {
        doc.render_colored(100, StandardStream::stdout(ColorChoice::Auto))
            .unwrap();
    } else {
        match write {
            Some(mut w) => doc.render(100, &mut w).unwrap(),
            None => doc.render(100, &mut std::io::stdout()).unwrap(),
        }
    }
}

pub trait PrettyPrint {
    /// Convert `self` into an `RcDoc`. the `area` of type `&Bump`
    /// is provided in case objects need to be allocated while producing
    /// The RcDoc. Call `arena.alloc(obj)` to allocate `obj` and use the
    /// returned reference for printing.
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec>;
    fn pretty_print(&self) {
        // XXX(sam) this leaks memory atm because we put vecs into this
        let mut arena = bumpalo::Bump::new();
        {
            let str = self.prettify(&arena);
            if atty::is(Stream::Stdout) {
                str.render_colored(
                    100,
                    StandardStream::stdout(ColorChoice::Auto),
                )
                .unwrap();
            } else {
                str.render(100, &mut io::stdout()).unwrap();
            }
        }
        arena.reset();
    }
}

/* =============== Generic impls ================ */

impl PrettyPrint for u64 {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        RcDoc::text(self.to_string())
    }
}

impl<T: PrettyPrint> PrettyPrint for Vec<T> {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let docs = self.iter().map(|s| s.prettify(&arena));
        RcDoc::intersperse(docs, RcDoc::text(",").append(RcDoc::space()))
    }
}

/* =============== Toplevel ================ */

impl PrettyPrint for Id {
    fn prettify<'a>(&self, _arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        RcDoc::text(self.to_string())
    }
}

impl PrettyPrint for NamespaceDef {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let comps = self.components.iter().map(|s| s.prettify(&arena));
        RcDoc::intersperse(comps, RcDoc::line().append(RcDoc::hardline()))
    }
}

impl PrettyPrint for ComponentDef {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let name = RcDoc::text("component")
            .define_color()
            .append(RcDoc::space())
            .append(self.name.prettify(&arena).ident_color())
            .append(self.signature.prettify(&arena));
        let body = RcDoc::nil()
            .append(block(
                RcDoc::text("cells").keyword_color(),
                stmt_vec(self.cells.iter().map(|x| x.prettify(&arena))),
            ))
            .append(RcDoc::line())
            .append(RcDoc::line())
            .append(block(
                RcDoc::text("wires").keyword_color(),
                stmt_vec(self.connections.iter().map(|x| x.prettify(&arena))),
            ))
            .append(RcDoc::line())
            .append(RcDoc::line())
            .append(block(
                RcDoc::text("control").keyword_color(),
                self.control.prettify(&arena),
            ));
        block(name, body)
    }
}

impl PrettyPrint for Signature {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        self.inputs
            .prettify(&arena)
            .parens()
            .append(RcDoc::space())
            .append(RcDoc::text("->"))
            .append(RcDoc::space())
            .append(self.outputs.prettify(&arena).parens())
    }
}

impl PrettyPrint for Portdef {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        self.name
            .prettify(&arena)
            .ident_color()
            .append(RcDoc::text(":"))
            .append(RcDoc::space())
            .append(RcDoc::text(self.width.to_string()))
    }
}

/* ============== Impls for Structure ================= */

impl PrettyPrint for Cell {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        match self {
            Cell::Decl { data } => data.prettify(&arena),
            Cell::Prim { data } => data.prettify(&arena),
        }
    }
}

impl PrettyPrint for Decl {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        self.name
            .prettify(&arena)
            .ident_color()
            .append(RcDoc::space())
            .append(RcDoc::text("="))
            .append(RcDoc::space())
            .append(self.component.prettify(&arena))
    }
}

impl PrettyPrint for Prim {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        self.name
            .prettify(&arena)
            .ident_color()
            .append(RcDoc::space())
            .append(RcDoc::text("="))
            .append(RcDoc::space())
            .append(RcDoc::text("prim").keyword_color())
            .append(RcDoc::space())
            .append(self.instance.prettify(&arena))
    }
}

impl PrettyPrint for Compinst {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        self.name
            .prettify(&arena)
            .append(self.params.prettify(&arena).parens())
    }
}

/* ============== Impls for Connections ================= */

impl PrettyPrint for Connection {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        match self {
            Connection::Group(g) => g.prettify(&arena),
            Connection::Wire(w) => w.prettify(&arena),
        }
    }
}

impl PrettyPrint for Group {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let title = RcDoc::text("group")
            .keyword_color()
            .append(RcDoc::space())
            .append(self.name.prettify(&arena).ident_color());

        let body = stmt_vec(self.wires.iter().map(|x| x.prettify(&arena)));
        block(title, body)
    }
}

impl PrettyPrint for Wire {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let lhs = self
            .dest
            .prettify(&arena)
            .append(RcDoc::space())
            .append(RcDoc::text("="))
            .append(RcDoc::space());

        let rhs = match self.src.as_slice() {
            [] => unreachable!(),
            [(guard, atom)] if guard.exprs.is_empty() => atom.prettify(&arena),
            rest => {
                let body = rest.iter().map(|(guard, atom)| {
                    guard
                        .prettify(&arena)
                        .append(RcDoc::space())
                        .append(RcDoc::text("=>"))
                        .append(RcDoc::space())
                        .append(atom.prettify(&arena))
                });
                block(RcDoc::text("switch").keyword_color(), stmt_vec(body))
            }
        };
        lhs.append(rhs)
    }
}

impl PrettyPrint for Guard {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        RcDoc::intersperse(
            self.exprs.iter().map(|x| x.prettify(&arena)),
            RcDoc::space()
                .append(RcDoc::text("&"))
                .append(RcDoc::space()),
        )
    }
}

impl PrettyPrint for GuardExpr {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let binop = |e1: &Atom, sym: &'static str, e2: &Atom| {
            e1.prettify(&arena)
                .append(RcDoc::space())
                .append(RcDoc::text(sym))
                .append(RcDoc::space())
                .append(e2.prettify(&arena))
        };
        match self {
            GuardExpr::Eq(e1, e2) => binop(e1, "==", e2),
            GuardExpr::Neq(e1, e2) => binop(e1, "!=", e2),
            GuardExpr::Gt(e1, e2) => binop(e1, ">", e2),
            GuardExpr::Lt(e1, e2) => binop(e1, "<", e2),
            GuardExpr::Geq(e1, e2) => binop(e1, ">=", e2),
            GuardExpr::Leq(e1, e2) => binop(e1, "<=", e2),
            GuardExpr::Atom(e) => e.prettify(&arena),
        }
    }
}

impl PrettyPrint for Atom {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        match self {
            Atom::Port(p) => p.prettify(&arena),
            Atom::Num(n) => n.prettify(&arena),
        }
    }
}

/* ============== Impls for Control ================= */

impl PrettyPrint for Control {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        match self {
            Control::Seq { data } => data.prettify(&arena),
            Control::Par { data } => data.prettify(&arena),
            Control::If { data } => data.prettify(&arena),
            Control::While { data } => data.prettify(&arena),
            Control::Print { data } => data.prettify(&arena),
            Control::Enable { data } => data.prettify(&arena),
            Control::Empty { data } => data.prettify(&arena),
        }
    }
}

impl PrettyPrint for Seq {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        block(
            RcDoc::text("seq").control_color(),
            stmt_vec(self.stmts.iter().map(|x| x.prettify(&arena))),
        )
    }
}

impl PrettyPrint for Par {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        block(
            RcDoc::text("par").control_color(),
            stmt_vec(self.stmts.iter().map(|x| x.prettify(&arena))),
        )
    }
}

impl PrettyPrint for If {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let title = RcDoc::text("if")
            .control_color()
            .append(RcDoc::space())
            .append(self.port.prettify(&arena))
            .append(self.cond.as_ref().map_or(RcDoc::nil(), |x| {
                RcDoc::space()
                    .append(RcDoc::text("with").control_color())
                    .append(RcDoc::space())
                    .append(x.prettify(&arena))
            }));

        let body = self.tbranch.prettify(&arena);
        let tbranch = block(title, body);
        if let Control::Empty { .. } = *self.fbranch {
            tbranch
        } else if let Control::If { .. } = *self.fbranch {
            tbranch
                .append(RcDoc::space())
                .append(RcDoc::text("else"))
                .append(RcDoc::space())
                .append(self.fbranch.prettify(&arena))
        } else {
            tbranch.append(RcDoc::space()).append(block(
                RcDoc::text("else"),
                self.fbranch.prettify(&arena),
            ))
        }
    }
}

impl PrettyPrint for While {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        let title = RcDoc::text("while")
            .control_color()
            .append(RcDoc::space())
            .append(self.port.prettify(&arena))
            .append(self.cond.as_ref().map_or(RcDoc::nil(), |x| {
                RcDoc::space()
                    .append(RcDoc::text("with").control_color())
                    .append(RcDoc::space())
                    .append(x.prettify(&arena))
            }));

        let body = self.body.prettify(&arena);
        block(title, body)
    }
}

impl PrettyPrint for Print {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        RcDoc::text("print")
            .control_color()
            .append(RcDoc::line())
            .append(self.var.prettify(&arena))
            .group()
            .parens()
    }
}

impl PrettyPrint for Enable {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        self.comp.prettify(&arena)
    }
}

impl PrettyPrint for Empty {
    fn prettify<'a>(&self, _arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        RcDoc::text("empty").control_color().parens()
    }
}

impl PrettyPrint for Port {
    fn prettify<'a>(&self, arena: &'a bumpalo::Bump) -> RcDoc<'a, ColorSpec> {
        match self {
            Port::Comp { component, port } => component
                .prettify(&arena)
                .append(RcDoc::text("."))
                .append(port.prettify(&arena)),
            Port::This { port } => port.prettify(&arena),
            Port::Hole { group, name } => group
                .prettify(&arena)
                .append(name.prettify(&arena).brackets()),
        }
    }
}

impl Display for Portdef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(port {} {})", self.name.to_string(), self.width)
    }
}
