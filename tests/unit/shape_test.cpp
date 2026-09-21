#include "linnet/shape/dim.hpp"
#include "linnet/shape/solver.hpp"

#include "test.hpp"

#include <limits>

using namespace linnet::shape;

namespace {

struct Fixture {
    DimContext context;
    Poly b = context.symbol(context.add_symbol("B", SymbolKind::Dim));
    Poly s = context.symbol(context.add_symbol("S", SymbolKind::Dim));
    Poly h = context.symbol(context.add_symbol("H", SymbolKind::Dim));
    Poly n = context.symbol(context.add_symbol("N", SymbolKind::Dim));
    Poly d = context.symbol(context.add_symbol("D", SymbolKind::Dim));

    std::string str(const Poly& poly) const { return context.to_string(poly); }
};

} // namespace

TEST("poly: canonical form makes equal expressions identical") {
    const Fixture f;
    CHECK(f.b * f.s * f.h == f.h * (f.s * f.b));
    CHECK(f.h + f.h == Poly(2) * f.h);
    CHECK((f.b + f.s) * (f.b - f.s) == f.b * f.b - f.s * f.s);
    CHECK((f.b + Poly(0)) == f.b);
    CHECK((f.b - f.b).is_zero());
    CHECK(!(f.b == f.s));
    CHECK(Poly(3) * Poly(4) + Poly(2) == Poly(14));
}

TEST("poly: overflow is sticky instead of wrapping") {
    Fixture f;
    const Poly big(std::numeric_limits<std::int64_t>::max());
    CHECK(!(big + Poly(1)).is_valid());
    CHECK(!(big * Poly(2)).is_valid());
    CHECK(!((big * Poly(2)) - big * Poly(2)).is_valid());
    CHECK(!f.context.floor_div(f.b, Poly(0)).is_valid());
    CHECK(!f.context.mod(f.b, Poly(0)).is_valid());
}

TEST("poly: floor division and remainder fold when exact") {
    Fixture f;
    DimContext& c = f.context;
    CHECK(c.floor_div(Poly(7), Poly(2)) == Poly(3));
    CHECK(c.floor_div(Poly(-7), Poly(2)) == Poly(-4));
    CHECK(c.mod(Poly(-7), Poly(2)) == Poly(1));
    CHECK(c.floor_div(Poly(4) * f.h, Poly(2)) == Poly(2) * f.h);
    CHECK(c.floor_div(Poly(4) * f.h + Poly(3), Poly(2)) == Poly(2) * f.h + Poly(1));
    CHECK(c.floor_div(f.b * f.h, f.h) == f.b);
    CHECK(c.floor_div(f.h, Poly(1)) == f.h);
    CHECK(c.mod(Poly(6) * f.h + Poly(5), Poly(3)) == Poly(2));
    CHECK(c.mod(f.b * f.h, f.h).is_zero());
    // Not exact: stays symbolic, and is interned.
    CHECK(c.floor_div(f.h, f.n) == c.floor_div(f.h, f.n));
    CHECK(!c.floor_div(f.h, f.n).constant().has_value());
    CHECK(c.min(f.b, f.s) == c.min(f.s, f.b));
    CHECK(c.max(f.b, f.b) == f.b);
    CHECK(c.min(Poly(3), Poly(5)) == Poly(3));
}

TEST("poly: rendering") {
    Fixture f;
    DimContext& c = f.context;
    CHECK_EQ(f.str(Poly(0)), "0");
    CHECK_EQ(f.str(f.b * f.s), "B * S");
    CHECK_EQ(f.str(f.d + Poly(1)), "D + 1");
    CHECK_EQ(f.str(Poly(2) * f.h - f.b - Poly(3)), "-B + 2 * H - 3");
    CHECK_EQ(f.str(c.floor_div(f.h, f.n)), "H / N");
    CHECK_EQ(f.str(c.floor_div(f.d + Poly(1), Poly(2))), "(D + 1) / 2");
    CHECK_EQ(f.str(f.n * c.floor_div(f.h, f.n)), "N * (H / N)");
    CHECK_EQ(f.str(c.mod(f.h, f.n)), "H % N");
    CHECK_EQ(f.str(c.min(f.s, f.b)), "min(B, S)");
}

TEST("poly: substitution rebuilds and refolds atoms") {
    Fixture f;
    DimContext& c = f.context;
    const SymbolId h_id = 2;
    const SymbolId n_id = 3;
    DimContext::Substitution substitution;
    substitution.dims[h_id] = Poly(4096);
    substitution.dims[n_id] = Poly(32);
    CHECK(c.substitute(c.floor_div(f.h, f.n) * f.n, substitution) == Poly(4096));
    substitution.dims[n_id] = f.b;
    CHECK(c.substitute(f.h * f.n + Poly(1), substitution) == Poly(4096) * f.b + Poly(1));
    CHECK(c.mentions(c.floor_div(f.h, f.n), n_id));
    CHECK(!c.mentions(c.floor_div(f.h, f.n), 0));
}

TEST("solver: equality needs no assumptions for polynomial identities") {
    Fixture f;
    Solver solver(f.context);
    CHECK(solver.prove_equal(f.b * (f.s + Poly(1)), f.b * f.s + f.b));
    CHECK(!solver.prove_equal(f.b, f.s));
    CHECK(!solver.prove(Relation::NotEqual, f.b, f.s));
}

TEST("solver: divisibility lets split dimensions multiply back") {
    Fixture f;
    DimContext& c = f.context;
    Solver solver(f.context);
    const Poly split = f.b * f.n * f.s * c.floor_div(f.h, f.n);
    CHECK(!solver.prove_equal(split, f.b * f.s * f.h));

    solver.assume(Relation::Equal, c.mod(f.h, f.n), Poly(0));
    solver.assume(Relation::Greater, f.n, Poly(0));
    CHECK(solver.prove_equal(split, f.b * f.s * f.h));
    CHECK(solver.prove_equal(c.mod(Poly(2) * f.h, f.n), Poly(0)));
    CHECK(solver.divides(f.n, f.b * f.h));
    CHECK(!solver.divides(f.n, f.b));
    CHECK(!solver.divides(f.n, f.h + Poly(1)));
}

TEST("solver: even dimension halves consistently for both strided slices") {
    Fixture f;
    DimContext& c = f.context;
    Solver solver(f.context);
    const Poly half = c.floor_div(f.d, Poly(2));
    const Poly even_slice = c.floor_div(f.d + Poly(1), Poly(2)); // x[0::2]
    CHECK(!solver.prove_equal(even_slice, half));
    solver.assume(Relation::Equal, c.mod(f.d, Poly(2)), Poly(0));
    CHECK(solver.prove_equal(even_slice, half));
    CHECK(solver.prove_equal(half + half, f.d));
    CHECK(solver.prove_equal(c.mod(f.d + Poly(3), Poly(2)), Poly(1)));
}

TEST("solver: equalities substitute symbols") {
    Fixture f;
    Solver solver(f.context);
    solver.assume(Relation::Equal, f.h, f.n * f.d);
    CHECK(solver.prove_equal(f.b * f.h, f.b * f.d * f.n));
    CHECK(solver.prove_equal(f.context.floor_div(f.h, f.n), f.d));

    Solver other(f.context);
    other.assume(Relation::Equal, f.b + f.s, f.h + f.n);
    CHECK(other.prove_equal(f.h + f.n, f.s + f.b));
    // An equality between products is only usable verbatim.
    Solver opaque(f.context);
    opaque.assume(Relation::Equal, f.b * f.s, f.h * f.n);
    CHECK(opaque.prove_equal(f.h * f.n, f.b * f.s));
    CHECK(!opaque.prove_equal(f.b, f.h));
}

TEST("solver: bounds and inequalities") {
    Fixture f;
    DimContext& c = f.context;
    Solver solver(f.context);
    CHECK(solver.prove(Relation::GreaterEqual, f.b, Poly(0)));
    CHECK(solver.prove(Relation::GreaterEqual, f.b * f.s + f.h, Poly(0)));
    CHECK(!solver.prove(Relation::Greater, f.b, Poly(0)));
    CHECK(!solver.prove(Relation::GreaterEqual, f.b - f.s, Poly(0)));

    solver.assume(Relation::Greater, f.n, Poly(0));
    solver.assume(Relation::LessEqual, f.n, Poly(8));
    solver.assume(Relation::GreaterEqual, f.b, f.s + Poly(2));
    CHECK(solver.prove(Relation::Greater, f.n, Poly(0)));
    CHECK(solver.prove(Relation::NotEqual, f.n, Poly(0)));
    CHECK(solver.prove(Relation::Less, f.n, Poly(9)));
    CHECK(!solver.prove(Relation::Less, f.n, Poly(8)));
    CHECK(solver.lower_bound(f.n * Poly(3) + Poly(1)) == std::optional<std::int64_t>(4));
    CHECK(solver.upper_bound(f.n * Poly(3) + Poly(1)) == std::optional<std::int64_t>(25));
    CHECK(solver.prove(Relation::Greater, f.b, f.s));
    CHECK(solver.prove(Relation::GreaterEqual, f.b - f.s, Poly(1)));
    CHECK(!solver.prove(Relation::Greater, f.s, f.b));
    CHECK(solver.prove(Relation::Less, c.mod(f.h, f.n), Poly(8)));
    CHECK(solver.prove_equal(c.max(f.b, f.s), f.b));
    CHECK(solver.prove_equal(c.min(f.b, f.s), f.s));

    Solver fresh(f.context);
    fresh.assume(Relation::NotEqual, f.n, Poly(0));
    CHECK(fresh.prove(Relation::GreaterEqual, f.n, Poly(1)));
}

TEST("solver: unprovable is not the same as false") {
    Fixture f;
    Solver solver(f.context);
    // H / N * N == H is false without divisibility, e.g. H = 5, N = 2.
    CHECK(!solver.prove_equal(f.context.floor_div(f.h, f.n) * f.n, f.h));
    CHECK(!solver.prove(Relation::NotEqual, f.context.floor_div(f.h, f.n) * f.n, f.h));
    const Poly overflow = Poly(std::numeric_limits<std::int64_t>::max()) * Poly(2);
    CHECK(!solver.prove_equal(overflow, overflow));
}
