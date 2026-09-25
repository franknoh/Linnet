#pragma once

#include "linnet/backend/graph_export.hpp"
#include "linnet/ir/ir.hpp"

#include <cstdint>
#include <expected>
#include <map>
#include <string>

namespace linnet::backend {

// Exports one entry of a root block as a StableHLO module in MLIR text.
//
// The module has a single `func.func` whose arguments are the entry's inputs
// followed by every parameter and buffer of the block hierarchy, in manifest
// order, each annotated with `linnet.path` so a runtime can bind weights by
// name. Shapes are static: every generic parameter of the root block and of
// the entry must be bound to a constant (`bindings`), and each optional
// parameter is present or absent as the checkpoint has it (`optionals_present`,
// less the paths in `absent`). Calls are
// inlined, `static for` is unrolled, and index notation becomes broadcasts,
// gathers, and reductions over the output grid.
//
// Everything StableHLO cannot express as captured is a capability failure
// reported in the error string, never a silent approximation.
using StableHloOptions = GraphExportOptions;

std::expected<std::string, std::string> export_stablehlo(ir::Module& module,
                                                         const StableHloOptions& options);

} // namespace linnet::backend
