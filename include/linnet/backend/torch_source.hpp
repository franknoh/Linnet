#pragma once

#include "linnet/backend/graph_export.hpp"

#include <expected>
#include <string>

namespace linnet::backend {

// Exports one entry of a root block as a Python module of straight-line
// PyTorch code: `main(*inputs, *parameters, *states)` returns the entry's
// results followed by the assigned states. The module's `PARAMETERS`,
// `STATES`, and `NEXT_STATES` lists give the argument and result order by
// parameter path, `RESULTS` the number of the entry's own results. Shapes
// are static, exactly as for `export_stablehlo`; semantic calls whose
// selected candidate is a PyTorch implementation (`opt::select_candidates`)
// become that library call instead of their decomposition.
using TorchSourceOptions = GraphExportOptions;

std::expected<std::string, std::string> export_torch_source(ir::Module& module,
                                                            const TorchSourceOptions& options);

} // namespace linnet::backend
