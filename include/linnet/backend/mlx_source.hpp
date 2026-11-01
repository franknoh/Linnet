#pragma once

#include "linnet/backend/graph_export.hpp"

#include <expected>
#include <memory>
#include <string>

namespace linnet::backend {

// Exports one entry of a root block as a Python module of straight-line
// `mlx.core` code, with the contract of `export_jax_source`: `main(*inputs,
// *parameters, *states)` returns the entry's results followed by the
// assigned states, and `PARAMETERS`, `STATES`, `NEXT_STATES`, and `RESULTS`
// give the order by parameter path. `PARAMETER_TYPES` and `STATE_TYPES` give
// each one's shape and Linnet dtype, and `DYNAMIC` says whether a runtime
// loop reads values back (`mx.compile` cannot trace it).
using MlxSourceOptions = GraphExportOptions;

std::expected<std::string, std::string> export_mlx_source(ir::Module& module,
                                                          const MlxSourceOptions& options);

// The target `export_mlx_source` prints with, for a caller that drives it
// itself (`RecordingTarget`, `replay_graph`).
std::expected<std::unique_ptr<GraphTarget>, std::string>
make_mlx_target(const MlxSourceOptions& options);

} // namespace linnet::backend
