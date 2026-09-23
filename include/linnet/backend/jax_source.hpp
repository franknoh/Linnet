#pragma once

#include "linnet/backend/graph_export.hpp"

#include <expected>
#include <string>

namespace linnet::backend {

// Exports one entry of a root block as a Python module of straight-line
// `jax.numpy` code, with the same contract as `export_torch_source`:
// `main(*inputs, *parameters, *states)` returns the entry's results followed
// by the assigned states, and `PARAMETERS`, `STATES`, `NEXT_STATES`, and
// `RESULTS` give the order by parameter path. The function is ordinary JAX,
// so `jax.jit`, `jax.grad`, and `jax.vmap` apply to it.
using JaxSourceOptions = GraphExportOptions;

std::expected<std::string, std::string> export_jax_source(ir::Module& module,
                                                          const JaxSourceOptions& options);

} // namespace linnet::backend
