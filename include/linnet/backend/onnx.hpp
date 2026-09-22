#pragma once

#include "linnet/backend/graph_export.hpp"

#include <expected>
#include <string>

namespace linnet::backend {

// Exports one entry of a root block as an ONNX model in the ONNX text
// format (`onnx.parser.parse_model` reads it). The graph is `main`; its
// inputs are the entry's inputs followed by every parameter and buffer of
// the block hierarchy in manifest order, named `param<N>`, with the model's
// `metadata_props` mapping each `linnet.path.param<N>` to its parameter
// path. Shapes are static, exactly as for `export_stablehlo`.
using OnnxOptions = GraphExportOptions;

std::expected<std::string, std::string> export_onnx(ir::Module& module, const OnnxOptions& options);

} // namespace linnet::backend
