#include "linnet/ir/ir.hpp"

#include <array>
#include <set>
#include <utility>

namespace linnet::ir {

namespace {

struct OpInfo {
    std::string_view spelling;
    int operands;
    int regions;
};

constexpr auto op_table = std::to_array<OpInfo>({
#define LINNET_IR_INFO(name, spelling, operands, regions) {spelling, operands, regions},
    LINNET_IR_OPS(LINNET_IR_INFO)
#undef LINNET_IR_INFO
});

} // namespace

std::string_view op_spelling(OpKind kind) {
    return op_table[static_cast<std::size_t>(kind)].spelling;
}

int op_operand_count(OpKind kind) {
    return op_table[static_cast<std::size_t>(kind)].operands;
}

int op_region_count(OpKind kind) {
    return op_table[static_cast<std::size_t>(kind)].regions;
}

std::string_view compare_spelling(CompareKind kind) {
    switch (kind) {
    case CompareKind::Eq:
        return "eq";
    case CompareKind::Ne:
        return "ne";
    case CompareKind::Lt:
        return "lt";
    case CompareKind::Le:
        return "le";
    case CompareKind::Gt:
        return "gt";
    case CompareKind::Ge:
        return "ge";
    }
    return "?";
}

std::string_view reduce_spelling(ReduceKind kind) {
    switch (kind) {
    case ReduceKind::Sum:
        return "sum";
    case ReduceKind::Prod:
        return "prod";
    case ReduceKind::Max:
        return "max";
    case ReduceKind::Min:
        return "min";
    case ReduceKind::Any:
        return "any";
    case ReduceKind::All:
        return "all";
    }
    return "?";
}

FunctionId Module::add_function(Function function) {
    functions_.push_back(std::move(function));
    return static_cast<FunctionId>(functions_.size() - 1);
}

RegionId Module::add_region(OpId parent) {
    regions_.push_back({{}, parent});
    return static_cast<RegionId>(regions_.size() - 1);
}

BlockId Module::add_block(RegionId region) {
    blocks_.push_back({{}, {}, region});
    const auto id = static_cast<BlockId>(blocks_.size() - 1);
    regions_[region].blocks.push_back(id);
    return id;
}

ValueId Module::add_argument(BlockId block, sema::TypeId type, std::string name) {
    values_.push_back({type, no_id, block, std::move(name)});
    const auto id = static_cast<ValueId>(values_.size() - 1);
    blocks_[block].arguments.push_back(id);
    return id;
}

OpId Module::add_op(BlockId block,
                    OpKind kind,
                    std::vector<ValueId> operands,
                    const std::vector<sema::TypeId>& result_types,
                    Attributes attributes,
                    SourceSpan span) {
    const auto id = static_cast<OpId>(ops_.size());
    Operation operation;
    operation.kind = kind;
    operation.operands = std::move(operands);
    operation.attributes = std::move(attributes);
    operation.span = span;
    operation.block = block;
    for (const sema::TypeId type : result_types) {
        values_.push_back({type, id, block, {}});
        operation.results.push_back(static_cast<ValueId>(values_.size() - 1));
    }
    ops_.push_back(std::move(operation));
    blocks_[block].ops.push_back(id);
    return id;
}

// -------------------------------------------------------------------- verifier

namespace {

class Verifier {
public:
    explicit Verifier(const Module& module) : module_(module) {}

    std::vector<std::string> run() {
        for (std::size_t i = 0; i < module_.functions().size(); ++i) {
            const Function& function = module_.functions()[i];
            const std::set<ValueId> visible;
            check_region(function.body, visible, "function " + function.name, &function);
        }
        return std::move(problems_);
    }

private:
    void check_region(RegionId id,
                      std::set<ValueId> visible,
                      const std::string& where,
                      const Function* function) {
        const Region& region = module_.region(id);
        if (region.blocks.size() != 1) {
            problems_.push_back(where + ": region must have exactly one block");
            return;
        }
        const Block& block = module_.block(region.blocks.front());
        for (const ValueId argument : block.arguments) {
            visible.insert(argument);
        }
        if (block.ops.empty()) {
            problems_.push_back(where + ": block has no terminator");
            return;
        }
        for (std::size_t i = 0; i < block.ops.size(); ++i) {
            const Operation& op = module_.op(block.ops[i]);
            const std::string at = where + ": " + std::string(op_spelling(op.kind));
            const bool is_terminator = op.kind == OpKind::Return || op.kind == OpKind::Yield;
            if (is_terminator != (i + 1 == block.ops.size())) {
                problems_.push_back(at + ": terminator must be the last operation");
            }
            if (op.kind == OpKind::Return && function == nullptr) {
                problems_.push_back(at + ": return inside a nested region");
            }
            if (op.kind == OpKind::Yield && function != nullptr) {
                problems_.push_back(at + ": yield at function level");
            }
            const int operands = op_operand_count(op.kind);
            if (operands >= 0 && op.operands.size() != static_cast<std::size_t>(operands)) {
                problems_.push_back(at + ": expected " + std::to_string(operands) + " operands");
            }
            const int regions = op_region_count(op.kind);
            if (regions >= 0 && op.regions.size() != static_cast<std::size_t>(regions)) {
                problems_.push_back(at + ": expected " + std::to_string(regions) + " regions");
            }
            for (const ValueId operand : op.operands) {
                if (!visible.contains(operand)) {
                    problems_.push_back(at + ": operand %" + std::to_string(operand) +
                                        " is not defined before its use");
                }
            }
            if (op.kind == OpKind::Return && function != nullptr &&
                op.operands.size() != function->results.size()) {
                problems_.push_back(at + ": returns " + std::to_string(op.operands.size()) +
                                    " values, function declares " +
                                    std::to_string(function->results.size()));
            }
            for (const RegionId nested : op.regions) {
                check_region(nested, visible, at, nullptr);
            }
            for (const ValueId result : op.results) {
                visible.insert(result);
            }
        }
    }

    const Module& module_;
    std::vector<std::string> problems_;
};

} // namespace

std::vector<std::string> verify(const Module& module) {
    return Verifier(module).run();
}

// --------------------------------------------------------------------- printer

namespace {

class Printer {
public:
    explicit Printer(const Module& module) : module_(module) {}

    std::string run() {
        for (const Function& function : module_.functions()) {
            print_function(function);
            out_ += '\n';
        }
        return std::move(out_);
    }

private:
    std::string type(sema::TypeId id) const {
        return id == sema::no_type ? "?" : module_.types().to_string(id);
    }

    std::string value(ValueId id) const {
        const Value& v = module_.value(id);
        return "%" + (v.name.empty() ? std::to_string(id) : v.name + std::to_string(id));
    }

    void line(const std::string& text) {
        out_.append(indent_ * 4, ' ');
        out_ += text;
        out_ += '\n';
    }

    void print_function(const Function& function) {
        std::string header = std::string(function.is_op      ? "op "
                                         : function.is_entry ? "entry "
                                                             : "fn ") +
                             "@" + function.name;
        if (!function.generics.empty()) {
            header += "<";
            for (std::size_t i = 0; i < function.generics.size(); ++i) {
                const sema::GenericInfo& generic = function.generics[i];
                header += i == 0 ? "" : ", ";
                header += generic.kind == sema::GenericKind::Pack ? "*" : "";
                header += generic.name;
            }
            header += ">";
        }
        const Block& body = module_.block(module_.region(function.body).blocks.front());
        header += "(";
        for (std::size_t i = 0; i < body.arguments.size(); ++i) {
            header += i == 0 ? "" : ", ";
            header += value(body.arguments[i]) + ": " + type(module_.value(body.arguments[i]).type);
        }
        header += ")";
        for (std::size_t i = 0; i < function.results.size(); ++i) {
            header += i == 0 ? " -> " : ", ";
            header += type(function.results[i]);
        }
        for (const sema::ConstraintInfo& constraint : function.constraints) {
            header += " where " + module_.model().dims.to_string(constraint.lhs) + " " +
                      std::string(constraint.relation == shape::Relation::Equal       ? "=="
                                  : constraint.relation == shape::Relation::NotEqual  ? "!="
                                  : constraint.relation == shape::Relation::Less      ? "<"
                                  : constraint.relation == shape::Relation::LessEqual ? "<="
                                  : constraint.relation == shape::Relation::Greater   ? ">"
                                                                                      : ">=") +
                      " " + module_.model().dims.to_string(constraint.rhs);
        }
        line(header + " {");
        ++indent_;
        print_block(body);
        --indent_;
        line("}");
    }

    void print_block(const Block& block) {
        for (const OpId id : block.ops) {
            print_op(module_.op(id));
        }
    }

    std::string attributes(const Operation& op) const {
        const Attributes& a = op.attributes;
        const sema::TypeStore& types = module_.types();
        std::string text;
        switch (op.kind) {
        case OpKind::ConstInt:
            return " " + std::to_string(a.integer);
        case OpKind::ConstBool:
            return a.integer != 0 ? " true" : " false";
        case OpKind::ConstFloat: {
            const std::string number = std::to_string(a.number);
            return " " + number;
        }
        case OpKind::ConstDim:
            return " " + module_.model().dims.to_string(a.dim);
        case OpKind::EnumConst:
        case OpKind::BlockParam:
        case OpKind::BlockSub:
            return " \"" + a.name + "\"";
        case OpKind::Compare:
            return " " + std::string(compare_spelling(a.compare));
        case OpKind::TupleGet:
        case OpKind::StructGet:
            return " " + std::to_string(a.integer);
        case OpKind::Concat:
            return " axis " + std::to_string(a.axis);
        case OpKind::Reshape:
        case OpKind::Broadcast:
        case OpKind::Fill:
        case OpKind::Permute:
        case OpKind::Iota:
            return " [" + types.to_string(a.shape) + "]";
        case OpKind::Slice:
            for (std::size_t i = 0; i < a.starts.size(); ++i) {
                text += i == 0 ? " [" : ", ";
                if (a.whole[i]) {
                    text += ":";
                } else if (a.squeezed[i]) {
                    text += module_.model().dims.to_string(a.starts[i]);
                } else {
                    text += module_.model().dims.to_string(a.starts[i]) + ":" +
                            module_.model().dims.to_string(a.stops[i]) + ":" +
                            std::to_string(a.steps[i]);
                }
            }
            return text + "]";
        case OpKind::Comprehension:
        case OpKind::Reduce: {
            text = op.kind == OpKind::Reduce ? " " + std::string(reduce_spelling(a.reduce)) : "";
            text += " [";
            for (std::size_t i = 0; i < a.names.size(); ++i) {
                text += i == 0 ? "" : ", ";
                text += a.names[i] + ": " +
                        (i < a.shape.size() ? types.to_string(sema::Shape{a.shape[i]}) : "?");
            }
            return text + "]";
        }
        case OpKind::Call:
        case OpKind::SemanticCall: {
            text = " @" + a.name;
            std::string bindings;
            for (const auto& [symbol, dim] : a.substitution.dims) {
                bindings += bindings.empty() ? "" : ", ";
                bindings += std::string(module_.model().dims.symbol_name(symbol)) + " = " +
                            module_.model().dims.to_string(dim);
            }
            for (const auto& [symbol, shape] : a.substitution.packs) {
                bindings += bindings.empty() ? "" : ", ";
                bindings += "*" + std::string(module_.model().dims.symbol_name(symbol)) + " = [" +
                            types.to_string(shape) + "]";
            }
            for (const auto& [var, dtype] : a.substitution.dtypes) {
                bindings += bindings.empty() ? "" : ", ";
                bindings += types.dtype_var(var).name + " = " + types.to_string(dtype);
            }
            return bindings.empty() ? text : text + "<" + bindings + ">";
        }
        case OpKind::EnumMatch:
            for (const std::string& variant : a.names) {
                text += " " + variant;
            }
            return text;
        default:
            return "";
        }
    }

    void print_op(const Operation& op) {
        std::string text;
        for (std::size_t i = 0; i < op.results.size(); ++i) {
            text += i == 0 ? "" : ", ";
            text += value(op.results[i]);
        }
        text += op.results.empty() ? "" : " = ";
        text += op_spelling(op.kind);
        text += attributes(op);
        for (std::size_t i = 0; i < op.operands.size(); ++i) {
            text += i == 0 ? " " : ", ";
            text += value(op.operands[i]);
        }
        for (std::size_t i = 0; i < op.results.size(); ++i) {
            text += i == 0 ? " : " : ", ";
            text += type(module_.value(op.results[i]).type);
        }
        if (op.regions.empty()) {
            line(text);
            return;
        }
        line(text + " {");
        for (std::size_t i = 0; i < op.regions.size(); ++i) {
            const Block& block = module_.block(module_.region(op.regions[i]).blocks.front());
            if (i != 0) {
                line("} {");
            }
            if (!block.arguments.empty()) {
                std::string args = "^(";
                for (std::size_t j = 0; j < block.arguments.size(); ++j) {
                    args += j == 0 ? "" : ", ";
                    args += value(block.arguments[j]) + ": " +
                            type(module_.value(block.arguments[j]).type);
                }
                line(args + "):");
            }
            ++indent_;
            print_block(block);
            --indent_;
        }
        line("}");
    }

    const Module& module_;
    std::string out_;
    std::size_t indent_ = 0;
};

} // namespace

std::string print(const Module& module) {
    return Printer(module).run();
}

} // namespace linnet::ir
