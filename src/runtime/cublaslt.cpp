// A linear layer's product by cuBLASLt, as a native executable. The library
// is loaded when first asked for, from the process when PyTorch or JAX has
// loaded it already, else from the library path.

#include "internal.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using Status = int; // cublasStatus_t
using Handle = struct cublasLtContext*;
using Desc = struct cublasLtMatmulDescOpaque_t*;
using Layout = struct cublasLtMatrixLayoutOpaque_t*;
using Preference = struct cublasLtMatmulPreferenceOpaque_t*;

struct Algo {
    std::uint64_t data[8];
};

struct Heuristic {
    Algo algo;
    std::size_t workspace_bytes;
    Status state;
    float waves;
    int reserved[4];
};

// cublasLt.h's and library_types.h's numbers.
constexpr int compute_32f = 68;
constexpr int r_32f = 0;
constexpr int r_16f = 2;
constexpr int r_16bf = 14;
constexpr int desc_transa = 3;
constexpr int desc_transb = 4;
constexpr std::int32_t op_n = 0;
constexpr std::int32_t op_t = 1;
constexpr int preference_max_workspace = 1;
constexpr int preference_alignment_a = 5; // then B, C and D
constexpr std::size_t workspace_bytes = 32U << 20U;

struct Library {
    Status (*create)(Handle*);
    Status (*destroy)(Handle);
    Status (*desc_create)(Desc*, int, int);
    Status (*desc_destroy)(Desc);
    Status (*desc_set)(Desc, int, const void*, std::size_t);
    Status (*layout_create)(Layout*, int, std::uint64_t, std::uint64_t, std::int64_t);
    Status (*layout_destroy)(Layout);
    Status (*preference_create)(Preference*);
    Status (*preference_destroy)(Preference);
    Status (*preference_set)(Preference, int, const void*, std::size_t);
    Status (*heuristic)(
        Handle, Desc, Layout, Layout, Layout, Layout, Preference, int, Heuristic*, int*);
    Status (*matmul)(Handle,
                     Desc,
                     const void*,
                     const void*,
                     Layout,
                     const void*,
                     Layout,
                     const void*,
                     const void*,
                     Layout,
                     void*,
                     Layout,
                     const Algo*,
                     void*,
                     std::size_t,
                     linnet::runtime::CUstream);
};

void* open_library() {
#if defined(_WIN32)
    for (const char* name : {"cublasLt64_13.dll", "cublasLt64_12.dll"}) {
        if (HMODULE found = LoadLibraryA(name)) {
            return reinterpret_cast<void*>(found);
        }
    }
    return nullptr;
#else
    for (const char* name : {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}) {
        if (void* found = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
            return found;
        }
    }
    return nullptr;
#endif
}

void* symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

const Library* library(std::string& error) {
    static std::once_flag once;
    static std::optional<Library> loaded;
    static std::string failure;
    std::call_once(once, [] {
        void* handle = open_library();
        if (handle == nullptr) {
            failure = "no cuBLASLt library to load";
            return;
        }
        Library lib{};
        std::string missing;
        const auto bind = [&](auto& field, const char* name) {
            field =
                reinterpret_cast<std::remove_reference_t<decltype(field)>>(symbol(handle, name));
            if (field == nullptr) {
                missing += missing.empty() ? name : std::string(", ") + name;
            }
        };
        bind(lib.create, "cublasLtCreate");
        bind(lib.destroy, "cublasLtDestroy");
        bind(lib.desc_create, "cublasLtMatmulDescCreate");
        bind(lib.desc_destroy, "cublasLtMatmulDescDestroy");
        bind(lib.desc_set, "cublasLtMatmulDescSetAttribute");
        bind(lib.layout_create, "cublasLtMatrixLayoutCreate");
        bind(lib.layout_destroy, "cublasLtMatrixLayoutDestroy");
        bind(lib.preference_create, "cublasLtMatmulPreferenceCreate");
        bind(lib.preference_destroy, "cublasLtMatmulPreferenceDestroy");
        bind(lib.preference_set, "cublasLtMatmulPreferenceSetAttribute");
        bind(lib.heuristic, "cublasLtMatmulAlgoGetHeuristic");
        bind(lib.matmul, "cublasLtMatmul");
        if (!missing.empty()) {
            failure = "cuBLASLt lacks " + missing;
            return;
        }
        loaded = lib;
    });
    if (!loaded) {
        error = failure;
        return nullptr;
    }
    return &*loaded;
}

struct Linear {
    const Library* lib = nullptr;
    LnRuntime* runtime = nullptr;
    Handle handle = nullptr;
    Desc desc = nullptr;
    std::array<Layout, 3> layouts{}; // the weight's, the input's, the output's
    Algo algo{};
    std::string error;

    ~Linear() {
        for (Layout layout : layouts) {
            if (layout != nullptr) {
                lib->layout_destroy(layout);
            }
        }
        if (desc != nullptr) {
            lib->desc_destroy(desc);
        }
    }
};

int linear_execute(void* state,
                   const LnTensor* inputs,
                   size_t input_count,
                   LnTensor* outputs,
                   size_t output_count,
                   LnContext* context) {
    auto* linear = static_cast<Linear*>(state);
    if (input_count != 2 || output_count != 1) {
        linear->error = "a linear product takes an input and a weight, and makes one output";
        return 1;
    }
    const auto at = [](const LnTensor& tensor) {
        return static_cast<char*>(tensor.data) + tensor.byte_offset;
    };
    const float alpha = 1.0F;
    const float beta = 0.0F;
    // Column-major, as cuBLAS sees it: the output's transpose is the weight
    // (its transpose stored) times the input's transpose.
    const Status status =
        linear->lib->matmul(linear->handle,
                            linear->desc,
                            &alpha,
                            at(inputs[1]),
                            linear->layouts[0],
                            at(inputs[0]),
                            linear->layouts[1],
                            &beta,
                            at(outputs[0]),
                            linear->layouts[2],
                            at(outputs[0]),
                            linear->layouts[2],
                            &linear->algo,
                            linear->runtime->workspace,
                            linear->runtime->workspace_bytes,
                            static_cast<linnet::runtime::CUstream>(context->stream));
    if (status != 0) {
        linear->error = "cublasLtMatmul failed with status " + std::to_string(status);
        return 1;
    }
    return 0;
}

const char* linear_error(void* state) {
    return static_cast<Linear*>(state)->error.c_str();
}

void linear_destroy(void* state) {
    delete static_cast<Linear*>(state);
}

std::optional<int> data_type(LnDataType dtype) {
    if (dtype.code == 2 && dtype.bits == 32) {
        return r_32f;
    }
    if (dtype.code == 2 && dtype.bits == 16) {
        return r_16f;
    }
    if (dtype.code == 4 && dtype.bits == 16) {
        return r_16bf;
    }
    return std::nullopt;
}

// The runtime's cuBLASLt handle and workspace, made when first asked for.
LnStatus prepare_runtime(LnRuntime* runtime, const Library* lib, Handle& handle) {
    static std::mutex mutex;
    const std::scoped_lock lock(mutex);
    if (runtime->workspace == nullptr) {
        runtime->workspace = runtime->allocate(workspace_bytes);
        if (runtime->workspace == nullptr) {
            return linnet::runtime::fail(LN_ERROR, "no memory for cuBLASLt's workspace");
        }
        runtime->allocations.insert(runtime->workspace);
        runtime->workspace_bytes = workspace_bytes;
    }
    static std::map<LnRuntime*, Handle> handles;
    const auto found = handles.find(runtime);
    if (found != handles.end()) {
        handle = found->second;
        return LN_OK;
    }
    if (const Status status = lib->create(&handle); status != 0) {
        return linnet::runtime::fail(LN_ERROR,
                                     "cublasLtCreate failed with status " + std::to_string(status));
    }
    handles.emplace(runtime, handle);
    runtime->cleanups.emplace_back([lib, runtime, made = handle] {
        lib->destroy(made);
        const std::scoped_lock relock(mutex);
        handles.erase(runtime);
    });
    return LN_OK;
}

} // namespace

extern "C" {

LnStatus ln_cublaslt_linear_create(
    LnRuntime* runtime, int64_t rows, int64_t n, int64_t k, LnDataType dtype, LnExecutable** out) {
    using linnet::runtime::fail;
    if (runtime == nullptr || out == nullptr || rows <= 0 || n <= 0 || k <= 0) {
        return fail(LN_ERROR, "ln_cublaslt_linear_create: a missing or empty argument");
    }
    if (!runtime->cuda()) {
        return fail(LN_UNAVAILABLE, "ln_cublaslt_linear_create: cuBLASLt runs on CUDA devices");
    }
    const auto type = data_type(dtype);
    if (!type) {
        return fail(LN_UNAVAILABLE, "ln_cublaslt_linear_create: f16, bf16 or f32 only");
    }
    std::string error;
    const Library* lib = library(error);
    if (lib == nullptr) {
        return fail(LN_UNAVAILABLE, error);
    }
    auto linear = std::make_unique<Linear>();
    linear->lib = lib;
    linear->runtime = runtime;
    const LnStatus status = runtime->in_context("ln_cublaslt_linear_create", [&] {
        if (const LnStatus prepared = prepare_runtime(runtime, lib, linear->handle);
            prepared != LN_OK) {
            return prepared;
        }
        const auto check = [](Status code, const char* what) {
            return code == 0
                       ? LN_OK
                       : fail(LN_ERROR,
                              std::string(what) + " failed with status " + std::to_string(code));
        };
        if (const auto s = check(lib->desc_create(&linear->desc, compute_32f, r_32f),
                                 "cublasLtMatmulDescCreate");
            s != LN_OK) {
            return s;
        }
        if (const auto s = check(lib->desc_set(linear->desc, desc_transa, &op_t, sizeof(op_t)),
                                 "setting TRANSA");
            s != LN_OK) {
            return s;
        }
        if (const auto s = check(lib->desc_set(linear->desc, desc_transb, &op_n, sizeof(op_n)),
                                 "setting TRANSB");
            s != LN_OK) {
            return s;
        }
        const auto un = [](std::int64_t value) { return static_cast<std::uint64_t>(value); };
        // The weight as stored ([k, n] column-major), the input ([k, rows]),
        // and the output ([n, rows]).
        if (const auto s = check(lib->layout_create(&linear->layouts[0], *type, un(k), un(n), k),
                                 "the weight's layout");
            s != LN_OK) {
            return s;
        }
        if (const auto s = check(lib->layout_create(&linear->layouts[1], *type, un(k), un(rows), k),
                                 "the input's layout");
            s != LN_OK) {
            return s;
        }
        if (const auto s = check(lib->layout_create(&linear->layouts[2], *type, un(n), un(rows), n),
                                 "the output's layout");
            s != LN_OK) {
            return s;
        }
        Preference preference = nullptr;
        if (const auto s = check(lib->preference_create(&preference), "a preference"); s != LN_OK) {
            return s;
        }
        const std::uint64_t limit = workspace_bytes;
        lib->preference_set(preference, preference_max_workspace, &limit, sizeof(limit));
        // Tensors may start at any 16 bytes: a slice of a larger product.
        const std::uint32_t alignment = 16;
        for (int which = 0; which < 4; ++which) {
            lib->preference_set(
                preference, preference_alignment_a + which, &alignment, sizeof(alignment));
        }
        Heuristic found{};
        int count = 0;
        const Status searched = lib->heuristic(linear->handle,
                                               linear->desc,
                                               linear->layouts[0],
                                               linear->layouts[1],
                                               linear->layouts[2],
                                               linear->layouts[2],
                                               preference,
                                               1,
                                               &found,
                                               &count);
        lib->preference_destroy(preference);
        if (searched != 0 || count == 0) {
            return fail(LN_UNAVAILABLE,
                        "cuBLASLt has no algorithm for this product (status " +
                            std::to_string(searched) + ")");
        }
        linear->algo = found.algo;
        return LN_OK;
    });
    if (status != LN_OK) {
        return status;
    }
    auto executable = std::make_unique<LnExecutable>();
    executable->abi_version = LN_ABI_VERSION;
    executable->state = linear.release();
    executable->prepare = nullptr;
    executable->execute = linear_execute;
    executable->error = linear_error;
    executable->destroy = linear_destroy;
    *out = executable.release();
    return LN_OK;
}

} // extern "C"
