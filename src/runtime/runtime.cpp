#include "linnet/runtime/runtime.h"

#include "driver.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using linnet::runtime::CUcontext;
using linnet::runtime::cuda_success;
using linnet::runtime::CUdevice;
using linnet::runtime::CUfunction;
using linnet::runtime::CUgraph;
using linnet::runtime::CUgraphExec;
using linnet::runtime::CUmodule;
using linnet::runtime::CUresult;
using linnet::runtime::CUstream;
using linnet::runtime::Driver;

namespace {

thread_local std::string last_error;

LnStatus fail(LnStatus status, std::string message) {
    last_error = std::move(message);
    return status;
}

constexpr std::size_t cpu_alignment = 256;

void* cpu_alloc(std::size_t bytes) {
    // Rounded up: `aligned_alloc` takes only a multiple of the alignment.
    const std::size_t rounded = (bytes + cpu_alignment - 1) / cpu_alignment * cpu_alignment;
#if defined(_WIN32)
    return _aligned_malloc(rounded == 0 ? cpu_alignment : rounded, cpu_alignment);
#else
    return std::aligned_alloc(cpu_alignment, rounded == 0 ? cpu_alignment : rounded);
#endif
}

void cpu_free(void* pointer) {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

} // namespace

struct LnRuntime {
    LnDevice device{};
    const Driver* driver = nullptr; // CUDA only
    CUdevice cuda_device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    std::set<void*> allocations;

    bool cuda() const { return driver != nullptr; }

    // Runs `body` with the device's context current, as the driver needs
    // for every call, and the caller's restored after.
    template <typename Body>
    LnStatus in_context(const char* what, Body body) {
        if (!cuda()) {
            return body();
        }
        if (const CUresult result = driver->cuCtxPushCurrent(context); result != cuda_success) {
            return fail(LN_ERROR, driver->message(what, result));
        }
        const LnStatus status = body();
        CUcontext popped = nullptr;
        driver->cuCtxPopCurrent(&popped);
        return status;
    }

    LnStatus check(const char* what, CUresult result) const {
        return result == cuda_success ? LN_OK : fail(LN_ERROR, driver->message(what, result));
    }

    void* allocate(std::size_t bytes) {
        if (!cuda()) {
            return cpu_alloc(bytes);
        }
        void* pointer = nullptr;
        if (driver->cuMemAlloc(&pointer, bytes == 0 ? 1 : bytes) != cuda_success) {
            return nullptr;
        }
        return pointer;
    }

    void release(void* pointer) {
        if (!cuda()) {
            cpu_free(pointer);
            return;
        }
        driver->cuMemFree(pointer);
    }
};

namespace {

// A tensor whose shape and strides the call keeps.
struct Owned {
    LnTensor tensor{};
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;

    explicit Owned(const LnTensor& from) : tensor(from) {
        const auto rank = static_cast<std::size_t>(from.ndim);
        shape.assign(from.shape, from.shape + rank);
        tensor.shape = shape.data();
        if (from.strides != nullptr) {
            strides.assign(from.strides, from.strides + rank);
            tensor.strides = strides.data();
        }
    }
};

struct Call {
    const LnExecutable* executable = nullptr;
    std::vector<std::unique_ptr<Owned>> inputs;
    std::vector<std::unique_ptr<Owned>> outputs;
    std::vector<LnTensor> input_views;
    std::vector<LnTensor> output_views;
    std::size_t workspace_offset = 0;
    std::size_t workspace_bytes = 0;
};

LnStatus executable_failed(const LnExecutable* executable, const char* what) {
    const char* message =
        executable->error != nullptr ? executable->error(executable->state) : nullptr;
    return fail(LN_ERROR,
                std::string(what) + " failed" + (message ? std::string(": ") + message : ""));
}

} // namespace

struct LnPlan {
    LnRuntime* runtime = nullptr;
    void* arena = nullptr;
    std::size_t arena_bytes = 0;
    std::vector<std::unique_ptr<Call>> calls;
    bool prepared = false;
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;

    LnContext context(const Call& call, bool capturing) const {
        LnContext out{};
        out.abi_version = LN_ABI_VERSION;
        out.device = runtime->device;
        out.stream = runtime->stream;
        out.workspace = static_cast<char*>(arena) + call.workspace_offset;
        out.workspace_bytes = call.workspace_bytes;
        out.capturing = capturing ? 1 : 0;
        return out;
    }

    LnStatus execute(bool capturing) {
        for (const auto& call : calls) {
            LnContext context = this->context(*call, capturing);
            const LnExecutable* executable = call->executable;
            if (executable->execute(executable->state,
                                    call->input_views.data(),
                                    call->input_views.size(),
                                    call->output_views.data(),
                                    call->output_views.size(),
                                    &context) != 0) {
                return executable_failed(executable, "an executable");
            }
        }
        return LN_OK;
    }

    void drop_graph() {
        const Driver* driver = runtime->driver;
        if (exec != nullptr) {
            driver->cuGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (graph != nullptr) {
            driver->cuGraphDestroy(graph);
            graph = nullptr;
        }
    }
};

extern "C" {

const char* ln_last_error(void) {
    return last_error.c_str();
}

LnStatus ln_runtime_create(LnDevice device, LnRuntime** out) {
    if (out == nullptr) {
        return fail(LN_ERROR, "ln_runtime_create: no place for the runtime");
    }
    auto runtime = std::make_unique<LnRuntime>();
    runtime->device = device;
    if (device.type == LN_CPU) {
        *out = runtime.release();
        return LN_OK;
    }
    if (device.type != LN_CUDA) {
        return fail(LN_UNAVAILABLE,
                    "devices of type " + std::to_string(device.type) + " have no runtime yet");
    }
    std::string error;
    const Driver* driver = linnet::runtime::driver(error);
    if (driver == nullptr) {
        return fail(LN_UNAVAILABLE, error);
    }
    runtime->driver = driver;
    if (const auto status =
            runtime->check("cuDeviceGet", driver->cuDeviceGet(&runtime->cuda_device, device.id));
        status != LN_OK) {
        return status;
    }
    // The device's primary context, which PyTorch and JAX use too, so
    // pointers pass between them.
    if (const auto status = runtime->check(
            "cuDevicePrimaryCtxRetain",
            driver->cuDevicePrimaryCtxRetain(&runtime->context, runtime->cuda_device));
        status != LN_OK) {
        return status;
    }
    LnRuntime* raw = runtime.get();
    const LnStatus status = raw->in_context("cuStreamCreate", [&] {
        return raw->check(
            "cuStreamCreate",
            driver->cuStreamCreate(&raw->stream, linnet::runtime::stream_non_blocking));
    });
    if (status != LN_OK) {
        driver->cuDevicePrimaryCtxRelease(runtime->cuda_device);
        return status;
    }
    *out = runtime.release();
    return LN_OK;
}

void ln_runtime_destroy(LnRuntime* runtime) {
    if (runtime == nullptr) {
        return;
    }
    runtime->in_context("destroy", [&] {
        for (void* pointer : runtime->allocations) {
            runtime->release(pointer);
        }
        if (runtime->stream != nullptr) {
            runtime->driver->cuStreamDestroy(runtime->stream);
        }
        return LN_OK;
    });
    if (runtime->cuda()) {
        runtime->driver->cuDevicePrimaryCtxRelease(runtime->cuda_device);
    }
    delete runtime;
}

void* ln_runtime_stream(LnRuntime* runtime) {
    return runtime == nullptr ? nullptr : runtime->stream;
}

LnStatus ln_alloc(LnRuntime* runtime, size_t bytes, void** out) {
    if (runtime == nullptr || out == nullptr) {
        return fail(LN_ERROR, "ln_alloc: no runtime or no place for the pointer");
    }
    return runtime->in_context("ln_alloc", [&] {
        void* pointer = runtime->allocate(bytes);
        if (pointer == nullptr) {
            return fail(LN_ERROR,
                        "ln_alloc: out of memory for " + std::to_string(bytes) + " bytes");
        }
        runtime->allocations.insert(pointer);
        *out = pointer;
        return LN_OK;
    });
}

LnStatus ln_free(LnRuntime* runtime, void* pointer) {
    if (runtime == nullptr || runtime->allocations.erase(pointer) == 0) {
        return fail(LN_ERROR, "ln_free: not memory this runtime allocated");
    }
    return runtime->in_context("ln_free", [&] {
        runtime->release(pointer);
        return LN_OK;
    });
}

LnStatus ln_plan_create(LnRuntime* runtime, size_t arena_bytes, LnPlan** out) {
    if (runtime == nullptr || out == nullptr) {
        return fail(LN_ERROR, "ln_plan_create: no runtime or no place for the plan");
    }
    auto plan = std::make_unique<LnPlan>();
    plan->runtime = runtime;
    plan->arena_bytes = arena_bytes;
    const LnStatus status = runtime->in_context("ln_plan_create", [&] {
        plan->arena = runtime->allocate(arena_bytes);
        return plan->arena == nullptr ? fail(LN_ERROR,
                                             "ln_plan_create: out of memory for an arena of " +
                                                 std::to_string(arena_bytes) + " bytes")
                                      : LN_OK;
    });
    if (status != LN_OK) {
        return status;
    }
    *out = plan.release();
    return LN_OK;
}

void ln_plan_destroy(LnPlan* plan) {
    if (plan == nullptr) {
        return;
    }
    LnRuntime* runtime = plan->runtime;
    runtime->in_context("ln_plan_destroy", [&] {
        if (runtime->cuda()) {
            plan->drop_graph();
        }
        runtime->release(plan->arena);
        return LN_OK;
    });
    delete plan;
}

void* ln_plan_arena(LnPlan* plan) {
    return plan == nullptr ? nullptr : plan->arena;
}

LnStatus ln_plan_add(LnPlan* plan,
                     const LnExecutable* executable,
                     const LnTensor* inputs,
                     size_t input_count,
                     const LnTensor* outputs,
                     size_t output_count,
                     size_t workspace_offset,
                     size_t workspace_bytes) {
    if (plan == nullptr || executable == nullptr) {
        return fail(LN_ERROR, "ln_plan_add: no plan or no executable");
    }
    if (executable->abi_version != LN_ABI_VERSION) {
        return fail(LN_ERROR,
                    "ln_plan_add: an executable of ABI version " +
                        std::to_string(executable->abi_version) + ", not " +
                        std::to_string(LN_ABI_VERSION));
    }
    if (workspace_offset + workspace_bytes > plan->arena_bytes) {
        return fail(LN_ERROR, "ln_plan_add: the workspace lies outside the arena");
    }
    if (plan->exec != nullptr) {
        return fail(LN_ERROR, "ln_plan_add: the plan is captured already");
    }
    auto call = std::make_unique<Call>();
    call->executable = executable;
    call->workspace_offset = workspace_offset;
    call->workspace_bytes = workspace_bytes;
    for (std::size_t i = 0; i < input_count; ++i) {
        call->inputs.push_back(std::make_unique<Owned>(inputs[i]));
        call->input_views.push_back(call->inputs.back()->tensor);
    }
    for (std::size_t i = 0; i < output_count; ++i) {
        call->outputs.push_back(std::make_unique<Owned>(outputs[i]));
        call->output_views.push_back(call->outputs.back()->tensor);
    }
    plan->calls.push_back(std::move(call));
    plan->prepared = false;
    return LN_OK;
}

LnStatus ln_plan_run(LnPlan* plan) {
    if (plan == nullptr) {
        return fail(LN_ERROR, "ln_plan_run: no plan");
    }
    LnRuntime* runtime = plan->runtime;
    return runtime->in_context("ln_plan_run", [&] {
        if (plan->exec != nullptr) {
            return runtime->check("cuGraphLaunch",
                                  runtime->driver->cuGraphLaunch(plan->exec, runtime->stream));
        }
        if (!plan->prepared) {
            for (const auto& call : plan->calls) {
                const LnExecutable* executable = call->executable;
                if (executable->prepare == nullptr) {
                    continue;
                }
                LnContext context = plan->context(*call, false);
                if (executable->prepare(executable->state, &context) != 0) {
                    return executable_failed(executable, "preparing an executable");
                }
            }
            plan->prepared = true;
        }
        return plan->execute(false);
    });
}

LnStatus ln_plan_capture(LnPlan* plan) {
    if (plan == nullptr) {
        return fail(LN_ERROR, "ln_plan_capture: no plan");
    }
    LnRuntime* runtime = plan->runtime;
    if (!runtime->cuda()) {
        return fail(LN_UNAVAILABLE, "ln_plan_capture: only a CUDA plan captures a graph");
    }
    if (!plan->prepared) {
        return fail(LN_ERROR, "ln_plan_capture: run the plan once before capturing it");
    }
    const Driver* driver = runtime->driver;
    return runtime->in_context("ln_plan_capture", [&] {
        plan->drop_graph();
        if (const auto status =
                runtime->check("cuStreamBeginCapture",
                               driver->cuStreamBeginCapture(
                                   runtime->stream, linnet::runtime::capture_mode_thread_local));
            status != LN_OK) {
            return status;
        }
        const LnStatus executed = plan->execute(true);
        const CUresult ended = driver->cuStreamEndCapture(runtime->stream, &plan->graph);
        if (executed != LN_OK) {
            plan->drop_graph();
            return executed;
        }
        if (const auto status = runtime->check("cuStreamEndCapture", ended); status != LN_OK) {
            plan->drop_graph();
            return status;
        }
        const auto status = runtime->check(
            "cuGraphInstantiate", driver->cuGraphInstantiateWithFlags(&plan->exec, plan->graph, 0));
        if (status != LN_OK) {
            plan->drop_graph();
        }
        return status;
    });
}

LnStatus ln_plan_synchronize(LnPlan* plan) {
    if (plan == nullptr) {
        return fail(LN_ERROR, "ln_plan_synchronize: no plan");
    }
    LnRuntime* runtime = plan->runtime;
    if (!runtime->cuda()) {
        return LN_OK;
    }
    return runtime->in_context("ln_plan_synchronize", [&] {
        return runtime->check("cuStreamSynchronize",
                              runtime->driver->cuStreamSynchronize(runtime->stream));
    });
}

} // extern "C"

// -------------------------------------------------------------- callbacks

namespace {

struct Callback {
    LnCallback callback = nullptr;
    void* user = nullptr;
};

int callback_execute(void* state,
                     const LnTensor* inputs,
                     size_t input_count,
                     LnTensor* outputs,
                     size_t output_count,
                     LnContext* context) {
    const auto* callback = static_cast<Callback*>(state);
    return callback->callback(callback->user, inputs, input_count, outputs, output_count, context);
}

const char* callback_error(void* /*state*/) {
    return "the callback returned an error";
}

void callback_destroy(void* state) {
    delete static_cast<Callback*>(state);
}

// ---------------------------------------------------------------- kernels

struct Kernel {
    LnRuntime* runtime = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    std::uint32_t grid[3] = {1, 1, 1};
    std::uint32_t block[3] = {1, 1, 1};
    std::uint32_t shared_bytes = 0;
    std::vector<LnArgument> arguments;
    std::string error;
};

int kernel_execute(void* state,
                   const LnTensor* inputs,
                   size_t input_count,
                   LnTensor* outputs,
                   size_t output_count,
                   LnContext* context) {
    auto* kernel = static_cast<Kernel*>(state);
    std::vector<std::uint64_t> values(kernel->arguments.size());
    std::vector<void*> pointers(kernel->arguments.size());
    for (std::size_t i = 0; i < kernel->arguments.size(); ++i) {
        const LnArgument& argument = kernel->arguments[i];
        if (argument.kind == LN_ARGUMENT_BYTES) {
            values[i] = argument.bytes;
        } else {
            const bool input = argument.kind == LN_ARGUMENT_INPUT;
            if (argument.index >= (input ? input_count : output_count)) {
                kernel->error = "argument " + std::to_string(i) + " names a missing tensor";
                return 1;
            }
            const LnTensor& tensor = input ? inputs[argument.index] : outputs[argument.index];
            values[i] = reinterpret_cast<std::uintptr_t>(tensor.data) + tensor.byte_offset;
        }
        pointers[i] = &values[i];
    }
    const Driver* driver = kernel->runtime->driver;
    const CUresult result = driver->cuLaunchKernel(kernel->function,
                                                   kernel->grid[0],
                                                   kernel->grid[1],
                                                   kernel->grid[2],
                                                   kernel->block[0],
                                                   kernel->block[1],
                                                   kernel->block[2],
                                                   kernel->shared_bytes,
                                                   static_cast<CUstream>(context->stream),
                                                   pointers.data(),
                                                   nullptr);
    if (result != cuda_success) {
        kernel->error = driver->message("cuLaunchKernel", result);
        return 1;
    }
    return 0;
}

const char* kernel_error(void* state) {
    return static_cast<Kernel*>(state)->error.c_str();
}

void kernel_destroy(void* state) {
    auto* kernel = static_cast<Kernel*>(state);
    LnRuntime* runtime = kernel->runtime;
    runtime->in_context("cuModuleUnload", [&] {
        runtime->driver->cuModuleUnload(kernel->module);
        return LN_OK;
    });
    delete kernel;
}

} // namespace

extern "C" {

LnStatus ln_callback_create(LnCallback callback, void* user, LnExecutable** out) {
    if (callback == nullptr || out == nullptr) {
        return fail(LN_ERROR, "ln_callback_create: no callback or no place for the executable");
    }
    auto executable = std::make_unique<LnExecutable>();
    executable->abi_version = LN_ABI_VERSION;
    executable->state = new Callback{callback, user};
    executable->prepare = nullptr;
    executable->execute = callback_execute;
    executable->error = callback_error;
    executable->destroy = callback_destroy;
    *out = executable.release();
    return LN_OK;
}

LnStatus ln_kernel_create(LnRuntime* runtime,
                          const void* image,
                          const char* function,
                          const uint32_t grid[3],
                          const uint32_t block[3],
                          uint32_t shared_bytes,
                          const LnArgument* arguments,
                          size_t argument_count,
                          LnExecutable** out) {
    if (runtime == nullptr || image == nullptr || function == nullptr || out == nullptr) {
        return fail(LN_ERROR, "ln_kernel_create: missing an argument");
    }
    if (!runtime->cuda()) {
        return fail(LN_UNAVAILABLE, "ln_kernel_create: kernels run on CUDA devices only");
    }
    auto kernel = std::make_unique<Kernel>();
    kernel->runtime = runtime;
    std::memcpy(kernel->grid, grid, sizeof(kernel->grid));
    std::memcpy(kernel->block, block, sizeof(kernel->block));
    kernel->shared_bytes = shared_bytes;
    kernel->arguments.assign(arguments, arguments + argument_count);
    const Driver* driver = runtime->driver;
    const LnStatus status = runtime->in_context("ln_kernel_create", [&] {
        if (const auto loaded = runtime->check("cuModuleLoadData",
                                               driver->cuModuleLoadData(&kernel->module, image));
            loaded != LN_OK) {
            return loaded;
        }
        if (const auto found = runtime->check(
                "cuModuleGetFunction",
                driver->cuModuleGetFunction(&kernel->function, kernel->module, function));
            found != LN_OK) {
            driver->cuModuleUnload(kernel->module);
            return found;
        }
        // Past 48 KiB of shared memory a kernel must ask for it.
        if (shared_bytes > 48U * 1024U) {
            const auto set = runtime->check(
                "cuFuncSetAttribute",
                driver->cuFuncSetAttribute(kernel->function,
                                           linnet::runtime::max_dynamic_shared_size_bytes,
                                           static_cast<int>(shared_bytes)));
            if (set != LN_OK) {
                driver->cuModuleUnload(kernel->module);
                return set;
            }
        }
        return LN_OK;
    });
    if (status != LN_OK) {
        return status;
    }
    auto executable = std::make_unique<LnExecutable>();
    executable->abi_version = LN_ABI_VERSION;
    executable->state = kernel.release();
    executable->prepare = nullptr;
    executable->execute = kernel_execute;
    executable->error = kernel_error;
    executable->destroy = kernel_destroy;
    *out = executable.release();
    return LN_OK;
}

void ln_executable_destroy(LnExecutable* executable) {
    if (executable == nullptr) {
        return;
    }
    if (executable->destroy != nullptr) {
        executable->destroy(executable->state);
    }
    delete executable;
}

} // extern "C"
