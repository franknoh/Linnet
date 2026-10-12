/* Linnet's runtime: runs an execution plan's calls on one device
 * (docs/design/execution.md).
 *
 * A plan is a list of calls of executables, each over tensors placed in the
 * plan's arena or in persistent buffers. The runtime owns the memory and the
 * stream, runs the calls in order, and on CUDA can capture them into one
 * graph and replay it. It links no CUDA library: it loads the driver when a
 * CUDA runtime is created, so it builds and runs its CPU path everywhere.
 *
 * Every function returns LN_OK or an error whose message `ln_last_error`
 * gives, for the calling thread. */

#ifndef LINNET_RUNTIME_H
#define LINNET_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The library exports its functions; LINNET_RUNTIME_STATIC builds them
 * into another target instead. */
#if defined(LINNET_RUNTIME_STATIC)
#define LN_API
#elif defined(_WIN32) && defined(LINNET_RUNTIME_BUILD)
#define LN_API __declspec(dllexport)
#elif defined(_WIN32)
#define LN_API __declspec(dllimport)
#else
#define LN_API __attribute__((visibility("default")))
#endif

#define LN_ABI_VERSION 1

typedef int32_t LnStatus;
#define LN_OK 0
#define LN_ERROR 1
#define LN_UNAVAILABLE 2 /* not on this device or platform */

/* Tensors, laid out as DLPack's DLDevice, DLDataType and DLTensor, so that
 * PyTorch, JAX and CuPy share them without copies. */
/* Device types, as DLPack numbers them. */
#define LN_CPU 1
#define LN_CUDA 2

typedef struct LnDevice {
    int32_t type; /* LN_CPU, LN_CUDA */
    int32_t id;
} LnDevice;

typedef struct LnDataType {
    uint8_t code; /* DLPack's: 0 int, 1 uint, 2 float, 4 bfloat, 6 bool */
    uint8_t bits;
    uint16_t lanes;
} LnDataType;

typedef struct LnTensor {
    void* data;
    LnDevice device;
    int32_t ndim;
    LnDataType dtype;
    int64_t* shape;
    int64_t* strides; /* in elements; NULL for a contiguous tensor */
    uint64_t byte_offset;
} LnTensor;

/* What one call runs with. */
typedef struct LnContext {
    uint32_t abi_version;
    LnDevice device;
    void* stream; /* a CUstream, or NULL on the CPU */
    void* workspace;
    size_t workspace_bytes;
    int32_t capturing; /* nonzero while a CUDA graph records the stream */
} LnContext;

/* A compiled region. `execute` enqueues its work on the context's stream
 * and returns without waiting for it; `prepare` runs once before the first
 * call. Each returns 0 on success, otherwise a code `error` explains. */
typedef struct LnExecutable {
    uint32_t abi_version;
    void* state;
    int (*prepare)(void* state, LnContext* context);
    int (*execute)(void* state,
                   const LnTensor* inputs,
                   size_t input_count,
                   LnTensor* outputs,
                   size_t output_count,
                   LnContext* context);
    const char* (*error)(void* state);
    void (*destroy)(void* state);
} LnExecutable;

typedef struct LnRuntime LnRuntime;
typedef struct LnPlan LnPlan;

LN_API const char* ln_last_error(void);

/* A runtime for `device`, with a stream of its own (none on the CPU). */
LN_API LnStatus ln_runtime_create(LnDevice device, LnRuntime** out);
LN_API void ln_runtime_destroy(LnRuntime* runtime);
LN_API void* ln_runtime_stream(LnRuntime* runtime);

/* Persistent memory: weights, states, and what `prepare` keeps. */
LN_API LnStatus ln_alloc(LnRuntime* runtime, size_t bytes, void** out);
LN_API LnStatus ln_free(LnRuntime* runtime, void* pointer);

/* A plan whose temporary tensors live in one arena of `arena_bytes`. */
LN_API LnStatus ln_plan_create(LnRuntime* runtime, size_t arena_bytes, LnPlan** out);
LN_API void ln_plan_destroy(LnPlan* plan);
LN_API void* ln_plan_arena(LnPlan* plan);

/* Appends a call of `executable`, which the plan does not own, over
 * `inputs` and `outputs`; their shapes and strides are copied. The call's
 * workspace is the arena's bytes at `workspace_offset`. */
LN_API LnStatus ln_plan_add(LnPlan* plan,
                            const LnExecutable* executable,
                            const LnTensor* inputs,
                            size_t input_count,
                            const LnTensor* outputs,
                            size_t output_count,
                            size_t workspace_offset,
                            size_t workspace_bytes);

/* Runs every call in order: `prepare` for each executable on the first run,
 * then `execute`; a captured plan replays its graph instead. Returns once
 * the work is enqueued. */
LN_API LnStatus ln_plan_run(LnPlan* plan);

/* Records the calls into a CUDA graph, which later runs replay. Every
 * executable must have been prepared (a run first) and be capturable. */
LN_API LnStatus ln_plan_capture(LnPlan* plan);

/* Waits for the plan's stream. */
LN_API LnStatus ln_plan_synchronize(LnPlan* plan);

/* An executable that calls back into its creator: `execute` with `user`
 * in place of the state. `ln_executable_destroy` frees it. */
typedef int (*LnCallback)(void* user,
                          const LnTensor* inputs,
                          size_t input_count,
                          LnTensor* outputs,
                          size_t output_count,
                          LnContext* context);
LN_API LnStatus ln_callback_create(LnCallback callback, void* user, LnExecutable** out);

/* A CUDA kernel from a cubin or PTX image, launched natively: each argument
 * is an input's or an output's data pointer, or bytes passed as they are. */
#define LN_ARGUMENT_INPUT 0
#define LN_ARGUMENT_OUTPUT 1
#define LN_ARGUMENT_BYTES 2

typedef struct LnArgument {
    int32_t kind;   /* LN_ARGUMENT_INPUT, _OUTPUT or _BYTES */
    uint32_t index; /* the input or output it passes */
    uint64_t bytes; /* LN_ARGUMENT_BYTES: up to 8 bytes, little end first */
    uint32_t size;  /* their count */
} LnArgument;

LN_API LnStatus ln_kernel_create(LnRuntime* runtime,
                                 const void* image,
                                 const char* function,
                                 const uint32_t grid[3],
                                 const uint32_t block[3],
                                 uint32_t shared_bytes,
                                 const LnArgument* arguments,
                                 size_t argument_count,
                                 LnExecutable** out);

/* A linear layer's product by cuBLASLt: outputs[0] ([rows, n]) is
 * inputs[0] ([rows, k]) times inputs[1] ([n, k], a weight) transposed, all
 * of `dtype` (f16, bf16 or f32, row-major), summed in f32. The library is
 * loaded when first asked for. */
LN_API LnStatus ln_cublaslt_linear_create(
    LnRuntime* runtime, int64_t rows, int64_t n, int64_t k, LnDataType dtype, LnExecutable** out);

/* Runs one executable outside a plan, on the context's stream: its
 * `prepare` once (`ln_executable_prepare`), then each `execute`. */
LN_API LnStatus ln_executable_prepare(LnExecutable* executable, LnContext* context);
LN_API LnStatus ln_executable_execute(LnExecutable* executable,
                                      const LnTensor* inputs,
                                      size_t input_count,
                                      LnTensor* outputs,
                                      size_t output_count,
                                      LnContext* context);

/* Destroys an executable the runtime made. */
LN_API void ln_executable_destroy(LnExecutable* executable);

#ifdef __cplusplus
}
#endif

#endif /* LINNET_RUNTIME_H */
