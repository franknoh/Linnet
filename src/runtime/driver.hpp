#pragma once

// The CUDA driver API, loaded when first asked for: the runtime links no
// CUDA library, so it builds and runs its CPU path where there is none.

#include <cstddef>
#include <cstdint>
#include <string>

namespace linnet::runtime {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUgraph = struct CUgraph_st*;
using CUgraphExec = struct CUgraphExec_st*;

static_assert(sizeof(void*) == sizeof(std::uint64_t), "CUDA runs on 64-bit platforms only");

inline constexpr CUresult cuda_success = 0;
inline constexpr unsigned stream_non_blocking = 1;
inline constexpr int capture_mode_thread_local = 1;
inline constexpr int max_dynamic_shared_size_bytes = 8; // CU_FUNC_ATTRIBUTE_...

struct Driver {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuGetErrorString)(CUresult, const char**);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxPushCurrent)(CUcontext);
    CUresult (*cuCtxPopCurrent)(CUcontext*);
    CUresult (*cuStreamCreate)(CUstream*, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    // A device pointer is a 64-bit integer to the driver and a pointer to
    // its callers; on the 64-bit platforms CUDA runs on, the two agree.
    CUresult (*cuMemAlloc)(void**, std::size_t);
    CUresult (*cuMemFree)(void*);
    CUresult (*cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuFuncSetAttribute)(CUfunction, int, int);
    CUresult (*cuLaunchKernel)(CUfunction,
                               unsigned,
                               unsigned,
                               unsigned,
                               unsigned,
                               unsigned,
                               unsigned,
                               unsigned,
                               CUstream,
                               void**,
                               void**);
    CUresult (*cuStreamBeginCapture)(CUstream, int);
    CUresult (*cuStreamEndCapture)(CUstream, CUgraph*);
    CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec*, CUgraph, unsigned long long);
    CUresult (*cuGraphLaunch)(CUgraphExec, CUstream);
    CUresult (*cuGraphExecDestroy)(CUgraphExec);
    CUresult (*cuGraphDestroy)(CUgraph);

    // What a failed call said, as "<what>: <driver's message>".
    std::string message(const char* what, CUresult result) const;
};

// The driver, loaded and initialized once; nullptr where it cannot be,
// with `error` saying why.
const Driver* driver(std::string& error);

} // namespace linnet::runtime
