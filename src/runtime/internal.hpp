#pragma once

// What the runtime's sources share: the runtime itself, and how a failure
// is reported (`ln_last_error`).

#include "linnet/runtime/runtime.h"

#include "driver.hpp"

#include <cstdlib>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace linnet::runtime {

// Records `message` for `ln_last_error` and returns `status`.
LnStatus fail(LnStatus status, std::string message);

// An executable's failure, with what it says about it.
LnStatus executable_failed(const LnExecutable* executable, const char* what);

void* cpu_alloc(std::size_t bytes);
void cpu_free(void* pointer);

} // namespace linnet::runtime

struct LnRuntime {
    LnDevice device{};
    const linnet::runtime::Driver* driver = nullptr; // CUDA only
    linnet::runtime::CUdevice cuda_device = 0;
    linnet::runtime::CUcontext context = nullptr;
    linnet::runtime::CUstream stream = nullptr;
    std::set<void*> allocations;
    // Run when the runtime is destroyed, before its memory goes: libraries'
    // handles (cuBLASLt's).
    std::vector<std::function<void()>> cleanups;
    // A library's scratch memory, made when first asked for.
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;

    bool cuda() const { return driver != nullptr; }

    // Runs `body` with the device's context current, as the driver needs
    // for every call, and the caller's restored after.
    template <typename Body>
    LnStatus in_context(const char* what, Body body) {
        if (!cuda()) {
            return body();
        }
        if (const linnet::runtime::CUresult result = driver->cuCtxPushCurrent(context);
            result != linnet::runtime::cuda_success) {
            return linnet::runtime::fail(LN_ERROR, driver->message(what, result));
        }
        const LnStatus status = body();
        linnet::runtime::CUcontext popped = nullptr;
        driver->cuCtxPopCurrent(&popped);
        return status;
    }

    LnStatus check(const char* what, linnet::runtime::CUresult result) const {
        return result == linnet::runtime::cuda_success
                   ? LN_OK
                   : linnet::runtime::fail(LN_ERROR, driver->message(what, result));
    }

    void* allocate(std::size_t bytes) {
        if (!cuda()) {
            return linnet::runtime::cpu_alloc(bytes);
        }
        void* pointer = nullptr;
        if (driver->cuMemAlloc(&pointer, bytes == 0 ? 1 : bytes) != linnet::runtime::cuda_success) {
            return nullptr;
        }
        return pointer;
    }

    void release(void* pointer) {
        if (!cuda()) {
            linnet::runtime::cpu_free(pointer);
            return;
        }
        driver->cuMemFree(pointer);
    }
};
