#include "driver.hpp"

#include <mutex>
#include <optional>
#include <string>
#include <type_traits>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace linnet::runtime {

namespace {

// The library's handle, or nullptr where there is none.
void* open_library() {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
#elif defined(__APPLE__)
    return nullptr;
#else
    return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

struct Loaded {
    std::optional<Driver> driver;
    std::string error;
};

Loaded load() {
    Loaded out;
    void* library = open_library();
    if (library == nullptr) {
        out.error = "no CUDA driver on this machine";
        return out;
    }
    Driver d{};
    std::string missing;
    // The versioned names are the ones the driver exports for each call's
    // current behaviour.
    const auto bind = [&](auto& field, const char* name) {
        // A function's address, as the loader returns it.
        field = reinterpret_cast<std::remove_reference_t<decltype(field)>>(symbol(library, name));
        if (field == nullptr) {
            missing += missing.empty() ? name : std::string(", ") + name;
        }
    };
    bind(d.cuInit, "cuInit");
    bind(d.cuGetErrorString, "cuGetErrorString");
    bind(d.cuDeviceGet, "cuDeviceGet");
    bind(d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    bind(d.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    bind(d.cuCtxPushCurrent, "cuCtxPushCurrent_v2");
    bind(d.cuCtxPopCurrent, "cuCtxPopCurrent_v2");
    bind(d.cuStreamCreate, "cuStreamCreate");
    bind(d.cuStreamDestroy, "cuStreamDestroy_v2");
    bind(d.cuStreamSynchronize, "cuStreamSynchronize");
    bind(d.cuMemAlloc, "cuMemAlloc_v2");
    bind(d.cuMemFree, "cuMemFree_v2");
    bind(d.cuModuleLoadData, "cuModuleLoadData");
    bind(d.cuModuleGetFunction, "cuModuleGetFunction");
    bind(d.cuModuleUnload, "cuModuleUnload");
    bind(d.cuFuncSetAttribute, "cuFuncSetAttribute");
    bind(d.cuLaunchKernel, "cuLaunchKernel");
    bind(d.cuStreamBeginCapture, "cuStreamBeginCapture_v2");
    bind(d.cuStreamEndCapture, "cuStreamEndCapture");
    bind(d.cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags");
    bind(d.cuGraphLaunch, "cuGraphLaunch");
    bind(d.cuGraphExecDestroy, "cuGraphExecDestroy");
    bind(d.cuGraphDestroy, "cuGraphDestroy");
    if (!missing.empty()) {
        out.error = "the CUDA driver lacks " + missing;
        return out;
    }
    if (const CUresult result = d.cuInit(0); result != cuda_success) {
        out.error = d.message("cuInit", result);
        return out;
    }
    out.driver = d;
    return out;
}

} // namespace

std::string Driver::message(const char* what, CUresult result) const {
    const char* text = nullptr;
    if (cuGetErrorString == nullptr || cuGetErrorString(result, &text) != cuda_success ||
        text == nullptr) {
        return std::string(what) + ": CUDA error " + std::to_string(result);
    }
    return std::string(what) + ": " + text;
}

const Driver* driver(std::string& error) {
    static std::once_flag once;
    static Loaded loaded;
    std::call_once(once, [] { loaded = load(); });
    if (!loaded.driver) {
        error = loaded.error;
        return nullptr;
    }
    return &*loaded.driver;
}

} // namespace linnet::runtime
