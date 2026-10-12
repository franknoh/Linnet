// The runtime's CPU path: a plan of calls over its arena, in order, and
// what it refuses there. Its CUDA path runs where there is a GPU.

#include "linnet/runtime/runtime.h"

#include "test.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace {

// `outputs[0] = inputs[0] * factor`, over float32 vectors.
int scale(void* user,
          const LnTensor* inputs,
          size_t input_count,
          LnTensor* outputs,
          size_t output_count,
          LnContext* context) {
    if (input_count != 1 || output_count != 1 || context->stream != nullptr) {
        return 1;
    }
    const float factor = *static_cast<const float*>(user);
    const auto* in = static_cast<const float*>(inputs[0].data);
    auto* out = static_cast<float*>(outputs[0].data);
    for (std::int64_t i = 0; i < inputs[0].shape[0]; ++i) {
        out[i] = in[i] * factor;
    }
    return 0;
}

int refuse(void*, const LnTensor*, size_t, LnTensor*, size_t, LnContext*) {
    return 1;
}

LnTensor vector_at(void* data, std::int64_t* length) {
    LnTensor tensor{};
    tensor.data = data;
    tensor.device = {LN_CPU, 0};
    tensor.ndim = 1;
    tensor.dtype = {2, 32, 1};
    tensor.shape = length;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    return tensor;
}

} // namespace

TEST("runtime: a CPU plan runs its calls in order over its arena") {
    LnRuntime* runtime = nullptr;
    CHECK_EQ(ln_runtime_create({LN_CPU, 0}, &runtime), LN_OK);
    CHECK(ln_runtime_stream(runtime) == nullptr);
    LnPlan* plan = nullptr;
    CHECK_EQ(ln_plan_create(runtime, sizeof(float) * 12, &plan), LN_OK);
    auto* arena = static_cast<float*>(ln_plan_arena(plan));
    std::int64_t length = 4;
    const LnTensor a = vector_at(arena, &length);
    const LnTensor b = vector_at(arena + 4, &length);
    const LnTensor c = vector_at(arena + 8, &length);
    float two = 2.0F;
    float ten = 10.0F;
    LnExecutable* doubled = nullptr;
    LnExecutable* tenfold = nullptr;
    CHECK_EQ(ln_callback_create(scale, &two, &doubled), LN_OK);
    CHECK_EQ(ln_callback_create(scale, &ten, &tenfold), LN_OK);
    CHECK_EQ(ln_plan_add(plan, doubled, &a, 1, &b, 1, 0, 0), LN_OK);
    CHECK_EQ(ln_plan_add(plan, tenfold, &b, 1, &c, 1, 0, 0), LN_OK);
    for (int i = 0; i < 4; ++i) {
        arena[i] = static_cast<float>(i);
    }
    CHECK_EQ(ln_plan_run(plan), LN_OK);
    CHECK_EQ(ln_plan_synchronize(plan), LN_OK);
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(arena[8 + i], static_cast<float>(i) * 20.0F);
    }
    // A CPU plan has no graph to capture, and no kernel to launch.
    CHECK_EQ(ln_plan_capture(plan), LN_UNAVAILABLE);
    const std::array<std::uint32_t, 3> one{1, 1, 1};
    LnExecutable* kernel = nullptr;
    CHECK_EQ(ln_kernel_create(runtime, "", "f", one.data(), one.data(), 0, nullptr, 0, &kernel),
             LN_UNAVAILABLE);
    ln_plan_destroy(plan);
    ln_executable_destroy(doubled);
    ln_executable_destroy(tenfold);
    ln_runtime_destroy(runtime);
}

TEST("runtime: a failing call stops the run and says which") {
    LnRuntime* runtime = nullptr;
    CHECK_EQ(ln_runtime_create({LN_CPU, 0}, &runtime), LN_OK);
    LnPlan* plan = nullptr;
    CHECK_EQ(ln_plan_create(runtime, 64, &plan), LN_OK);
    LnExecutable* failing = nullptr;
    CHECK_EQ(ln_callback_create(refuse, nullptr, &failing), LN_OK);
    CHECK_EQ(ln_plan_add(plan, failing, nullptr, 0, nullptr, 0, 0, 64), LN_OK);
    CHECK_EQ(ln_plan_add(plan, failing, nullptr, 0, nullptr, 0, 32, 64), LN_ERROR);
    CHECK(std::string(ln_last_error()).find("outside the arena") != std::string::npos);
    CHECK_EQ(ln_plan_run(plan), LN_ERROR);
    CHECK(std::string(ln_last_error()).find("callback returned an error") != std::string::npos);
    ln_plan_destroy(plan);
    ln_executable_destroy(failing);
    ln_runtime_destroy(runtime);
}

TEST("runtime: memory goes back to the runtime that gave it") {
    LnRuntime* runtime = nullptr;
    CHECK_EQ(ln_runtime_create({LN_CPU, 0}, &runtime), LN_OK);
    void* pointer = nullptr;
    CHECK_EQ(ln_alloc(runtime, 100, &pointer), LN_OK);
    CHECK(reinterpret_cast<std::uintptr_t>(pointer) % 256 == 0);
    CHECK_EQ(ln_free(runtime, pointer), LN_OK);
    CHECK_EQ(ln_free(runtime, pointer), LN_ERROR);
    // Another device type has no runtime yet.
    LnRuntime* other = nullptr;
    CHECK_EQ(ln_runtime_create({7, 0}, &other), LN_UNAVAILABLE);
    ln_runtime_destroy(runtime);
}
