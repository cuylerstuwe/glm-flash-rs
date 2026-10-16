// Qualification-only forwarding library. Never selected by normal service pins.
// Rust remains the owner; this injects one bounded native failure/delay so the
// all-rank error and actual-retirement boundaries can be exercised deterministically.
#include "bridge.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>

#ifndef AXG_REAL_LIBRARY
#error "Build with an explicit immutable qualified native library path"
#endif

namespace {
struct Real {
  void* library;
  Real() : library(dlopen(AXG_REAL_LIBRARY, RTLD_NOW | RTLD_LOCAL)) {
    if (!library) {
      std::fputs("grammar qualification shim: qualified library unavailable\n", stderr);
      std::abort();
    }
  }
  // Keep the backing library loaded until process exit, including every native
  // handle destructor. This is an intentionally separate disposable test process.
};
template <typename Function> Function symbol(const char* name) {
  static Real real;
  void* address = dlsym(real.library, name);
  if (!address) {
    std::fputs("grammar qualification shim: native symbol unavailable\n", stderr);
    std::abort();
  }
  return reinterpret_cast<Function>(address);
}
const char* mode() {
  static const char* value = [] {
    const char* s = std::getenv("ALASKA_GRAMMAR_QUALIFICATION_FAULT");
    if (!s) return "none";
    for (const char* allowed : {"none", "compile-error", "mask-error", "accept-error", "compile-delay"}) {
      if (std::strcmp(s, allowed) == 0) return s;
    }
    std::fputs("grammar qualification shim: unknown fault mode\n", stderr);
    std::abort();
  }();
  return value;
}
bool once(const char* wanted, std::atomic<bool>& used) {
  if (std::strcmp(mode(), wanted) != 0 || used.exchange(true)) return false;
  std::fprintf(stderr, "ALASKA_GRAMMAR_QUALIFICATION_FAULT %s\n", wanted);
  std::fflush(stderr);
  return true;
}
int injected(char* error, size_t capacity, const char* where) {
  if (error && capacity) std::snprintf(error, capacity, "injected qualification %s failure", where);
  return -1;
}
}

#define FORWARD(result, name, parameters, arguments) \
extern "C" result name parameters { \
  static auto real = symbol<decltype(&name)>(#name); \
  return real arguments; \
}

FORWARD(uint32_t, axg_abi_version, (void), ())
FORWARD(const char*, axg_source_revision, (void), ())
FORWARD(int, axg_compiler_new,
  (const axg_bytes* v, size_t n, int type, int prefix, const int32_t* stops, size_t ns,
   int threads, uint64_t bytes, axg_compiler** out, char* error, size_t cap),
  (v, n, type, prefix, stops, ns, threads, bytes, out, error, cap))
FORWARD(void, axg_compiler_free, (axg_compiler* p), (p))
FORWARD(void, axg_grammar_free, (axg_grammar* p), (p))
FORWARD(int, axg_grammar_bytes, (const axg_grammar* p, uint64_t* n, char* e, size_t c), (p,n,e,c))
FORWARD(int, axg_compiler_cache_bytes, (const axg_compiler* p, uint64_t* n, char* e, size_t c), (p,n,e,c))
FORWARD(int, axg_matcher_new, (const axg_grammar* p, axg_matcher** out, char* e, size_t c), (p,out,e,c))
FORWARD(void, axg_matcher_free, (axg_matcher* p), (p))
FORWARD(int, axg_matcher_rollback, (axg_matcher* p, size_t n, char* e, size_t c), (p,n,e,c))
FORWARD(int, axg_matcher_terminated, (const axg_matcher* p, int* out, char* e, size_t c), (p,out,e,c))
FORWARD(int, axg_matcher_fork, (const axg_matcher* p, axg_matcher** out, char* e, size_t c), (p,out,e,c))

extern "C" int axg_compile(axg_compiler* p, int kind, axg_bytes source,
                             axg_grammar** out, char* error, size_t cap) {
  static auto real = symbol<decltype(&axg_compile)>("axg_compile");
  static std::atomic<bool> failed{false}, delayed{false};
  if (once("compile-error", failed)) {
    if (out) *out = nullptr;
    return injected(error, cap, "compile");
  }
  if (once("compile-delay", delayed)) {
    // Below the owner retirement deadline; cancellation must retain its native
    // job and foreground ticket until this call and result destruction finish.
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    std::fputs("ALASKA_GRAMMAR_QUALIFICATION_DELAY_FINISHED\n", stderr);
    std::fflush(stderr);
  }
  return real(p, kind, source, out, error, cap);
}
extern "C" int axg_matcher_fill(axg_matcher* p, int32_t* words, size_t n,
                                  int* needed, char* error, size_t cap) {
  static auto real = symbol<decltype(&axg_matcher_fill)>("axg_matcher_fill");
  static std::atomic<bool> failed{false};
  if (once("mask-error", failed)) {
    if (needed) *needed = 0;
    if (words && n) std::memset(words, 0, n * sizeof(int32_t));
    return injected(error, cap, "mask");
  }
  return real(p, words, n, needed, error, cap);
}
extern "C" int axg_matcher_accept(axg_matcher* p, int32_t id, int* accepted,
                                    char* error, size_t cap) {
  static auto real = symbol<decltype(&axg_matcher_accept)>("axg_matcher_accept");
  static std::atomic<bool> failed{false};
  if (once("accept-error", failed)) {
    if (accepted) *accepted = 0;
    return injected(error, cap, "accept");
  }
  return real(p, id, accepted, error, cap);
}
