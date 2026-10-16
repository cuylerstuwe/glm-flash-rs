#include "bridge.h"
#include <xgrammar/xgrammar.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kMaxVocab = 154880;
constexpr size_t kInputBytes = 16 << 20;
constexpr uint64_t kCacheBytes = 128 << 20;
constexpr const char *kRevision = "97787376faee5ed8466cfad57c99855e4ce2f6aa";
void require(bool okay, const char *message) {
  if (!okay) throw std::invalid_argument(message);
}
std::string read(axg_bytes bytes) {
  require(bytes.length <= kInputBytes && (bytes.data || !bytes.length), "input byte bounds");
  return bytes.length ? std::string(reinterpret_cast<const char *>(bytes.data), bytes.length) : std::string();
}
void message(char *error, size_t capacity, const char *text) noexcept {
  if (!error || !capacity) return;
  const size_t length = strnlen(text, capacity - 1);
  std::memcpy(error, text, length);
  error[length] = '\0';
}
template <class F> int checked(char *error, size_t capacity, F &&f) noexcept {
  if (error && capacity) error[0] = '\0';
  try { f(); return 0; }
  catch (const std::exception &e) { message(error, capacity, e.what()); }
  catch (...) { message(error, capacity, "unknown native grammar exception"); }
  return -1;
}
}
struct axg_compiler {
  xgrammar::GrammarCompiler value;
  int vocab;
};
struct axg_grammar {
  xgrammar::CompiledGrammar value;
  int vocab;
};
struct axg_matcher {
  xgrammar::GrammarMatcher value;
  int vocab;
  size_t depth;
  bool failed;
};
namespace {
template <class F> int mutate(axg_matcher *matcher, char *error, size_t capacity, F &&f) noexcept {
  const int result = checked(error, capacity, [&] {
    require(matcher && !matcher->failed, "missing or failed grammar matcher");
    f();
  });
  if (result != 0 && matcher) matcher->failed = true;
  return result;
}
}
extern "C" {
uint32_t axg_abi_version(void) { return 1; }
const char *axg_source_revision(void) { return kRevision; }
int axg_compiler_new(const axg_bytes *vocab, size_t count, int type, int prefix,
                     const int32_t *stops, size_t stop_count, int threads, uint64_t cache,
                     axg_compiler **out, char *error, size_t capacity) {
  if (out) *out = nullptr;
  return checked(error, capacity, [&] {
    require(out && vocab && count > 0 && count <= kMaxVocab, "vocabulary bounds");
    require(type >= 0 && type <= 2 && (prefix == 0 || prefix == 1), "tokenizer metadata");
    require(stops && stop_count > 0 && stop_count <= 16, "stop token count");
    require(threads > 0 && threads <= 8 && cache > 0 && cache <= kCacheBytes, "compiler bounds");
    std::vector<std::string> strings;
    strings.reserve(count);
    size_t bytes = 0;
    for (size_t i = 0; i < count; ++i) {
      require(vocab[i].length <= kInputBytes - bytes, "vocabulary byte bound");
      bytes += vocab[i].length;
      strings.push_back(read(vocab[i]));
    }
    std::vector<int32_t> ids;
    for (size_t i = 0; i < stop_count; ++i) {
      require(stops[i] >= 0 && static_cast<size_t>(stops[i]) < count, "stop token outside vocabulary");
      require(std::find(ids.begin(), ids.end(), stops[i]) == ids.end(), "duplicate stop token");
      ids.push_back(stops[i]);
    }
    xgrammar::TokenizerInfo tokenizer(strings, static_cast<xgrammar::VocabType>(type),
                                      static_cast<int>(count), ids, prefix != 0);
    *out = new axg_compiler{xgrammar::GrammarCompiler(tokenizer, threads, true, static_cast<int64_t>(cache)),
                            static_cast<int>(count)};
  });
}
void axg_compiler_free(axg_compiler *compiler) { delete compiler; }
int axg_compile(axg_compiler *compiler, int kind, axg_bytes source, axg_grammar **out,
                char *error, size_t capacity) {
  if (out) *out = nullptr;
  return checked(error, capacity, [&] {
    require(compiler && out, "missing grammar compiler/output");
    const auto text = read(source);
    require(!text.empty(), "empty grammar source");
    xgrammar::CompiledGrammar compiled(xgrammar::NullObj{});
    switch (kind) {
      case 0: compiled = compiler->value.CompileJSONSchema(text, true, std::nullopt, std::nullopt,
                                                         true, 32, false); break;
      case 1: compiled = compiler->value.CompileRegex(text); break;
      case 2: compiled = compiler->value.CompileGrammar(text); break;
      case 3: compiled = compiler->value.CompileStructuralTag(text); break;
      default: throw std::invalid_argument("unknown grammar kind");
    }
    require(compiled.MemorySizeBytes() <= kCacheBytes, "compiled grammar exceeds active byte bound");
    *out = new axg_grammar{std::move(compiled), compiler->vocab};
  });
}
void axg_grammar_free(axg_grammar *grammar) { delete grammar; }
int axg_grammar_bytes(const axg_grammar *grammar, uint64_t *out, char *error, size_t capacity) {
  if (out) *out = 0;
  return checked(error, capacity, [&] {
    require(grammar && out, "missing compiled grammar/output");
    *out = grammar->value.MemorySizeBytes();
  });
}
int axg_compiler_cache_bytes(const axg_compiler *compiler, uint64_t *out, char *error, size_t capacity) {
  if (out) *out = 0;
  return checked(error, capacity, [&] {
    require(compiler && out, "missing compiler/output");
    const auto bytes = compiler->value.GetCacheSizeBytes();
    require(bytes >= 0, "negative cache size");
    *out = static_cast<uint64_t>(bytes);
  });
}
int axg_matcher_new(const axg_grammar *grammar, axg_matcher **out, char *error, size_t capacity) {
  if (out) *out = nullptr;
  return checked(error, capacity, [&] {
    require(grammar && out, "missing compiled grammar/output");
    *out = new axg_matcher{xgrammar::GrammarMatcher(grammar->value), grammar->vocab, 0, false};
  });
}
void axg_matcher_free(axg_matcher *matcher) { delete matcher; }
int axg_matcher_accept(axg_matcher *matcher, int32_t token, int *accepted,
                       char *error, size_t capacity) {
  if (accepted) *accepted = 0;
  return mutate(matcher, error, capacity, [&] {
    require(accepted && token >= 0 && token < matcher->vocab, "accepted token bounds");
    require(matcher->depth < (1 << 20), "matcher token bound");
    *accepted = matcher->value.AcceptToken(token);
    if (*accepted) ++matcher->depth;
  });
}
int axg_matcher_fill(axg_matcher *matcher, int32_t *words, size_t count, int *needs,
                     char *error, size_t capacity) {
  if (needs) *needs = 0;
  return mutate(matcher, error, capacity, [&] {
    require(words && needs && count == static_cast<size_t>((matcher->vocab + 31) / 32), "bitmask bounds");
    require(reinterpret_cast<uintptr_t>(words) % alignof(int32_t) == 0, "bitmask alignment");
    std::fill(words, words + count, -1);
    int64_t shape = static_cast<int64_t>(count);
    DLTensor tensor{words, {kDLCPU, 0}, 1, {kDLInt, 32, 1}, &shape, nullptr, 0};
    *needs = matcher->value.FillNextTokenBitmask(&tensor);
  });
}
int axg_matcher_rollback(axg_matcher *matcher, size_t steps, char *error, size_t capacity) {
  return mutate(matcher, error, capacity, [&] {
    require(steps <= matcher->depth && steps <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "rollback bounds");
    if (steps) matcher->value.Rollback(static_cast<int>(steps));
    matcher->depth -= steps;
  });
}
int axg_matcher_terminated(const axg_matcher *matcher, int *out, char *error, size_t capacity) {
  if (out) *out = 0;
  return checked(error, capacity, [&] {
    require(matcher && !matcher->failed && out, "missing or failed matcher/output");
    *out = matcher->value.IsTerminated();
  });
}
int axg_matcher_fork(const axg_matcher *matcher, axg_matcher **out, char *error, size_t capacity) {
  if (out) *out = nullptr;
  return checked(error, capacity, [&] {
    require(matcher && !matcher->failed && out, "missing or failed matcher/output");
    *out = new axg_matcher{matcher->value.Fork(), matcher->vocab, matcher->depth, false};
  });
}
}
