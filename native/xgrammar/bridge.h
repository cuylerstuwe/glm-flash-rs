#ifndef ALASKA_XGRAMMAR_BRIDGE_H
#define ALASKA_XGRAMMAR_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Opaque single-owner handles. Every operation returns 0 on success, -1 on
 * error. Error text is bounded and caller-owned; no exception crosses the ABI.
 * No handle may be used concurrently or after destruction. Vocabulary entries
 * are length-delimited bytes, not C strings. Outputs are initialized on error.
 */
typedef struct axg_compiler axg_compiler;
typedef struct axg_grammar axg_grammar;
typedef struct axg_matcher axg_matcher;
typedef struct { const uint8_t *data; size_t length; } axg_bytes;
uint32_t axg_abi_version(void);
const char *axg_source_revision(void);
int axg_compiler_new(const axg_bytes *vocab, size_t vocab_count,
                     int vocab_type, int add_prefix_space,
                     const int32_t *stop_ids, size_t stop_count,
                     int max_threads, uint64_t cache_bytes,
                     axg_compiler **out, char *error, size_t error_capacity);
void axg_compiler_free(axg_compiler *compiler);
/* kind: 0 JSON schema, 1 regex, 2 EBNF, 3 structural-tag JSON.
 * JSON schema defaults: strict, any whitespace, max 32 blanks,
 * insertion-ordered properties (any_order=false). Choice becomes EBNF in Rust.
 */
int axg_compile(axg_compiler *compiler, int kind, axg_bytes source,
                axg_grammar **out, char *error, size_t error_capacity);
void axg_grammar_free(axg_grammar *grammar);
int axg_grammar_bytes(const axg_grammar *grammar, uint64_t *out,
                     char *error, size_t error_capacity);
int axg_compiler_cache_bytes(const axg_compiler *compiler, uint64_t *out,
                            char *error, size_t error_capacity);
int axg_matcher_new(const axg_grammar *grammar, axg_matcher **out,
                    char *error, size_t error_capacity);
void axg_matcher_free(axg_matcher *matcher);
int axg_matcher_accept(axg_matcher *matcher, int32_t token, int *accepted,
                       char *error, size_t error_capacity);
int axg_matcher_fill(axg_matcher *matcher, int32_t *words, size_t word_count,
                     int *needs_mask, char *error, size_t error_capacity);
int axg_matcher_rollback(axg_matcher *matcher, size_t steps,
                         char *error, size_t error_capacity);
int axg_matcher_terminated(const axg_matcher *matcher, int *out,
                           char *error, size_t error_capacity);
int axg_matcher_fork(const axg_matcher *matcher, axg_matcher **out,
                     char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
