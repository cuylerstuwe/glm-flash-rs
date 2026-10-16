// Independent official TokenizerInfo conversion for whole-vocabulary Rust checks.
#include <xgrammar/tokenizer_info.h>
#include <picojson.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
using picojson::value;
using picojson::object;
using picojson::array;
static void u32(std::ofstream &out, uint32_t n) {
  char bytes[4];
  for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>(n >> (8*i));
  out.write(bytes, 4);
}
int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: tokenizer_oracle tokenizer.json new-output-prefix");
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) throw std::runtime_error("tokenizer input");
    std::string input((std::istreambuf_iterator<char>(in)), {});
    value root;
    const auto error = picojson::parse(root, input);
    if (!error.empty()) throw std::runtime_error(error);
    const auto &r = root.get<object>();
    std::vector<std::string> vocab(154880);
    const auto &base = r.at("model").get<object>().at("vocab").get<object>();
    for (const auto &[token, id] : base) vocab.at(static_cast<size_t>(id.get<double>())) = token;
    const auto &added = r.at("added_tokens").get<array>();
    for (const auto &token : added) {
      const auto &a = token.get<object>();
      vocab.at(static_cast<size_t>(a.at("id").get<double>())) = a.at("content").get<std::string>();
    }
    const auto metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(input);
    std::cout << "{\"detected_metadata\":" << metadata << ",\"views\":[";
    for (int tools = 0; tools < 2; ++tools) {
      auto encoded = vocab;
      for (const auto &token : added) {
        const auto &a = token.get<object>();
        auto id = static_cast<size_t>(a.at("id").get<double>());
        const bool keep = tools && (id==154843 || id==154844 || id==154847 || id==154848 || id==154849 || id==154850);
        if (!keep) encoded.at(id).clear();
      }
      xgrammar::TokenizerInfo info(encoded, xgrammar::VocabType::BYTE_LEVEL, 154880,
                                    std::vector<int32_t>{154820,154827,154829,154828}, false);
      const std::string path = std::string(argv[2]) + (tools ? "-tools.bin" : "-text.bin");
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      for (const auto &token : info.GetDecodedVocab()) {
        u32(out, token.size());
        out.write(token.data(), token.size());
      }
      if (!out) throw std::runtime_error("vocabulary output");
      if (tools) std::cout << ',';
      std::cout << "{\"tools\":" << (tools?"true":"false")
                << ",\"vocab\":" << info.GetVocabSize()
                << ",\"special\":" << info.GetSpecialTokenIds().size()
                << ",\"stops\":" << info.GetStopTokenIds().size() << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
