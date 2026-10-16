// C ABI for the retained native proxy/device protocol. Rust owns configuration,
// HCA discovery, consensus, bootstrap, CUDA buffers, health and graph ordering.
#include "roce_native.cpp"
#include <sstream>

struct AkRoceConfig {
    int32_t world, rank, n_hca, cpu, traffic_class, threads, hostreg, lean;
    uint64_t slot_bytes, counters;
    uint32_t stripe_min, inline_max;
    int32_t lazy_cq, copy_mode, max_blocks, reserved;
};
struct AkRoce {
    uint64_t host = 0, dev = 0, layout[7] = {};
    std::unique_ptr<tfroce::Proxy> proxy;
    std::unique_ptr<tfroce::Launcher> launcher;
    ~AkRoce() {
        proxy.reset();
        launcher.reset();
        if (host) tfroce::free_pinned(host);
    }
};
static thread_local std::string last_error;
template <class F> static int checked(F f) noexcept {
    try { f(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown native RoCE error"; return -1; }
}
extern "C" const char *ak_roce_error() noexcept { return last_error.c_str(); }
extern "C" uint64_t ak_roce_config_bytes() noexcept { return sizeof(AkRoceConfig); }
extern "C" uint64_t ak_roce_blob_bytes() noexcept { return sizeof(tfroce::Blob); }
extern "C" int ak_roce_new(const AkRoceConfig *c, const char *const *names,
                            const int32_t *gids, AkRoce **out) noexcept {
    return checked([&] {
        if (!c || !names || !gids || !out || c->world != 4 || c->rank < 0 || c->rank >= 4 ||
            c->n_hca < 1 || c->n_hca > 2 || !c->counters || c->max_blocks < 1 ||
            c->max_blocks > 64 || (c->max_blocks & (c->max_blocks - 1)))
            throw std::invalid_argument("invalid native RoCE configuration");
        *out = nullptr;
        auto h = std::make_unique<AkRoce>();
        if (!tfroce::layout(c->world, c->slot_bytes, h->layout))
            throw std::invalid_argument("invalid native RoCE layout");
        auto allocation = tfroce::alloc_pinned(h->layout[4], c->hostreg);
        h->host = allocation[0]; h->dev = allocation[1];
        std::vector<std::string> devices;
        std::vector<int> indices;
        for (int i = 0; i < c->n_hca; ++i) {
            if (!names[i]) throw std::invalid_argument("null HCA name");
            devices.emplace_back(names[i]); indices.push_back(gids[i]);
        }
        h->proxy = std::make_unique<tfroce::Proxy>(c->world, c->rank, devices, indices,
            c->traffic_class, h->host, h->layout[4], c->slot_bytes, c->stripe_min,
            c->inline_max, bool(c->lazy_cq), 0);
        int classes = 0;
        for (int n = c->max_blocks; n; n >>= 1) ++classes;
        h->launcher = std::make_unique<tfroce::Launcher>(h->dev + h->layout[0],
            h->dev + h->layout[1], h->dev + h->layout[2], h->dev + h->layout[3],
            c->slot_bytes, c->counters, c->counters + 4 * (1 + 2 * classes),
            c->world, c->rank, c->n_hca, int(h->layout[5]), int(h->layout[6]),
            c->threads, c->stripe_min, bool(c->lean), 0, 0, c->copy_mode);
        *out = h.release();
    });
}
extern "C" int ak_roce_blob(AkRoce *h, uint8_t *out, uint64_t bytes) noexcept {
    return checked([&] {
        if (!h || !out) throw std::invalid_argument("null RoCE blob argument");
        auto blob = h->proxy->local_blob();
        if (bytes != blob.size()) throw std::invalid_argument("RoCE blob size mismatch");
        memcpy(out, blob.data(), blob.size());
    });
}
extern "C" int ak_roce_connect(AkRoce *h, const uint8_t *blobs, uint64_t bytes,
                                int32_t cpu) noexcept {
    return checked([&] {
        if (!h || !blobs || bytes != 4 * sizeof(tfroce::Blob))
            throw std::invalid_argument("invalid RoCE connection records");
        h->proxy->connect(std::string(reinterpret_cast<const char *>(blobs), bytes));
        h->proxy->start(cpu);
    });
}
extern "C" int ak_roce_run(AkRoce *h, uint64_t input, uint64_t output, uint64_t bytes,
                            int32_t grid, uint64_t stage, uint64_t tail,
                            uint64_t timeout_ns, uint64_t stream) noexcept {
    return checked([&] {
        if (!h) throw std::invalid_argument("null RoCE runtime");
        h->launcher->run(input, output, bytes, grid, stage, tail, timeout_ns, stream);
    });
}
extern "C" int ak_roce_check(AkRoce *h) noexcept {
    return checked([&] {
        if (!h) throw std::invalid_argument("null RoCE runtime");
        auto ctrl = reinterpret_cast<const volatile uint32_t *>(h->host + h->layout[3]);
        if (ctrl[tfroce::CTRL_FAILED] || h->proxy->failed()) {
            std::ostringstream message;
            message << "RoCE poisoned: " << h->proxy->error() << " control=";
            for (int i = 0; i < tfroce::CTRL_WORDS; ++i) message << uint32_t(ctrl[i]) << ',';
            message << " failure-watch=";
            for (auto word : h->proxy->watch()) message << word << ',';
            throw std::runtime_error(message.str());
        }
    });
}
// Rust synchronizes the owning CUDA stream before destruction; proxy stop joins
// the thread before unregistering/freeing any host or GPU-visible allocation.
extern "C" void ak_roce_destroy(AkRoce *h) noexcept { delete h; }
