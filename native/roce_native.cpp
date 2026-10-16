// Native implementation portions Copyright 2026 Jay Leaton, Apache-2.0.
// See LICENSES/Apache-2.0.txt and LICENSES/Native-Components-NOTICE.txt.
// Modified for a native Rust interface: remove Python/Torch bindings; explicit CUDA
// stream ownership and exception-safe C ABI. Retain proxy and device protocol.
// RDMA proxy and bindings for the one-shot RoCE all-gather.
//
// Adapted from b12x's RoCEnante proxy (b12x/comm/roce/_roce_proxy.c, https://github.com/local-inference-lab/b12x,
// Copyright 2026 Luke Alonso and the b12x contributors, Apache License 2.0; see LICENSES/Apache-2.0.txt and
// THIRD_PARTY_NOTICES.md). Changes: C++ in the torch extension instead of a separately built C library; one RoCE
// v2 GID index per HCA (they differ between the two CX7 functions and move on reboot); an optional CPU for the
// proxy thread; a magic word in the connection record; the pinned region allocation and the kernel launcher are
// bound here too.
//
// One rank owns one pinned host region, laid out as
//
//   recv[src][slot]       world * SLOTS * slot_bytes   filled by the peers' RDMA writes
//   flag[src][slot][hca]  world * SLOTS * MAX_HCAS * FLAG_STRIDE   the sequence written after that HCA's stripe
//   send[slot]            SLOTS * slot_bytes           staged by the local GPU kernel
//   ctrl                  FLAG_STRIDE                  the control record (roce_common.h)
//
// The GPU kernel stages its input into send[seq & 1], publishes the padded byte count and seq in ctrl, then spins
// on flag[peer][seq & 1][hca] for every peer and HCA. The proxy thread below spins on ctrl[CTRL_SEQ] and, for every
// peer, stripes the payload across the HCAs; each stripe is followed by its own 4-byte seq write on the same
// reliable-connected queue pair, so its flag cannot become visible before its payload. Nothing on the receive path
// involves the host: the DGX Spark's GB10 reads the pinned host memory the NIC wrote in place (no GPUDirect RDMA).
//
// the proxy thread (and the tests' Loop) also polls ``FailWatch`` (roce_watch.h) every iteration: the
// host's view of a failed wait's flag at the moment of the failure, and whether it arrives later.
//
// (each off by default): ``stripe_min`` -- a shard below it travels on one HCA
// (seq % HCAs; one post, one flag to wait for) instead of being striped; ``inline_max`` -- a stripe of at most that
// many bytes is posted inline (the CPU copies the staged bytes into the work request: no NIC DMA read of the send
// slot); ``lazy_cq`` -- completions are reaped in the idle loop, not right after each post; ``trace`` -- the proxy
// stamps when it saw each doorbell and when its posts returned (CLOCK_REALTIME ns, a ring the size of the kernel's).

#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "roce_common.h"
#include "roce_watch.h"

namespace tfroce {

constexpr int MAX_PEERS = 16;
constexpr int MAX_HCAS = 2;
constexpr int SLOTS = 2;
constexpr int FLAG_STRIDE = 128;
constexpr int PORT = 1;
constexpr int SEND_DEPTH = 256;
constexpr uint32_t BLOB_MAGIC = 0x54465243u;   // "TFRC", bumped with the record's layout
// Model graphs leave sub-millisecond gaps between collectives: keep the proxy hot across them (a sleep there adds a
// scheduler wakeup to every collective on the critical path); after this many idle polls it naps 20 us at a time.
constexpr uint64_t IDLE_SPINS = 20000000ull;

struct Blob {
    uint32_t magic;
    uint32_t n_hca;
    uint64_t region_addr;
    uint32_t rkey[MAX_HCAS];
    uint32_t lid[MAX_HCAS];
    uint8_t gid[MAX_HCAS][16];
    uint32_t mtu[MAX_HCAS];
    uint32_t qp_num[MAX_HCAS][MAX_PEERS];
};

struct Hca {
    uint32_t inline_cap = 0;         // the queue pairs' max_inline_data as created
    ibv_context *ctx = nullptr;
    ibv_pd *pd = nullptr;
    ibv_mr *mr = nullptr;
    ibv_cq *cq = nullptr;
    ibv_qp *qp[MAX_PEERS] = {};
    uint32_t outstanding[MAX_PEERS] = {};
    uint64_t writes_completed = 0;
    uint64_t bytes_posted = 0;
    ibv_gid gid{};
    uint16_t lid = 0;
    ibv_mtu mtu = IBV_MTU_1024;
    int gid_index = 0;
};

bool layout(int world, uint64_t slot_bytes, uint64_t out[7]) {
    // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots}
    if (world < 2 || world > MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0 ||
        slot_bytes > (uint64_t(1) << 40)) {
        return false;
    }
    const uint64_t recv_bytes = uint64_t(world) * SLOTS * slot_bytes;
    const uint64_t flag_bytes = uint64_t(world) * SLOTS * MAX_HCAS * FLAG_STRIDE;
    const uint64_t send_bytes = uint64_t(SLOTS) * slot_bytes;
    out[0] = 0;
    out[1] = recv_bytes;
    out[2] = recv_bytes + flag_bytes;
    out[3] = out[2] + send_bytes;
    out[4] = out[3] + FLAG_STRIDE;
    out[5] = FLAG_STRIDE;
    out[6] = SLOTS;
    return true;
}

class Proxy {
  public:
    Proxy(int world, int rank, std::vector<std::string> names, std::vector<int> gid_index, int traffic_class,
          uint64_t region, uint64_t region_bytes, uint64_t slot_bytes, uint32_t stripe_min, uint32_t inline_max,
          bool lazy_cq, uint32_t trace_entries)
        : world_(world), rank_(rank), n_hca_(int(names.size())), tc_(traffic_class),
          region_(reinterpret_cast<uint8_t *>(region)), region_bytes_(region_bytes), slot_bytes_(slot_bytes),
          stripe_min_(stripe_min), inline_max_(inline_max), lazy_cq_(lazy_cq) {
        if (trace_entries != 0) {
            if ((trace_entries & (trace_entries - 1)) != 0) {
                throw std::runtime_error("roce: trace entries must be a power of two");
            }
            trace_mask_ = trace_entries - 1;
            ptrace_.assign(size_t(trace_entries) * 2, 0);
        }
        uint64_t lay[7];
        if (!layout(world, slot_bytes, lay) || lay[4] > region_bytes || rank < 0 || rank >= world || n_hca_ < 1 ||
            n_hca_ > MAX_HCAS || gid_index.size() != names.size() || tc_ < 0 || tc_ > 255) {
            throw std::runtime_error("roce: invalid runtime geometry");
        }
        recv_off_ = lay[0];
        flag_off_ = lay[1];
        send_off_ = lay[2];
        ctrl_off_ = lay[3];
        for (int h = 0; h < n_hca_; h++) {
            hca_[h].gid_index = gid_index[h];
            if (!open_hca(h, names[h])) {
                std::string e = err_;
                destroy();
                throw std::runtime_error("roce: " + e);
            }
        }
    }

    ~Proxy() { destroy(); }

    std::string local_blob() const {
        Blob b;
        memset(&b, 0, sizeof(b));
        b.magic = BLOB_MAGIC;
        b.n_hca = uint32_t(n_hca_);
        b.region_addr = uint64_t(reinterpret_cast<uintptr_t>(region_));
        for (int h = 0; h < n_hca_; h++) {
            b.rkey[h] = hca_[h].mr->rkey;
            b.lid[h] = hca_[h].lid;
            b.mtu[h] = uint32_t(hca_[h].mtu);
            memcpy(b.gid[h], hca_[h].gid.raw, 16);
            for (int p = 0; p < world_; p++) {
                b.qp_num[h][p] = p == rank_ ? 0 : hca_[h].qp[p]->qp_num;
            }
        }
        return std::string(reinterpret_cast<const char *>(&b), sizeof(b));
    }

    void connect(const std::string &blobs) {
        if (blobs.size() != sizeof(Blob) * size_t(world_)) {
            throw std::runtime_error("roce: connection records of the wrong size (different builds on the ranks?)");
        }
        std::vector<Blob> all(world_);
        memcpy(all.data(), blobs.data(), blobs.size());
        for (int p = 0; p < world_; p++) {
            if (p == rank_) {
                continue;
            }
            if (all[p].magic != BLOB_MAGIC || int(all[p].n_hca) != n_hca_) {
                throw std::runtime_error("roce: rank " + std::to_string(p) +
                                         "'s connection record does not match (build or HCA count)");
            }
            peer_addr_[p] = all[p].region_addr;
            for (int h = 0; h < n_hca_; h++) {
                peer_rkey_[h][p] = all[p].rkey[h];
                if (!connect_qp(h, p, all[p])) {
                    throw std::runtime_error("roce: " + err_);
                }
            }
        }
    }

    void start(int cpu) {
        if (running_.load()) {
            return;
        }
        if (!started_) {
            // a restart continues from the last posted sequence, so ops rung while stopped are still posted
            last_seq_ = ctrl()[CTRL_SEQ];
            started_ = true;
        }
        cpu_ = cpu;
        failed_.store(0);
        running_.store(1);
        int rc = pthread_create(&thread_, nullptr, &Proxy::main_tramp, this);
        if (rc != 0) {
            running_.store(0);
            throw std::runtime_error(std::string("roce: pthread_create: ") + strerror(rc));
        }
    }

    void stop() {
        if (running_.exchange(0)) {
            pthread_join(thread_, nullptr);
        }
    }

    bool failed() const { return failed_.load() != 0; }

    std::string error() const { return err_; }

    std::vector<uint64_t> watch() const {       // FailWatch::read
        uint64_t out[7];
        watch_.read(out);
        return std::vector<uint64_t>(out, out + 7);
    }

    std::vector<uint64_t> stats() const {
        std::vector<uint64_t> out = {ops_posted_, writes_completed_, last_seq_};
        for (int h = 0; h < n_hca_; h++) {
            out.push_back(hca_[h].writes_completed);
            out.push_back(hca_[h].bytes_posted);
        }
        return out;
    }

    // per HCA the inline capacity the queue pairs got, then the posts sent inline and single-HCA ops
    std::vector<uint64_t> stats2() const {
        std::vector<uint64_t> out = {inline_posts_, single_ops_};
        for (int h = 0; h < n_hca_; h++) {
            out.push_back(hca_[h].inline_cap);
        }
        return out;
    }

    // (seen ns, posted ns) of the op whose sequence is ``seq`` (0, 0 without a trace or if overwritten)
    std::vector<uint64_t> trace(uint32_t seq) const {
        if (ptrace_.empty()) {
            return {0, 0};
        }
        const size_t i = size_t(seq & trace_mask_) * 2;
        return {ptrace_[i], ptrace_[i + 1]};
    }

    void destroy() {
        stop();
        for (int h = 0; h < MAX_HCAS; h++) {
            Hca &x = hca_[h];
            for (int p = 0; p < MAX_PEERS; p++) {
                if (x.qp[p] != nullptr) {
                    ibv_destroy_qp(x.qp[p]);
                    x.qp[p] = nullptr;
                }
            }
            if (x.cq != nullptr) {
                ibv_destroy_cq(x.cq);
                x.cq = nullptr;
            }
            if (x.mr != nullptr) {
                ibv_dereg_mr(x.mr);
                x.mr = nullptr;
            }
            if (x.pd != nullptr) {
                ibv_dealloc_pd(x.pd);
                x.pd = nullptr;
            }
            if (x.ctx != nullptr) {
                ibv_close_device(x.ctx);
                x.ctx = nullptr;
            }
        }
    }

  private:
    volatile uint32_t *ctrl() const { return reinterpret_cast<volatile uint32_t *>(region_ + ctrl_off_); }

    // the host address of flag[peer][slot][hca] (nullptr out of range)
    const volatile uint32_t *flag_at(uint32_t peer, uint32_t slot, uint32_t hca) const {
        if (peer >= uint32_t(world_) || slot >= uint32_t(SLOTS) || hca >= uint32_t(n_hca_)) {
            return nullptr;
        }
        return reinterpret_cast<const volatile uint32_t *>(
            region_ + flag_off_ + ((uint64_t(peer) * SLOTS + slot) * uint64_t(n_hca_) + hca) * FLAG_STRIDE);
    }

    void set_err(const char *what, int e) { err_ = std::string(what) + ": " + (e ? strerror(e) : "failed"); }

    bool open_hca(int h, const std::string &name) {
        int num = 0;
        ibv_device **list = ibv_get_device_list(&num);
        if (list == nullptr) {
            set_err("ibv_get_device_list", errno);
            return false;
        }
        ibv_device *dev = nullptr;
        for (int i = 0; i < num; i++) {
            if (name == ibv_get_device_name(list[i])) {
                dev = list[i];
                break;
            }
        }
        if (dev == nullptr) {
            ibv_free_device_list(list);
            err_ = "RDMA device " + name + " not found (is /dev/infiniband passed to the container?)";
            return false;
        }
        Hca &x = hca_[h];
        x.ctx = ibv_open_device(dev);
        ibv_free_device_list(list);
        if (x.ctx == nullptr) {
            set_err("ibv_open_device", errno);
            return false;
        }
        ibv_port_attr port;
        if (ibv_query_port(x.ctx, PORT, &port) != 0) {
            set_err("ibv_query_port", errno);
            return false;
        }
        if (port.state != IBV_PORT_ACTIVE) {
            err_ = "RDMA device " + name + " port 1 is not active";
            return false;
        }
        x.lid = port.lid;
        x.mtu = port.active_mtu;
        if (ibv_query_gid(x.ctx, PORT, x.gid_index, &x.gid) != 0) {
            set_err("ibv_query_gid", errno);
            return false;
        }
        x.pd = ibv_alloc_pd(x.ctx);
        if (x.pd == nullptr) {
            set_err("ibv_alloc_pd", errno);
            return false;
        }
        x.mr = ibv_reg_mr(x.pd, region_, region_bytes_, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (x.mr == nullptr) {
            set_err("ibv_reg_mr(pinned region; memlock limit? run with --ulimit memlock=-1)", errno);
            return false;
        }
        x.cq = ibv_create_cq(x.ctx, SEND_DEPTH * MAX_PEERS, nullptr, nullptr, 0);
        if (x.cq == nullptr) {
            set_err("ibv_create_cq", errno);
            return false;
        }
        for (int p = 0; p < world_; p++) {
            if (p == rank_) {
                continue;
            }
            ibv_qp_init_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.send_cq = x.cq;
            attr.recv_cq = x.cq;
            attr.qp_type = IBV_QPT_RC;
            attr.cap.max_send_wr = SEND_DEPTH;
            attr.cap.max_recv_wr = 1;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            attr.cap.max_inline_data = inline_max_ > 16 ? inline_max_ : 16;
            x.qp[p] = ibv_create_qp(x.pd, &attr);
            if (x.qp[p] == nullptr && attr.cap.max_inline_data > 16) {
                // the device refused the inline size: queue pair without inline payloads
                attr.cap.max_inline_data = 16;
                x.qp[p] = ibv_create_qp(x.pd, &attr);
            }
            if (x.qp[p] == nullptr) {
                set_err("ibv_create_qp", errno);
                return false;
            }
            // ibv_create_qp writes the capacities it granted back into attr.cap ()
            const uint32_t cap = attr.cap.max_inline_data;
            x.inline_cap = (inline_max_ == 0 || cap < 16) ? 0 : (cap < inline_max_ ? cap : inline_max_);
            ibv_qp_attr init;
            memset(&init, 0, sizeof(init));
            init.qp_state = IBV_QPS_INIT;
            init.pkey_index = 0;
            init.port_num = PORT;
            init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
            int rc = ibv_modify_qp(x.qp[p], &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
            if (rc != 0) {
                set_err("ibv_modify_qp(INIT)", rc);
                return false;
            }
        }
        return true;
    }

    bool connect_qp(int h, int p, const Blob &peer) {
        Hca &x = hca_[h];
        ibv_qp_attr rtr;
        memset(&rtr, 0, sizeof(rtr));
        rtr.qp_state = IBV_QPS_RTR;
        rtr.path_mtu = ibv_mtu(peer.mtu[h] < uint32_t(x.mtu) ? peer.mtu[h] : uint32_t(x.mtu));
        rtr.dest_qp_num = peer.qp_num[h][rank_];
        rtr.rq_psn = 0;
        rtr.max_dest_rd_atomic = 1;
        rtr.min_rnr_timer = 12;
        rtr.ah_attr.is_global = 1;
        rtr.ah_attr.dlid = uint16_t(peer.lid[h]);
        rtr.ah_attr.sl = 0;
        rtr.ah_attr.src_path_bits = 0;
        rtr.ah_attr.port_num = PORT;
        memcpy(rtr.ah_attr.grh.dgid.raw, peer.gid[h], 16);
        rtr.ah_attr.grh.sgid_index = uint8_t(x.gid_index);
        rtr.ah_attr.grh.hop_limit = 64;
        rtr.ah_attr.grh.traffic_class = uint8_t(tc_);
        rtr.ah_attr.grh.flow_label = 0;
        int rc = ibv_modify_qp(x.qp[p], &rtr,
                               IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                   IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (rc != 0) {
            set_err("ibv_modify_qp(RTR)", rc);
            return false;
        }
        ibv_qp_attr rts;
        memset(&rts, 0, sizeof(rts));
        rts.qp_state = IBV_QPS_RTS;
        rts.timeout = 14;
        rts.retry_cnt = 7;
        rts.rnr_retry = 7;
        rts.sq_psn = 0;
        rts.max_rd_atomic = 1;
        rc = ibv_modify_qp(x.qp[p], &rts,
                           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                               IBV_QP_MAX_QP_RD_ATOMIC);
        if (rc != 0) {
            set_err("ibv_modify_qp(RTS)", rc);
            return false;
        }
        return true;
    }

    bool drain_cq(int h) {
        ibv_wc wc[32];
        int n = ibv_poll_cq(hca_[h].cq, 32, wc);
        if (n < 0) {
            set_err("ibv_poll_cq", errno);
            return false;
        }
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                char buf[256];
                snprintf(buf, sizeof(buf), "RDMA write to rank %u on HCA %d failed: %s (vendor_err 0x%x, seq %u)",
                         unsigned(wc[i].wr_id), h, ibv_wc_status_str(wc[i].status), wc[i].vendor_err, last_seq_);
                err_ = buf;
                return false;
            }
            hca_[h].outstanding[wc[i].wr_id] -= 1;
            hca_[h].writes_completed += 1;
            writes_completed_ += 1;
        }
        return true;
    }

    bool post_op(uint32_t seq, uint32_t nbytes) {
        if (nbytes == 0 || nbytes % 16 != 0 || nbytes > slot_bytes_) {
            err_ = "RoCE payload of " + std::to_string(nbytes) + " bytes (seq " + std::to_string(seq) +
                   "): expected a positive multiple of 16 up to the slot size";
            return false;
        }
        const uint32_t slot = seq & 1u;
        uint8_t *send = region_ + send_off_ + size_t(slot) * slot_bytes_;
        uint32_t seq_copy = seq;
        const uint32_t total_packs = nbytes / 16;
        const bool one = one_hca(nbytes, stripe_min_, n_hca_);          // (the kernel's rule)
        const int used = one ? single_hca(seq, n_hca_) : -1;
        single_ops_ += one ? 1 : 0;
        for (int p = 0; p < world_; p++) {
            if (p == rank_) {
                continue;
            }
            uint32_t pack_offset = 0;
            for (int h = 0; h < n_hca_; h++) {
                if (used >= 0 && h != used) {
                    continue;
                }
                Hca &x = hca_[h];
                // two work requests per stripe; keep the queue at most a quarter full
                while (x.outstanding[p] >= SEND_DEPTH / 4) {
                    if (!drain_cq(h)) {
                        return false;
                    }
                    // a peer that stopped acknowledging keeps the QP retrying for a long time: honour a stop
                    if (!running_.load(std::memory_order_relaxed)) {
                        err_ = "RoCE proxy stopped with " + std::to_string(x.outstanding[p]) +
                               " writes outstanding to rank " + std::to_string(p);
                        return false;
                    }
                }
                uint32_t stripe_packs = total_packs / uint32_t(n_hca_);
                if (uint32_t(h) < total_packs % uint32_t(n_hca_)) {
                    stripe_packs += 1;
                }
                if (used >= 0) {
                    stripe_packs = total_packs;          // the whole shard on this HCA
                }
                const uint32_t stripe_bytes = stripe_packs * 16;
                const uint64_t byte_offset = uint64_t(pack_offset) * 16;
                const uint64_t remote = peer_addr_[p];

                ibv_sge flag_sge;
                flag_sge.addr = uint64_t(reinterpret_cast<uintptr_t>(&seq_copy));
                flag_sge.length = 4;
                flag_sge.lkey = 0;
                ibv_send_wr flag_wr;
                memset(&flag_wr, 0, sizeof(flag_wr));
                flag_wr.wr_id = uint64_t(p);
                flag_wr.sg_list = &flag_sge;
                flag_wr.num_sge = 1;
                flag_wr.opcode = IBV_WR_RDMA_WRITE;
                flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
                flag_wr.wr.rdma.remote_addr =
                    remote + flag_off_ + ((uint64_t(rank_) * SLOTS + slot) * uint64_t(n_hca_) + uint64_t(h)) * FLAG_STRIDE;
                flag_wr.wr.rdma.rkey = peer_rkey_[h][p];

                ibv_send_wr data_wr;
                ibv_sge data_sge;
                ibv_send_wr *first = &flag_wr;
                if (stripe_bytes != 0) {
                    data_sge.addr = uint64_t(reinterpret_cast<uintptr_t>(send + byte_offset));
                    data_sge.length = stripe_bytes;
                    data_sge.lkey = x.mr->lkey;
                    memset(&data_wr, 0, sizeof(data_wr));
                    data_wr.wr_id = uint64_t(p);
                    data_wr.next = &flag_wr;
                    data_wr.sg_list = &data_sge;
                    data_wr.num_sge = 1;
                    data_wr.opcode = IBV_WR_RDMA_WRITE;
                    data_wr.wr.rdma.remote_addr =
                        remote + recv_off_ + (uint64_t(rank_) * SLOTS + slot) * slot_bytes_ + byte_offset;
                    data_wr.wr.rdma.rkey = peer_rkey_[h][p];
                    if (x.inline_cap != 0 && stripe_bytes <= x.inline_cap) {
                        // the CPU copies the staged bytes into the work request at post time. The
                        // doorbell's acquire load (main_loop) ordered the GPU's staging before this read.
                        data_wr.send_flags = IBV_SEND_INLINE;
                        inline_posts_ += 1;
                    }
                    first = &data_wr;
                }
                ibv_send_wr *bad = nullptr;
                int rc = ibv_post_send(x.qp[p], first, &bad);
                if (rc != 0) {
                    set_err("ibv_post_send", rc);
                    return false;
                }
                x.outstanding[p] += 1;
                x.bytes_posted += stripe_bytes;
                pack_offset += stripe_packs;
            }
        }
        ops_posted_ += 1;
        if (lazy_cq_) {
            return true;                                 // reaped by the idle loop (and the depth bound)
        }
        for (int h = 0; h < n_hca_; h++) {
            if (!drain_cq(h)) {
                return false;
            }
        }
        return true;
    }

    static uint64_t now_ns() {
        timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
    }

    static void *main_tramp(void *self) {
        static_cast<Proxy *>(self)->main_loop();
        return nullptr;
    }

    void main_loop() {
        if (cpu_ >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu_, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);   // best effort
        }
        volatile uint32_t *c = ctrl();
        uint64_t idle = 0;
        const timespec nap = {0, 20000};
        auto flags = [this](uint32_t p, uint32_t sl, uint32_t h) { return flag_at(p, sl, h); };
        while (running_.load(std::memory_order_relaxed)) {
            watch_.poll(c, flags);
            const uint32_t seq = __atomic_load_n(&c[CTRL_SEQ], __ATOMIC_ACQUIRE);
            if (seq == last_seq_) {
                idle++;
                if (idle % 64 == 0) {
                    for (int h = 0; h < n_hca_; h++) {
                        if (!drain_cq(h)) {
                            failed_.store(1);
                            return;
                        }
                    }
                }
                if (idle >= IDLE_SPINS) {
                    nanosleep(&nap, nullptr);
                }
                continue;
            }
            idle = 0;
            // The doorbell holds only the newest sequence. Our kernel for op N completes on the peers' payloads
            // alone, so op N+1 can ring before this thread saw op N. A peer cannot get more than one op ahead of
            // us, so at most SLOTS doorbells are pending and every send slot is intact: post each missed sequence
            // in order with its slot's byte count.
            uint32_t pending = seq - last_seq_;
            if (pending > uint32_t(SLOTS)) {
                char buf[160];
                snprintf(buf, sizeof(buf), "doorbell skipped %u ops (last posted %u, now %u)", pending, last_seq_, seq);
                err_ = buf;
                failed_.store(1);
                return;
            }
            const uint64_t seen = ptrace_.empty() ? 0 : now_ns();
            for (uint32_t s = last_seq_ + 1; pending > 0; s++, pending--) {
                const uint32_t nbytes = c[CTRL_SLOT_NBYTES + (s & 1u)];
                if (!post_op(s, nbytes)) {
                    failed_.store(1);
                    return;
                }
                last_seq_ = s;
                if (!ptrace_.empty()) {
                    const size_t i = size_t(s & trace_mask_) * 2;
                    ptrace_[i] = seen;
                    ptrace_[i + 1] = now_ns();
                }
            }
        }
    }

    int world_, rank_, n_hca_, tc_;
    uint8_t *region_;
    uint64_t region_bytes_, slot_bytes_;
    uint64_t recv_off_ = 0, flag_off_ = 0, send_off_ = 0, ctrl_off_ = 0;
    Hca hca_[MAX_HCAS];
    uint64_t peer_addr_[MAX_PEERS] = {};
    uint32_t peer_rkey_[MAX_HCAS][MAX_PEERS] = {};
    pthread_t thread_{};
    std::atomic<int> running_{0};
    std::atomic<int> failed_{0};
    bool started_ = false;
    int cpu_ = -1;
    uint32_t last_seq_ = 0;
    uint64_t ops_posted_ = 0;
    uint64_t writes_completed_ = 0;
    std::string err_;
    FailWatch watch_;
    uint32_t stripe_min_ = 0, inline_max_ = 0;
    bool lazy_cq_ = false;
    uint64_t inline_posts_ = 0, single_ops_ = 0;
    uint32_t trace_mask_ = 0;
    std::vector<uint64_t> ptrace_;
};

// -- pinned region -------------------------------------------------------------------------------------------------
static void check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("roce: ") + what + ": " + cudaGetErrorString(e));
    }
}

// Allocate the region in 64 KiB-aligned malloc'd pages registered with
// cudaHostRegister (and mlock'd, best effort; the NIC's registration pins them too) instead of one cudaHostAlloc
// block. On GB10 the GPU reaches cudaHostAlloc memory over an uncached path, registered malloc memory over the
// coherent, L2-cached one (cached host-memory load measurements). Same layout, same
// protocol; each allocation remembers its kind so free_pinned releases it the matching way.
static std::mutex g_pinned_mu;
static std::unordered_map<uintptr_t, std::pair<int, uint64_t>> g_pinned;

static void release_pinned(void *host, int hostreg, uint64_t nbytes) {
    if (hostreg) {
        cudaHostUnregister(host);
        munlock(host, nbytes);
        free(host);
    } else {
        cudaFreeHost(host);
    }
}

// (host address, device address) of a zeroed, pinned, device-mapped host allocation
std::vector<uint64_t> alloc_pinned(uint64_t nbytes, int hostreg) {
    void *host = nullptr;
    if (hostreg) {
        if (posix_memalign(&host, size_t(1) << 16, nbytes) != 0 || host == nullptr) {
            throw std::runtime_error("roce: posix_memalign of the pinned region failed");
        }
        memset(host, 0, nbytes);
        (void)mlock(host, nbytes);
        cudaError_t r = cudaHostRegister(host, nbytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (r != cudaSuccess) {
            munlock(host, nbytes);
            free(host);
            check_cuda(r, "cudaHostRegister");
        }
    } else {
        check_cuda(cudaHostAlloc(&host, nbytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
        memset(host, 0, nbytes);
    }
    void *dev = nullptr;
    cudaError_t e = cudaHostGetDevicePointer(&dev, host, 0);
    if (e != cudaSuccess) {
        release_pinned(host, hostreg, nbytes);
        check_cuda(e, "cudaHostGetDevicePointer");
    }
    {
        std::lock_guard<std::mutex> g(g_pinned_mu);
        g_pinned[reinterpret_cast<uintptr_t>(host)] = {hostreg ? 1 : 0, nbytes};
    }
    return {uint64_t(reinterpret_cast<uintptr_t>(host)), uint64_t(reinterpret_cast<uintptr_t>(dev))};
}

void free_pinned(uint64_t host) {
    int hostreg = 0;
    uint64_t nbytes = 0;
    {
        std::lock_guard<std::mutex> g(g_pinned_mu);
        auto it = g_pinned.find(uintptr_t(host));
        if (it != g_pinned.end()) {
            hostreg = it->second.first;
            nbytes = it->second.second;
            g_pinned.erase(it);
        }
    }
    release_pinned(reinterpret_cast<void *>(host), hostreg, nbytes);
}

std::vector<uint64_t> py_layout(int world, uint64_t slot_bytes) {
    uint64_t out[7];
    if (!layout(world, slot_bytes, out)) {
        throw std::invalid_argument("roce: unsupported geometry (2..16 ranks, slot_bytes a positive multiple of 4096)");
    }
    return std::vector<uint64_t>(out, out + 7);
}

// -- the launcher -----------------------------------------------------------------------------------------------------
class Launcher {
  public:
    Launcher(uint64_t recv_base, uint64_t flag_base, uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes,
             uint64_t epoch, uint64_t poison, int world, int rank, int n_hca, int flag_stride, int slots, int threads,
             uint32_t stripe_min, bool lean, uint64_t trace, uint32_t trace_entries, int copy_mode)
        : threads_(threads) {
        memset(&a_, 0, sizeof(a_));
        a_.recv_base = recv_base;
        a_.flag_base = flag_base;
        a_.send_base = send_base;
        a_.ctrl_base = ctrl_base;
        a_.slot_bytes = slot_bytes;
        a_.epoch = reinterpret_cast<uint32_t *>(epoch);
        a_.poison = reinterpret_cast<uint32_t *>(poison);
        a_.world = world;
        a_.rank = rank;
        a_.n_hca = n_hca;
        a_.flag_stride = flag_stride;
        a_.slots = slots;
        a_.stripe_min = stripe_min;
        a_.lean = lean ? 1u : 0u;
        a_.trace = reinterpret_cast<uint64_t *>(trace);
        a_.trace_mask = trace_entries == 0 ? 0u : trace_entries - 1;
        if (copy_mode != 1 && copy_mode != 2) throw std::invalid_argument("roce: copy mode must be 1 or 2");
        a_.copy2 = copy_mode == 2 ? 1u : 0u;
        if (trace != 0 && (trace_entries == 0 || (trace_entries & (trace_entries - 1)) != 0)) {
            throw std::invalid_argument("roce: trace entries must be a power of two");
        }
        if (threads < world * n_hca || threads % 32 != 0 || threads > 1024) {
            throw std::invalid_argument("roce: threads must be a multiple of 32, <= 1024 and >= world x HCAs");
        }
    }

    // recv [world x nbytes] <- every rank's [nbytes] in rank order, on the current stream
    void run(uint64_t in, uint64_t out, uint64_t nbytes, int grid, uint64_t stage_ctr, uint64_t tail_ctr,
             uint64_t timeout_ns, uint64_t stream) {
        if (nbytes == 0 || nbytes > a_.slot_bytes || grid < 1 || (grid & (grid - 1)) != 0) {
            throw std::invalid_argument("roce: gather of " + std::to_string(nbytes) + " bytes on " +
                                        std::to_string(grid) + " blocks (slot " + std::to_string(a_.slot_bytes) + ")");
        }
        GatherArgs a = a_;
        a.in = reinterpret_cast<const uint8_t *>(in);
        a.out = reinterpret_cast<uint8_t *>(out);
        a.nbytes = nbytes;
        a.stage_ctr = reinterpret_cast<uint32_t *>(stage_ctr);
        a.tail_ctr = reinterpret_cast<uint32_t *>(tail_ctr);
        a.timeout_ns = timeout_ns;
        check_cuda(launch_gather(a, grid, threads_, reinterpret_cast<cudaStream_t>(stream)), "gather launch");
    }

  private:
    GatherArgs a_;
    int threads_;
};

// -- tests on one GPU: a thread that plays every peer ------------------------------------------------------------------
// It polls the doorbell like the proxy and, for each sequence, "delivers" every peer's shard as this rank's staged
// shard XOR ``xor_key`` into recv[peer][slot], then the per-HCA flags (release stores after a full fence: the NIC's
// payload-before-flag order). ``stall_after`` >= 0: stop delivering after that many ops (timeout tests);
// ``late``: resume delivering once the kernel's failure word is up (a peer that arrives after the
// timeout: the watch must call it late, not "delivered but not seen").
class Loop {
  public:
    Loop(uint64_t host, uint64_t recv_off, uint64_t flag_off, uint64_t send_off, uint64_t ctrl_off, uint64_t slot_bytes,
         int world, int rank, int n_hca, int flag_stride, int slots, int xor_key, int64_t stall_after, bool late,
         uint32_t stripe_min)
        : region_(reinterpret_cast<uint8_t *>(host)), recv_off_(recv_off), flag_off_(flag_off), send_off_(send_off),
          ctrl_off_(ctrl_off), slot_bytes_(slot_bytes), world_(world), rank_(rank), n_hca_(n_hca),
          flag_stride_(flag_stride), slots_(slots), xor_(uint8_t(xor_key)), stall_after_(stall_after), late_(late),
          stripe_min_(stripe_min) {
        last_ = ctrl()[CTRL_SEQ];
        running_.store(1);
        int rc = pthread_create(&thread_, nullptr, &Loop::tramp, this);
        if (rc != 0) {
            running_.store(0);
            throw std::runtime_error(std::string("roce loop: pthread_create: ") + strerror(rc));
        }
    }

    ~Loop() { destroy(); }

    void destroy() {
        if (running_.exchange(0)) {
            pthread_join(thread_, nullptr);
        }
    }

    int64_t delivered() const { return delivered_.load(); }

    std::vector<uint64_t> watch() const {
        uint64_t out[7];
        watch_.read(out);
        return std::vector<uint64_t>(out, out + 7);
    }

  private:
    volatile uint32_t *ctrl() const { return reinterpret_cast<volatile uint32_t *>(region_ + ctrl_off_); }

    const volatile uint32_t *flag_at(uint32_t peer, uint32_t slot, uint32_t hca) const {
        if (peer >= uint32_t(world_) || slot >= uint32_t(slots_) || hca >= uint32_t(n_hca_)) {
            return nullptr;
        }
        return reinterpret_cast<const volatile uint32_t *>(
            region_ + flag_off_ + ((uint64_t(peer) * slots_ + slot) * n_hca_ + hca) * flag_stride_);
    }

    static void *tramp(void *self) {
        static_cast<Loop *>(self)->run();
        return nullptr;
    }

    void run() {
        volatile uint32_t *c = ctrl();
        auto flags = [this](uint32_t p, uint32_t sl, uint32_t h) { return flag_at(p, sl, h); };
        while (running_.load(std::memory_order_relaxed)) {
            watch_.poll(c, flags);
            const uint32_t seq = __atomic_load_n(&c[CTRL_SEQ], __ATOMIC_ACQUIRE);
            if (seq == last_) {
                continue;
            }
            for (uint32_t s = last_ + 1; s != seq + 1; s++) {
                if (stall_after_ >= 0 && delivered_.load() >= stall_after_ &&
                    !(late_ && watch_.state.load(std::memory_order_acquire) != FailWatch::IDLE)) {
                    break;                          // stalled (an op not delivered stays pending)
                }
                const uint32_t slot = s & 1u;
                const uint32_t nbytes = c[CTRL_SLOT_NBYTES + slot];
                const uint8_t *send = region_ + send_off_ + size_t(slot) * slot_bytes_;
                for (int p = 0; p < world_; p++) {
                    if (p == rank_) {
                        continue;
                    }
                    uint8_t *dst = region_ + recv_off_ + (uint64_t(p) * slots_ + slot) * slot_bytes_;
                    for (uint32_t i = 0; i < nbytes; i++) {
                        dst[i] = uint8_t(send[i] ^ xor_);
                    }
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    // a single-HCA op sets only that HCA's flag (a kernel that waited for another
                    // HCA's would time out: the GPU tests' check of the kernel's rule)
                    const int used = one_hca(nbytes, stripe_min_, n_hca_) ? single_hca(s, n_hca_) : -1;
                    for (int h = 0; h < n_hca_; h++) {
                        if (used >= 0 && h != used) {
                            continue;
                        }
                        uint32_t *flag = reinterpret_cast<uint32_t *>(
                            region_ + flag_off_ + ((uint64_t(p) * slots_ + slot) * n_hca_ + h) * flag_stride_);
                        __atomic_store_n(flag, s, __ATOMIC_RELEASE);
                    }
                }
                delivered_.fetch_add(1);
                last_ = s;
            }
        }
    }

    uint8_t *region_;
    uint64_t recv_off_, flag_off_, send_off_, ctrl_off_, slot_bytes_;
    int world_, rank_, n_hca_, flag_stride_, slots_;
    uint8_t xor_;
    int64_t stall_after_;
    bool late_;
    uint32_t stripe_min_;
    uint32_t last_ = 0;
    FailWatch watch_;
    pthread_t thread_{};
    std::atomic<int> running_{0};
    std::atomic<int64_t> delivered_{0};
};

// CUDA's id of the capture the current stream is in, 0 when it is not capturing
uint64_t capture_id(uint64_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    check_cuda(cudaStreamGetCaptureInfo(reinterpret_cast<cudaStream_t>(stream), &status, &id), "capture info");
    return status == cudaStreamCaptureStatusActive ? uint64_t(id) : 0;
}

}  // namespace tfroce
