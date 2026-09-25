#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define CERM_EXPORT __declspec(dllexport)
#else
#define CERM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" CERM_EXPORT int cerm_training_core_abi_version() { return 2; }

static int validate_state_layout(
    const int32_t* offsets, int width, int total_states) {
    if (!offsets || width <= 0 || total_states <= 0) return -1;
    if (offsets[0] != 0 || offsets[width] != total_states) return -2;
    for (int slot = 0; slot < width; ++slot) {
        if (offsets[slot] < 0 || offsets[slot + 1] <= offsets[slot] ||
            offsets[slot + 1] > total_states) {
            return -3;
        }
    }
    return 0;
}

extern "C" CERM_EXPORT int cerm_state_histogram_mt_u8(
    const uint8_t* states, int n, int width,
    const int32_t* offsets, int total_states,
    const double* values, int n_values,
    const double* sample_weight, int n_threads,
    double* out_mass, double* out_sums) {
    if (!states || !out_mass || n < 0 || n_values < 0) return -10;
    if (n_values > 0 && (!values || !out_sums)) return -11;
    const int layout_status = validate_state_layout(offsets, width, total_states);
    if (layout_status != 0) return -20 + layout_status;

    std::fill(out_mass, out_mass + total_states, 0.0);
    if (n_values > 0) {
        std::fill(
            out_sums,
            out_sums + static_cast<size_t>(total_states) * n_values,
            0.0);
    }

    n_threads = std::max(1, std::min(n_threads, width));
    auto work = [&](int slot_begin, int slot_end) -> int {
        for (int slot = slot_begin; slot < slot_end; ++slot) {
            const int base = offsets[slot];
            const int card = offsets[slot + 1] - base;
            for (int i = 0; i < n; ++i) {
                const int code = static_cast<int>(
                    states[static_cast<size_t>(i) * width + slot]);
                if (code < 0 || code >= card) return -30;
                const int row = base + code;
                const double weight = sample_weight ? sample_weight[i] : 1.0;
                out_mass[row] += weight;
                if (n_values > 0) {
                    const double* src = values + static_cast<size_t>(i) * n_values;
                    double* dst = out_sums + static_cast<size_t>(row) * n_values;
                    for (int q = 0; q < n_values; ++q) {
                        dst[q] += weight * src[q];
                    }
                }
            }
        }
        return 0;
    };

    if (n_threads == 1) return work(0, width);
    std::vector<std::thread> workers;
    std::vector<int> status(n_threads, 0);
    workers.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        const int begin = static_cast<int>(
            static_cast<int64_t>(width) * thread / n_threads);
        const int end = static_cast<int>(
            static_cast<int64_t>(width) * (thread + 1) / n_threads);
        workers.emplace_back([&, thread, begin, end]() {
            status[thread] = work(begin, end);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const int value : status) {
        if (value != 0) return value;
    }
    return 0;
}

extern "C" CERM_EXPORT int cerm_selected_pair_histogram_mt_u8(
    const uint8_t* states, int n, int width,
    const int32_t* state_offsets, int total_states,
    const int32_t* pair_slots, int n_pairs,
    const int32_t* pair_offsets, const int32_t* right_cards,
    int total_pair_states,
    const double* values, int n_values,
    const double* sample_weight, int n_threads,
    double* out_mass, double* out_sums) {
    if (!states || !pair_slots || !pair_offsets || !right_cards || !out_mass ||
        n < 0 || n_pairs < 0 || total_pair_states < 0 || n_values < 0) {
        return -40;
    }
    if (n_values > 0 && (!values || !out_sums)) return -41;
    const int layout_status =
        validate_state_layout(state_offsets, width, total_states);
    if (layout_status != 0) return -50 + layout_status;
    if (pair_offsets[0] != 0 || pair_offsets[n_pairs] != total_pair_states) {
        return -54;
    }

    std::fill(out_mass, out_mass + total_pair_states, 0.0);
    if (n_values > 0) {
        std::fill(
            out_sums,
            out_sums + static_cast<size_t>(total_pair_states) * n_values,
            0.0);
    }
    if (n_pairs == 0) return 0;

    for (int pair_index = 0; pair_index < n_pairs; ++pair_index) {
        const int left = pair_slots[2 * pair_index];
        const int right = pair_slots[2 * pair_index + 1];
        if (left < 0 || left >= width || right < 0 || right >= width) return -60;
        const int left_card = state_offsets[left + 1] - state_offsets[left];
        const int right_card = state_offsets[right + 1] - state_offsets[right];
        if (right_cards[pair_index] != right_card) return -61;
        if (pair_offsets[pair_index + 1] - pair_offsets[pair_index] !=
            left_card * right_card) {
            return -62;
        }
    }

    n_threads = std::max(1, std::min(n_threads, n_pairs));
    auto work = [&](int pair_begin, int pair_end) -> int {
        for (int pair_index = pair_begin; pair_index < pair_end; ++pair_index) {
            const int left = pair_slots[2 * pair_index];
            const int right = pair_slots[2 * pair_index + 1];
            const int left_card = state_offsets[left + 1] - state_offsets[left];
            const int right_card = right_cards[pair_index];
            const int base = pair_offsets[pair_index];
            for (int i = 0; i < n; ++i) {
                const uint8_t* row = states + static_cast<size_t>(i) * width;
                const int left_code = static_cast<int>(row[left]);
                const int right_code = static_cast<int>(row[right]);
                if (left_code >= left_card || right_code >= right_card) return -70;
                const int state = base + left_code * right_card + right_code;
                const double weight = sample_weight ? sample_weight[i] : 1.0;
                out_mass[state] += weight;
                if (n_values > 0) {
                    const double* src = values + static_cast<size_t>(i) * n_values;
                    double* dst = out_sums + static_cast<size_t>(state) * n_values;
                    for (int q = 0; q < n_values; ++q) {
                        dst[q] += weight * src[q];
                    }
                }
            }
        }
        return 0;
    };

    if (n_threads == 1) return work(0, n_pairs);
    std::vector<std::thread> workers;
    std::vector<int> status(n_threads, 0);
    workers.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        const int begin = static_cast<int>(
            static_cast<int64_t>(n_pairs) * thread / n_threads);
        const int end = static_cast<int>(
            static_cast<int64_t>(n_pairs) * (thread + 1) / n_threads);
        workers.emplace_back([&, thread, begin, end]() {
            status[thread] = work(begin, end);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const int value : status) {
        if (value != 0) return value;
    }
    return 0;
}


extern "C" CERM_EXPORT int cerm_selected_triad_histogram_mt_u8(
    const uint8_t* states, int n, int width,
    const int32_t* triad_slots, int n_triads,
    const double* values, int n_values,
    int n_threads, double* out_sums) {
    if (!states || !triad_slots || !out_sums ||
        n < 0 || width <= 0 || n_triads < 0 || n_values < 0) {
        return -300;
    }
    if (n_values > 0 && !values) return -301;
    if (n_triads == 0) return 0;

    for (int triad_index = 0; triad_index < n_triads; ++triad_index) {
        const int a = triad_slots[3 * triad_index];
        const int b = triad_slots[3 * triad_index + 1];
        const int c = triad_slots[3 * triad_index + 2];
        if (a < 0 || a >= width || b < 0 || b >= width ||
            c < 0 || c >= width || a == b || a == c || b == c) {
            return -310;
        }
    }

    const size_t total =
        static_cast<size_t>(n_triads) * 64 * static_cast<size_t>(n_values);
    if (n_values > 0) std::fill(out_sums, out_sums + total, 0.0);

    n_threads = std::max(1, std::min(n_threads, n_triads));
    auto work = [&](int triad_begin, int triad_end) -> int {
        for (int triad_index = triad_begin;
             triad_index < triad_end;
             ++triad_index) {
            const int a = triad_slots[3 * triad_index];
            const int b = triad_slots[3 * triad_index + 1];
            const int c = triad_slots[3 * triad_index + 2];
            double* table =
                out_sums +
                static_cast<size_t>(triad_index) * 64 * n_values;
            for (int i = 0; i < n; ++i) {
                const uint8_t* row =
                    states + static_cast<size_t>(i) * width;
                const int sa = static_cast<int>(row[a]);
                const int sb = static_cast<int>(row[b]);
                const int sc = static_cast<int>(row[c]);
                if (sa >= 4 || sb >= 4 || sc >= 4) return -320;
                const int state = (sa << 4) | (sb << 2) | sc;
                if (n_values > 0) {
                    const double* src =
                        values + static_cast<size_t>(i) * n_values;
                    double* dst =
                        table + static_cast<size_t>(state) * n_values;
                    for (int q = 0; q < n_values; ++q) {
                        dst[q] += src[q];
                    }
                }
            }
        }
        return 0;
    };

    if (n_threads == 1) return work(0, n_triads);
    std::vector<std::thread> workers;
    std::vector<int> status(n_threads, 0);
    workers.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        const int begin = static_cast<int>(
            static_cast<int64_t>(n_triads) * thread / n_threads);
        const int end = static_cast<int>(
            static_cast<int64_t>(n_triads) * (thread + 1) / n_threads);
        workers.emplace_back([&, thread, begin, end]() {
            status[thread] = work(begin, end);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const int value : status) {
        if (value != 0) return value;
    }
    return 0;
}


extern "C" CERM_EXPORT int cerm_selected_triad_fused_stage1_mt_u8(
    const uint8_t* states, int n, int width,
    const int32_t* triad_slots, int n_triads,
    const double* values, int n_values,
    double critical, double guard_rel,
    int n_threads,
    uint8_t* out_flags, double* out_cheap, double* out_sums) {
    if (!states || !triad_slots || !out_flags || !out_cheap ||
        n < 0 || width <= 0 || n_triads < 0 || n_values < 2) {
        return -350;
    }
    if (!values) return -351;
    if (n_triads == 0) return 0;

    for (int triad_index = 0; triad_index < n_triads; ++triad_index) {
        const int a = triad_slots[3 * triad_index];
        const int b = triad_slots[3 * triad_index + 1];
        const int c = triad_slots[3 * triad_index + 2];
        if (a < 0 || a >= width || b < 0 || b >= width ||
            c < 0 || c >= width || a == b || a == c || b == c) {
            return -360;
        }
    }

    std::fill(out_flags, out_flags + n_triads, static_cast<uint8_t>(0));
    std::fill(out_cheap, out_cheap + n_triads, 0.0);

    n_threads = std::max(1, std::min(n_threads, n_triads));
    auto work = [&](int triad_begin, int triad_end) -> int {
        for (int triad_index = triad_begin;
             triad_index < triad_end;
             ++triad_index) {
            const int a = triad_slots[3 * triad_index];
            const int b = triad_slots[3 * triad_index + 1];
            const int c = triad_slots[3 * triad_index + 2];

            double local_G[64] = {0.0};
            double local_H[64] = {0.0};

            for (int i = 0; i < n; ++i) {
                const uint8_t* row =
                    states + static_cast<size_t>(i) * width;
                const int sa = static_cast<int>(row[a]);
                const int sb = static_cast<int>(row[b]);
                const int sc = static_cast<int>(row[c]);
                if (sa >= 4 || sb >= 4 || sc >= 4) return -370;
                const int state = (sa << 4) | (sb << 2) | sc;
                const double* src =
                    values + static_cast<size_t>(i) * n_values;
                local_G[state] += src[0];
                local_H[state] += src[1];
            }

            bool full_support = true;
            double full_gain = 0.0;
            double pair_G_ab[16] = {0.0}, pair_H_ab[16] = {0.0};
            double pair_G_ac[16] = {0.0}, pair_H_ac[16] = {0.0};
            double pair_G_bc[16] = {0.0}, pair_H_bc[16] = {0.0};

            for (int s = 0; s < 64; ++s) {
                const double g_val = local_G[s];
                const double h_val = local_H[s];
                if (h_val <= 0.0) {
                    full_support = false;
                } else {
                    full_gain += 0.5 * (g_val * g_val / h_val);
                }
                const int sa = (s >> 4) & 3;
                const int sb = (s >> 2) & 3;
                const int sc = s & 3;

                const int ab = (sa << 2) | sb;
                pair_G_ab[ab] += g_val;
                pair_H_ab[ab] += h_val;

                const int ac = (sa << 2) | sc;
                pair_G_ac[ac] += g_val;
                pair_H_ac[ac] += h_val;

                const int bc = (sb << 2) | sc;
                pair_G_bc[bc] += g_val;
                pair_H_bc[bc] += h_val;
            }

            double pair_gain_ab = 0.0, pair_gain_ac = 0.0, pair_gain_bc = 0.0;
            for (int k = 0; k < 16; ++k) {
                if (pair_H_ab[k] > 0.0) {
                    pair_gain_ab += 0.5 * (pair_G_ab[k] * pair_G_ab[k] / pair_H_ab[k]);
                }
                if (pair_H_ac[k] > 0.0) {
                    pair_gain_ac += 0.5 * (pair_G_ac[k] * pair_G_ac[k] / pair_H_ac[k]);
                }
                if (pair_H_bc[k] > 0.0) {
                    pair_gain_bc += 0.5 * (pair_G_bc[k] * pair_G_bc[k] / pair_H_bc[k]);
                }
            }

            const double max_pair_gain =
                std::max({pair_gain_ab, pair_gain_ac, pair_gain_bc});
            const double cheap = std::max(0.0, full_gain - max_pair_gain);

            const double threshold = 0.5 * critical;
            const double scale = std::max(
                {1.0,
                 std::abs(cheap),
                 std::abs(threshold),
                 std::abs(full_gain),
                 pair_gain_ab,
                 pair_gain_ac,
                 pair_gain_bc});
            const double guard = guard_rel * scale;
            const bool ambiguous = (std::abs(cheap - threshold) <= guard);
            const bool clear_reject =
                full_support && (cheap < threshold - guard) && !ambiguous;

            uint8_t flags = 0;
            if (clear_reject) flags |= 1;
            if (ambiguous) flags |= 2;
            if (full_support) flags |= 4;

            out_flags[triad_index] = flags;
            out_cheap[triad_index] = cheap;

            if (!clear_reject && out_sums) {
                double* dst =
                    out_sums + static_cast<size_t>(triad_index) * 64 * n_values;
                for (int s = 0; s < 64; ++s) {
                    dst[s * n_values] = local_G[s];
                    dst[s * n_values + 1] = local_H[s];
                    for (int q = 2; q < n_values; ++q) {
                        dst[s * n_values + q] = 0.0;
                    }
                }
            }
        }
        return 0;
    };

    if (n_threads == 1) return work(0, n_triads);
    std::vector<std::thread> workers;
    std::vector<int> status(n_threads, 0);
    workers.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        const int begin = static_cast<int>(
            static_cast<int64_t>(n_triads) * thread / n_threads);
        const int end = static_cast<int>(
            static_cast<int64_t>(n_triads) * (thread + 1) / n_threads);
        workers.emplace_back([&, thread, begin, end]() {
            status[thread] = work(begin, end);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const int value : status) {
        if (value != 0) return value;
    }
    return 0;
}


extern "C" CERM_EXPORT int cerm_selected_quotient_pair_histogram_mt_u8(
    const uint8_t* states, int n, int width,
    const int32_t* state_offsets, int total_states,
    const uint8_t* coarse_maps, const int32_t* coarse_cards,
    const int32_t* pair_slots, int n_pairs,
    const int32_t* pair_offsets, int total_pair_states,
    const double* values, int n_values,
    const double* sample_weight, int n_threads,
    double* out_mass, double* out_sums) {
    if (!states || !coarse_maps || !coarse_cards || !pair_slots ||
        !pair_offsets || !out_mass || n < 0 || n_pairs < 0 ||
        total_pair_states < 0 || n_values < 0) {
        return -140;
    }
    if (n_values > 0 && (!values || !out_sums)) return -141;
    const int layout_status =
        validate_state_layout(state_offsets, width, total_states);
    if (layout_status != 0) return -150 + layout_status;
    if (pair_offsets[0] != 0 || pair_offsets[n_pairs] != total_pair_states) {
        return -154;
    }

    for (int slot = 0; slot < width; ++slot) {
        const int fine_card = state_offsets[slot + 1] - state_offsets[slot];
        const int coarse_card = coarse_cards[slot];
        if (coarse_card <= 0 || coarse_card > fine_card) return -155;
        const int base = state_offsets[slot];
        for (int state = 0; state < fine_card; ++state) {
            if (static_cast<int>(coarse_maps[base + state]) >= coarse_card) {
                return -156;
            }
        }
    }

    for (int pair_index = 0; pair_index < n_pairs; ++pair_index) {
        const int left = pair_slots[2 * pair_index];
        const int right = pair_slots[2 * pair_index + 1];
        if (left < 0 || left >= width || right < 0 || right >= width ||
            left == right) {
            return -160;
        }
        const int left_card = state_offsets[left + 1] - state_offsets[left];
        const int right_card = state_offsets[right + 1] - state_offsets[right];
        const int left_coarse = coarse_cards[left];
        const int right_coarse = coarse_cards[right];
        const int expected =
            left_coarse * right_card + right_coarse * left_card;
        if (pair_offsets[pair_index + 1] - pair_offsets[pair_index] != expected) {
            return -161;
        }
    }

    std::fill(out_mass, out_mass + total_pair_states, 0.0);
    if (n_values > 0) {
        std::fill(
            out_sums,
            out_sums + static_cast<size_t>(total_pair_states) * n_values,
            0.0);
    }
    if (n_pairs == 0) return 0;

    n_threads = std::max(1, std::min(n_threads, n_pairs));
    auto work = [&](int pair_begin, int pair_end) -> int {
        for (int pair_index = pair_begin; pair_index < pair_end; ++pair_index) {
            const int left = pair_slots[2 * pair_index];
            const int right = pair_slots[2 * pair_index + 1];
            const int left_card = state_offsets[left + 1] - state_offsets[left];
            const int right_card = state_offsets[right + 1] - state_offsets[right];
            const int left_coarse = coarse_cards[left];
            const int right_coarse = coarse_cards[right];
            const int left_map_base = state_offsets[left];
            const int right_map_base = state_offsets[right];
            const int fine_width = left_card * right_card;

            std::vector<double> fine_mass(fine_width, 0.0);
            std::vector<double> fine_sums(
                static_cast<size_t>(fine_width) * n_values, 0.0);

            for (int i = 0; i < n; ++i) {
                const uint8_t* row = states + static_cast<size_t>(i) * width;
                const int left_code = static_cast<int>(row[left]);
                const int right_code = static_cast<int>(row[right]);
                if (left_code >= left_card || right_code >= right_card) return -170;
                const int fine_state = left_code * right_card + right_code;
                const double weight = sample_weight ? sample_weight[i] : 1.0;
                fine_mass[fine_state] += weight;
                if (n_values > 0) {
                    const double* src =
                        values + static_cast<size_t>(i) * n_values;
                    double* dst =
                        fine_sums.data() + static_cast<size_t>(fine_state) * n_values;
                    for (int q = 0; q < n_values; ++q) {
                        dst[q] += weight * src[q];
                    }
                }
            }

            const int base = pair_offsets[pair_index];
            const int reverse_base = base + left_coarse * right_card;
            for (int left_state = 0; left_state < left_card; ++left_state) {
                const int left_group =
                    static_cast<int>(coarse_maps[left_map_base + left_state]);
                for (int right_state = 0; right_state < right_card; ++right_state) {
                    const int right_group =
                        static_cast<int>(coarse_maps[right_map_base + right_state]);
                    const int fine_state = left_state * right_card + right_state;
                    const int forward =
                        base + left_group * right_card + right_state;
                    const int reverse =
                        reverse_base + right_group * left_card + left_state;
                    const double mass = fine_mass[fine_state];
                    out_mass[forward] += mass;
                    out_mass[reverse] += mass;
                    if (n_values > 0) {
                        const double* src = fine_sums.data() +
                            static_cast<size_t>(fine_state) * n_values;
                        double* dst_forward = out_sums +
                            static_cast<size_t>(forward) * n_values;
                        double* dst_reverse = out_sums +
                            static_cast<size_t>(reverse) * n_values;
                        for (int q = 0; q < n_values; ++q) {
                            dst_forward[q] += src[q];
                            dst_reverse[q] += src[q];
                        }
                    }
                }
            }
        }
        return 0;
    };

    if (n_threads == 1) return work(0, n_pairs);
    std::vector<std::thread> workers;
    std::vector<int> status(n_threads, 0);
    workers.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        const int begin = static_cast<int>(
            static_cast<int64_t>(n_pairs) * thread / n_threads);
        const int end = static_cast<int>(
            static_cast<int64_t>(n_pairs) * (thread + 1) / n_threads);
        workers.emplace_back([&, thread, begin, end]() {
            status[thread] = work(begin, end);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const int value : status) {
        if (value != 0) return value;
    }
    return 0;
}


static void evaluate_quotient_direction(
    const double* table, int gate_card, int target_card,
    double gain_l2, double min_hessian,
    double min_support_a, double min_support_b, double min_support_full,
    double* out_full, double* out_a, double* out_b, uint8_t* out_flags) {
    for (int gate_state = 0; gate_state < gate_card; ++gate_state) {
        double mass_a = 0.0, grad_a = 0.0, hess_a = 0.0;
        double mass_b = 0.0, grad_b = 0.0, hess_b = 0.0;
        double child_a = 0.0, child_b = 0.0, child_full = 0.0;
        int active_a = 0, active_b = 0, active_full = 0;

        for (int target_state = 0; target_state < target_card; ++target_state) {
            const double* cell =
                table + static_cast<size_t>(
                    gate_state * target_card + target_state) * 6;
            mass_a += cell[0];
            grad_a += cell[1];
            hess_a += cell[2];
            mass_b += cell[3];
            grad_b += cell[4];
            hess_b += cell[5];

            if (cell[2] > 1e-12) {
                child_a += cell[1] * cell[1] / (cell[2] + gain_l2);
                ++active_a;
            }
            if (cell[5] > 1e-12) {
                child_b += cell[4] * cell[4] / (cell[5] + gain_l2);
                ++active_b;
            }
            const double grad_full = cell[1] + cell[4];
            const double hess_full = cell[2] + cell[5];
            if (hess_full > 1e-12) {
                child_full +=
                    grad_full * grad_full / (hess_full + gain_l2);
                ++active_full;
            }
        }

        uint8_t flags = 0;
        if (mass_a >= min_support_a &&
            hess_a >= min_hessian &&
            active_a > 1) {
            out_a[gate_state] =
                0.5 * std::max(
                    0.0,
                    child_a - grad_a * grad_a / (hess_a + gain_l2));
            flags |= 1;
        } else {
            out_a[gate_state] = 0.0;
        }

        if (mass_b >= min_support_b &&
            hess_b >= min_hessian &&
            active_b > 1) {
            out_b[gate_state] =
                0.5 * std::max(
                    0.0,
                    child_b - grad_b * grad_b / (hess_b + gain_l2));
            flags |= 2;
        } else {
            out_b[gate_state] = 0.0;
        }

        const double mass_full = mass_a + mass_b;
        const double grad_full = grad_a + grad_b;
        const double hess_full = hess_a + hess_b;
        if (mass_full >= min_support_full &&
            hess_full >= min_hessian &&
            active_full > 1) {
            out_full[gate_state] =
                0.5 * std::max(
                    0.0,
                    child_full -
                        grad_full * grad_full / (hess_full + gain_l2));
            flags |= 4;
        } else {
            out_full[gate_state] = 0.0;
        }

        out_flags[gate_state] = flags;
    }
}

extern "C" CERM_EXPORT int cerm_selected_quotient_pair_gain_mt_u8(
    const uint8_t* states, int n, int width,
    const int32_t* state_offsets, int total_states,
    const uint8_t* coarse_maps, const int32_t* coarse_cards,
    const int32_t* pair_slots, int n_pairs,
    const int32_t* candidate_offsets, int total_candidates,
    const double* fold_values6, int n_threads,
    double gain_l2, double min_hessian,
    double min_support_a, double min_support_b, double min_support_full,
    double* out_full, double* out_a, double* out_b, uint8_t* out_flags) {
    if (!states || !coarse_maps || !coarse_cards || !pair_slots ||
        !candidate_offsets || !fold_values6 ||
        !out_full || !out_a || !out_b || !out_flags ||
        n < 0 || n_pairs < 0 || total_candidates < 0 ||
        gain_l2 < 0.0 || min_hessian < 0.0) {
        return -240;
    }
    const int layout_status =
        validate_state_layout(state_offsets, width, total_states);
    if (layout_status != 0) return -250 + layout_status;
    if (candidate_offsets[0] != 0 ||
        candidate_offsets[n_pairs] != total_candidates) {
        return -254;
    }

    for (int slot = 0; slot < width; ++slot) {
        const int fine_card = state_offsets[slot + 1] - state_offsets[slot];
        const int coarse_card = coarse_cards[slot];
        if (coarse_card <= 0 || coarse_card > fine_card) return -255;
        const int base = state_offsets[slot];
        for (int state = 0; state < fine_card; ++state) {
            if (static_cast<int>(coarse_maps[base + state]) >= coarse_card) {
                return -256;
            }
        }
    }

    for (int pair_index = 0; pair_index < n_pairs; ++pair_index) {
        const int left = pair_slots[2 * pair_index];
        const int right = pair_slots[2 * pair_index + 1];
        if (left < 0 || left >= width || right < 0 || right >= width ||
            left == right) {
            return -260;
        }
        const int expected = coarse_cards[left] + coarse_cards[right];
        if (candidate_offsets[pair_index + 1] -
                candidate_offsets[pair_index] != expected) {
            return -261;
        }
    }

    std::fill(out_full, out_full + total_candidates, 0.0);
    std::fill(out_a, out_a + total_candidates, 0.0);
    std::fill(out_b, out_b + total_candidates, 0.0);
    std::fill(out_flags, out_flags + total_candidates, static_cast<uint8_t>(0));
    if (n_pairs == 0) return 0;

    n_threads = std::max(1, std::min(n_threads, n_pairs));
    auto work = [&](int pair_begin, int pair_end) -> int {
        for (int pair_index = pair_begin; pair_index < pair_end; ++pair_index) {
            const int left = pair_slots[2 * pair_index];
            const int right = pair_slots[2 * pair_index + 1];
            const int left_card = state_offsets[left + 1] - state_offsets[left];
            const int right_card = state_offsets[right + 1] - state_offsets[right];
            const int left_coarse = coarse_cards[left];
            const int right_coarse = coarse_cards[right];
            const int left_map_base = state_offsets[left];
            const int right_map_base = state_offsets[right];
            const int fine_width = left_card * right_card;

            std::vector<double> fine(
                static_cast<size_t>(fine_width) * 6, 0.0);
            for (int i = 0; i < n; ++i) {
                const uint8_t* row =
                    states + static_cast<size_t>(i) * width;
                const int left_state = static_cast<int>(row[left]);
                const int right_state = static_cast<int>(row[right]);
                if (left_state >= left_card || right_state >= right_card) {
                    return -270;
                }
                const int fine_state =
                    left_state * right_card + right_state;
                double* dst =
                    fine.data() + static_cast<size_t>(fine_state) * 6;
                const double* src =
                    fold_values6 + static_cast<size_t>(i) * 6;
                for (int q = 0; q < 6; ++q) {
                    dst[q] += src[q];
                }
            }

            const int candidate_base = candidate_offsets[pair_index];
            std::vector<double> direction(
                static_cast<size_t>(
                    std::max(
                        left_coarse * right_card,
                        right_coarse * left_card)) * 6,
                0.0);

            const int forward_cells = left_coarse * right_card;
            std::fill(
                direction.begin(),
                direction.begin() + static_cast<size_t>(forward_cells) * 6,
                0.0);
            for (int left_state = 0; left_state < left_card; ++left_state) {
                const int group =
                    static_cast<int>(
                        coarse_maps[left_map_base + left_state]);
                for (int right_state = 0;
                     right_state < right_card;
                     ++right_state) {
                    const double* src =
                        fine.data() +
                        static_cast<size_t>(
                            left_state * right_card + right_state) * 6;
                    double* dst =
                        direction.data() +
                        static_cast<size_t>(
                            group * right_card + right_state) * 6;
                    for (int q = 0; q < 6; ++q) {
                        dst[q] += src[q];
                    }
                }
            }
            evaluate_quotient_direction(
                direction.data(),
                left_coarse,
                right_card,
                gain_l2,
                min_hessian,
                min_support_a,
                min_support_b,
                min_support_full,
                out_full + candidate_base,
                out_a + candidate_base,
                out_b + candidate_base,
                out_flags + candidate_base);

            const int reverse_cells = right_coarse * left_card;
            std::fill(
                direction.begin(),
                direction.begin() + static_cast<size_t>(reverse_cells) * 6,
                0.0);
            for (int right_state = 0; right_state < right_card; ++right_state) {
                const int group =
                    static_cast<int>(
                        coarse_maps[right_map_base + right_state]);
                for (int left_state = 0;
                     left_state < left_card;
                     ++left_state) {
                    const double* src =
                        fine.data() +
                        static_cast<size_t>(
                            left_state * right_card + right_state) * 6;
                    double* dst =
                        direction.data() +
                        static_cast<size_t>(
                            group * left_card + left_state) * 6;
                    for (int q = 0; q < 6; ++q) {
                        dst[q] += src[q];
                    }
                }
            }
            evaluate_quotient_direction(
                direction.data(),
                right_coarse,
                left_card,
                gain_l2,
                min_hessian,
                min_support_a,
                min_support_b,
                min_support_full,
                out_full + candidate_base + left_coarse,
                out_a + candidate_base + left_coarse,
                out_b + candidate_base + left_coarse,
                out_flags + candidate_base + left_coarse);
        }
        return 0;
    };

    if (n_threads == 1) return work(0, n_pairs);
    std::vector<std::thread> workers;
    std::vector<int> status(n_threads, 0);
    workers.reserve(n_threads);
    for (int thread = 0; thread < n_threads; ++thread) {
        const int begin = static_cast<int>(
            static_cast<int64_t>(n_pairs) * thread / n_threads);
        const int end = static_cast<int>(
            static_cast<int64_t>(n_pairs) * (thread + 1) / n_threads);
        workers.emplace_back([&, thread, begin, end]() {
            status[thread] = work(begin, end);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const int value : status) {
        if (value != 0) return value;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Exact CSR logistic solver (L2R_LR)
//
// TRON control flow and logistic objective code in this section are adapted from
// the LIBLINEAR/scikit-learn vendored implementation. CERM changes only the
// sparse storage access: SciPy CSR is read directly instead of being expanded to
// LIBLINEAR feature_node arrays. The BSD-3-Clause notices are reproduced in
// THIRD_PARTY_NOTICES.md and LICENSES/.
// -----------------------------------------------------------------------------
typedef double (*dot_func)(int, const double*, int, const double*, int);
typedef void (*axpy_func)(int, double, const double*, int, double*, int);
typedef void (*scal_func)(int, double, const double*, int);
typedef double (*nrm2_func)(int, const double*, int);
typedef struct BlasFunctions { dot_func dot; axpy_func axpy; scal_func scal; nrm2_func nrm2; } BlasFunctions;

class function {
public:
    virtual double fun(double *w) = 0;
    virtual void grad(double *w, double *g) = 0;
    virtual void Hv(double *s, double *Hs) = 0;
    virtual int get_nr_variable(void) = 0;
    virtual ~function(void) {}
};

class TRON {
public:
    TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000, BlasFunctions *blas = 0);
    ~TRON();
    int tron(double *w);
    void set_print_string(void (*i_print) (const char *buf));
private:
    int trcg(double delta, double *g, double *s, double *r);
    double norm_inf(int n, double *x);
    double eps;
    int max_iter;
    function *fun_obj;
    BlasFunctions *blas;
    void info(const char *fmt,...);
    void (*tron_print_string)(const char *buf);
};

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

static void default_print(const char *buf) { fputs(buf,stdout); fflush(stdout); }

void TRON::info(const char *fmt,...) {
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsnprintf(buf,sizeof buf,fmt,ap);
    va_end(ap);
    (*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, double eps, int max_iter, BlasFunctions *blas) {
    this->fun_obj=const_cast<function *>(fun_obj);
    this->eps=eps;
    this->max_iter=max_iter;
    this->blas=blas;
    tron_print_string = default_print;
}
TRON::~TRON() {}

int TRON::tron(double *w) {
    double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
    double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;
    int n = fun_obj->get_nr_variable();
    int i, cg_iter;
    double delta, snorm;
    double alpha, f, fnew, prered, actred, gs;
    int search = 1, iter = 1, inc = 1;
    double *s = new double[n];
    double *r = new double[n];
    double *w_new = new double[n];
    double *g = new double[n];
    for (i=0; i<n; i++) w[i] = 0;
    f = fun_obj->fun(w);
    fun_obj->grad(w, g);
    delta = blas->nrm2(n, g, inc);
    double gnorm1 = delta;
    double gnorm = gnorm1;
    if (gnorm <= eps*gnorm1) search = 0;
    iter = 1;
    while (iter <= max_iter && search) {
        cg_iter = trcg(delta, g, s, r);
        memcpy(w_new, w, sizeof(double)*n);
        blas->axpy(n, 1.0, s, inc, w_new, inc);
        gs = blas->dot(n, g, inc, s, inc);
        prered = -0.5*(gs - blas->dot(n, s, inc, r, inc));
        fnew = fun_obj->fun(w_new);
        actred = f - fnew;
        snorm = blas->nrm2(n, s, inc);
        if (iter == 1) delta = min(delta, snorm);
        if (fnew - f - gs <= 0) alpha = sigma3;
        else alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));
        if (actred < eta0*prered)
            delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
        else if (actred < eta1*prered)
            delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
        else if (actred < eta2*prered)
            delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
        else
            delta = max(delta, min(alpha*snorm, sigma3*delta));
        info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);
        if (actred > eta0*prered) {
            iter++;
            memcpy(w, w_new, sizeof(double)*n);
            f = fnew;
            fun_obj->grad(w, g);
            gnorm = blas->nrm2(n, g, inc);
            if (gnorm <= eps*gnorm1) break;
        }
        if (f < -1.0e+32) { info("WARNING: f < -1.0e+32\n"); break; }
        if (fabs(actred) <= 0 && prered <= 0) { info("WARNING: actred and prered <= 0\n"); break; }
        if (fabs(actred) <= 1.0e-12*fabs(f) && fabs(prered) <= 1.0e-12*fabs(f)) {
            info("WARNING: actred and prered too small\n"); break;
        }
    }
    delete[] g;
    delete[] r;
    delete[] w_new;
    delete[] s;
    return --iter;
}

int TRON::trcg(double delta, double *g, double *s, double *r) {
    int i, inc = 1;
    int n = fun_obj->get_nr_variable();
    double *d = new double[n];
    double *Hd = new double[n];
    double rTr, rnewTrnew, alpha, beta, cgtol;
    for (i=0; i<n; i++) { s[i] = 0; r[i] = -g[i]; d[i] = r[i]; }
    cgtol = 0.1 * blas->nrm2(n, g, inc);
    int cg_iter = 0;
    rTr = blas->dot(n, r, inc, r, inc);
    while (1) {
        if (blas->nrm2(n, r, inc) <= cgtol) break;
        cg_iter++;
        fun_obj->Hv(d, Hd);
        alpha = rTr / blas->dot(n, d, inc, Hd, inc);
        blas->axpy(n, alpha, d, inc, s, inc);
        if (blas->nrm2(n, s, inc) > delta) {
            info("cg reaches trust region boundary\n");
            alpha = -alpha;
            blas->axpy(n, alpha, d, inc, s, inc);
            double std = blas->dot(n, s, inc, d, inc);
            double sts = blas->dot(n, s, inc, s, inc);
            double dtd = blas->dot(n, d, inc, d, inc);
            double dsq = delta*delta;
            double rad = sqrt(std*std + dtd*(dsq-sts));
            if (std >= 0) alpha = (dsq - sts)/(std + rad);
            else alpha = (rad - std)/dtd;
            blas->axpy(n, alpha, d, inc, s, inc);
            alpha = -alpha;
            blas->axpy(n, alpha, Hd, inc, r, inc);
            break;
        }
        alpha = -alpha;
        blas->axpy(n, alpha, Hd, inc, r, inc);
        rnewTrnew = blas->dot(n, r, inc, r, inc);
        beta = rnewTrnew/rTr;
        blas->scal(n, beta, d, inc);
        blas->axpy(n, 1.0, r, inc, d, inc);
        rTr = rnewTrnew;
    }
    delete[] d;
    delete[] Hd;
    return(cg_iter);
}

double TRON::norm_inf(int n, double *x) {
    double dmax = fabs(x[0]);
    for (int i=1; i<n; i++) if (fabs(x[i]) >= dmax) dmax = fabs(x[i]);
    return(dmax);
}
void TRON::set_print_string(void (*print_string) (const char *buf)) { tron_print_string = print_string; }

using f_ddot = double (*)(int*, double*, int*, double*, int*);
using f_daxpy = void (*)(int*, double*, double*, int*, double*, int*);
using f_dscal = void (*)(int*, double*, double*, int*);
using f_dnrm2 = double (*)(int*, double*, int*);
static f_ddot g_ddot=nullptr;
static f_daxpy g_daxpy=nullptr;
static f_dscal g_dscal=nullptr;
static f_dnrm2 g_dnrm2=nullptr;

extern "C" CERM_EXPORT int cerm_training_core_set_blas(void* ddot, void* daxpy, void* dscal, void* dnrm2) {
    if(!ddot||!daxpy||!dscal||!dnrm2) return -1;
    g_ddot=reinterpret_cast<f_ddot>(ddot);
    g_daxpy=reinterpret_cast<f_daxpy>(daxpy);
    g_dscal=reinterpret_cast<f_dscal>(dscal);
    g_dnrm2=reinterpret_cast<f_dnrm2>(dnrm2);
    return 0;
}
static double blas_dot(int n,const double*x,int ix,const double*y,int iy) { return g_ddot(&n,const_cast<double*>(x),&ix,const_cast<double*>(y),&iy); }
static void blas_axpy(int n,double a,const double*x,int ix,double*y,int iy) { g_daxpy(&n,&a,const_cast<double*>(x),&ix,y,&iy); }
static void blas_scal(int n,double a,const double*x,int ix) { g_dscal(&n,&a,const_cast<double*>(x),&ix); }
static double blas_nrm2(int n,const double*x,int ix) { return g_dnrm2(&n,const_cast<double*>(x),&ix); }
static void print_null(const char*) {}

class csr_l2r_lr_fun : public function {
public:
    csr_l2r_lr_fun(const double* data,const int32_t* indices,const int32_t* indptr,
                   int n_samples,int n_features,const double* y01,const double* sw,double C)
      : data_(data),indices_(indices),indptr_(indptr),l_(n_samples),nf_(n_features),n_(n_features+1),
        z_(n_samples),D_(n_samples),C_(n_samples),perm_(n_samples),y_(n_samples) {
        int active=0;
        for(int i=0;i<l_;++i) if(sw[i]>0.0 && (int)y01[i]==0) perm_[active++]=i;
        const int neg=active;
        for(int i=0;i<l_;++i) if(sw[i]>0.0 && (int)y01[i]==1) perm_[active++]=i;
        neg_=neg;
        pos_=active-neg;
        active_=active;
        perm_.resize(active_);
        y_.resize(active_);
        C_.resize(active_);
        z_.resize(active_);
        D_.resize(active_);
        for(int q=0;q<active_;++q) {
            const int r=perm_[q];
            y_[q]=(q<neg_)?-1.0:1.0;
            C_[q]=sw[r]*C;
        }
    }
    double fun(double *w) override {
        double f=0.0;
        Xv(w,z_.data());
        for(int i=0;i<n_;++i) f+=w[i]*w[i];
        f/=2.0;
        for(int i=0;i<active_;++i) {
            const double yz=y_[i]*z_[i];
            if(yz>=0) f+=C_[i]*std::log(1+std::exp(-yz));
            else f+=C_[i]*(-yz+std::log(1+std::exp(yz)));
        }
        return f;
    }
    void grad(double*w,double*g) override {
        for(int i=0;i<active_;++i) {
            z_[i]=1/(1+std::exp(-y_[i]*z_[i]));
            D_[i]=z_[i]*(1-z_[i]);
            z_[i]=C_[i]*(z_[i]-1)*y_[i];
        }
        XTv(z_.data(),g);
        for(int i=0;i<n_;++i) g[i]=w[i]+g[i];
    }
    void Hv(double*s,double*Hs) override {
        std::vector<double> wa(active_);
        Xv(s,wa.data());
        for(int i=0;i<active_;++i) wa[i]=C_[i]*D_[i]*wa[i];
        XTv(wa.data(),Hs);
        for(int i=0;i<n_;++i) Hs[i]=s[i]+Hs[i];
    }
    int get_nr_variable() override { return n_; }
    double tol(double eps) const { return eps*std::max(std::min(pos_,neg_),1)/double(active_); }
private:
    void Xv(const double*v,double*out) {
        for(int q=0;q<active_;++q) {
            const int r=perm_[q];
            double sum=0.0;
            for(int k=indptr_[r];k<indptr_[r+1];++k) sum+=v[indices_[k]]*data_[k];
            sum+=v[nf_];
            out[q]=sum;
        }
    }
    void XTv(const double*v,double*out) {
        std::fill(out,out+n_,0.0);
        for(int q=0;q<active_;++q) {
            const int r=perm_[q];
            const double vi=v[q];
            for(int k=indptr_[r];k<indptr_[r+1];++k) out[indices_[k]]+=vi*data_[k];
            out[nf_]+=vi;
        }
    }
    const double*data_;
    const int32_t*indices_;
    const int32_t*indptr_;
    int l_,nf_,n_,neg_=0,pos_=0,active_=0;
    std::vector<double> z_,D_,C_,y_;
    std::vector<int> perm_;
};

extern "C" CERM_EXPORT int cerm_liblinear_train_csr_direct_f64(
    const double*data,const int32_t*indices,const int32_t*indptr,
    int32_t n_samples,int32_t n_features,int32_t nnz,
    const double*y,const double*sw,double C,uint32_t seed,int32_t max_iter,
    double*out_w,int32_t out_len,int32_t*out_n_iter) {
    (void)nnz;
    (void)seed;
    if(!g_ddot||!g_daxpy||!g_dscal||!g_dnrm2) return -10;
    if(!data||!indices||!indptr||!y||!sw||!out_w||!out_n_iter) return -1;
    if(out_len!=n_features+1||n_samples<=0) return -2;
    int n0=0,n1=0;
    for(int i=0;i<n_samples;++i) {
        const int yi=(int)y[i];
        if(yi!=0&&yi!=1) return -3;
        if(sw[i]<0.0||!std::isfinite(sw[i])) return -4;
        if(sw[i]>0.0) {
            if(yi==0) ++n0;
            else ++n1;
        }
    }
    if(n0==0||n1==0) return -5;
    csr_l2r_lr_fun fun(data,indices,indptr,n_samples,n_features,y,sw,C);
    BlasFunctions blas{blas_dot,blas_axpy,blas_scal,blas_nrm2};
    TRON tron(&fun,fun.tol(1e-4),max_iter,&blas);
    tron.set_print_string(print_null);
    const int nit=tron.tron(out_w);
    *out_n_iter=nit;
    return 0;
}


extern "C" CERM_EXPORT int cerm_weighted_quantile_f64(
    const double* values,
    const double* weights,
    int n,
    const double* quantiles,
    int n_q,
    double* out) {
    if (!values || !weights || !quantiles || !out || n < 0 || n_q < 0) {
        return -1;
    }
    if (n == 0 || n_q == 0) return 0;

    for (int q = 0; q < n_q; ++q) {
        if (!std::isfinite(quantiles[q]) ||
            quantiles[q] < 0.0 || quantiles[q] > 1.0) {
            return -5;
        }
    }

    struct Item {
        double val;
        double w;
    };
    std::vector<Item> items;
    items.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(values[i]) ||
            !std::isfinite(weights[i]) ||
            weights[i] < 0.0) {
            return -6;
        }
        if (weights[i] > 0.0) {
            items.push_back({values[i], weights[i]});
        }
    }
    const int active_n = static_cast<int>(items.size());
    if (active_n == 0) return -2;

    std::stable_sort(
        items.begin(), items.end(),
        [](const Item& a, const Item& b) { return a.val < b.val; });

    bool is_int = true;
    bool integer_mass_safe = true;
    int64_t checked_total = 0;
    const long double int64_limit = static_cast<long double>(INT64_MAX);
    for (int i = 0; i < active_n; ++i) {
        const double rounded = std::round(items[i].w);
        if (items[i].w != rounded) {
            is_int = false;
            break;
        }
        const long double candidate =
            static_cast<long double>(checked_total)
            + static_cast<long double>(rounded);
        // Stay strictly below the ambiguous float/int64 boundary. Larger
        // integer-frequency masses fall back to the real-weight ECDF path.
        if (candidate >= int64_limit) {
            integer_mass_safe = false;
            break;
        }
        checked_total = static_cast<int64_t>(candidate);
    }

    if (is_int && integer_mass_safe) {
        std::vector<int64_t> cumulative(active_n);
        int64_t running = 0;
        for (int i = 0; i < active_n; ++i) {
            running += static_cast<int64_t>(std::round(items[i].w));
            cumulative[i] = running;
        }
        const int64_t total = cumulative.back();
        if (total <= 0) return -2;

        for (int q = 0; q < n_q; ++q) {
            const double virtual_index =
                (static_cast<double>(total) - 1.0) * quantiles[q];
            const int64_t lower_rank =
                static_cast<int64_t>(std::floor(virtual_index));
            const int64_t upper_rank =
                static_cast<int64_t>(std::ceil(virtual_index));
            const double fraction =
                virtual_index - static_cast<double>(lower_rank);

            auto lower_it = std::upper_bound(
                cumulative.begin(), cumulative.end(), lower_rank);
            auto upper_it = std::upper_bound(
                cumulative.begin(), cumulative.end(), upper_rank);
            int lower_idx =
                static_cast<int>(std::distance(cumulative.begin(), lower_it));
            int upper_idx =
                static_cast<int>(std::distance(cumulative.begin(), upper_it));
            if (lower_idx >= active_n) lower_idx = active_n - 1;
            if (upper_idx >= active_n) upper_idx = active_n - 1;

            const double lower_value = items[lower_idx].val;
            const double upper_value = items[upper_idx].val;
            const double difference = upper_value - lower_value;
            out[q] = fraction >= 0.5
                ? upper_value - difference * (1.0 - fraction)
                : lower_value + difference * fraction;
        }
        return 0;
    }

    double maximum_weight = 0.0;
    for (int i = 0; i < active_n; ++i) {
        maximum_weight = std::max(maximum_weight, items[i].w);
    }
    if (maximum_weight <= 0.0) return -3;

    std::vector<double> positions(active_n);
    double cumulative_weight = 0.0;
    for (int i = 0; i < active_n; ++i) {
        const double scaled = items[i].w / maximum_weight;
        cumulative_weight += scaled;
        positions[i] = cumulative_weight - 0.5 * scaled;
    }
    if (cumulative_weight <= 0.0 || !std::isfinite(cumulative_weight)) {
        return -4;
    }
    for (int i = 0; i < active_n; ++i) {
        positions[i] /= cumulative_weight;
    }

    for (int q = 0; q < n_q; ++q) {
        const double target = quantiles[q];
        if (target <= positions.front()) {
            out[q] = items.front().val;
            continue;
        }
        if (target >= positions.back()) {
            out[q] = items.back().val;
            continue;
        }
        auto it = std::lower_bound(
            positions.begin(), positions.end(), target);
        const int idx =
            static_cast<int>(std::distance(positions.begin(), it));
        const int previous = idx - 1;
        const double p0 = positions[previous];
        const double p1 = positions[idx];
        const double v0 = items[previous].val;
        const double v1 = items[idx].val;
        out[q] = p1 == p0
            ? v0
            : v0 + (v1 - v0) * (target - p0) / (p1 - p0);
    }
    return 0;
}


class csr_semantic_blocks_l2r_lr_fun : public function {
public:
    csr_semantic_blocks_l2r_lr_fun(
        const double* data,const int32_t* indices,const int32_t* indptr,
        int n_samples,int n_base_features,
        const uint8_t* target_states,int state_width,
        const int32_t* block_target,
        const int32_t* block_coef_offsets,
        const uint8_t* block_state_ids,
        const double* block_centers,const double* block_scales,
        const int32_t* block_row_offsets,const int32_t* block_rows,
        int n_blocks,int n_block_features,
        const double* y01,const double* sw,double C)
      : data_(data),indices_(indices),indptr_(indptr),
        target_states_(target_states),state_width_(state_width),
        block_target_(block_target),block_coef_offsets_(block_coef_offsets),
        block_state_ids_(block_state_ids),block_centers_(block_centers),
        block_scales_(block_scales),block_row_offsets_(block_row_offsets),
        block_rows_(block_rows),n_blocks_(n_blocks),
        l_(n_samples),nf_base_(n_base_features),nf_block_(n_block_features),
        n_(n_base_features+n_block_features+1),
        z_(n_samples),D_(n_samples),C_(n_samples),perm_(n_samples),y_(n_samples),
        active_of_row_(n_samples,-1),
        state_to_local_(static_cast<size_t>(n_blocks)*256,-1) {
        int active=0;
        for(int i=0;i<l_;++i) if(sw[i]>0.0 && (int)y01[i]==0) perm_[active++]=i;
        const int neg=active;
        for(int i=0;i<l_;++i) if(sw[i]>0.0 && (int)y01[i]==1) perm_[active++]=i;
        neg_=neg;
        pos_=active-neg;
        active_=active;
        perm_.resize(active_);
        y_.resize(active_);
        C_.resize(active_);
        z_.resize(active_);
        D_.resize(active_);
        for(int q=0;q<active_;++q) {
            const int r=perm_[q];
            active_of_row_[r]=q;
            y_[q]=(q<neg_)?-1.0:1.0;
            C_[q]=sw[r]*C;
        }
        for(int b=0;b<n_blocks_;++b) {
            const int begin=block_coef_offsets_[b];
            const int end=block_coef_offsets_[b+1];
            for(int p=begin;p<end;++p) {
                const int state=static_cast<int>(block_state_ids_[p]);
                state_to_local_[static_cast<size_t>(b)*256+state]=p-begin;
            }
        }

        const int n_block_rows=block_row_offsets_[n_blocks_];
        block_active_row_.resize(n_block_rows,-1);
        block_row_local_state_.resize(n_block_rows,-1);
        state_sum_scratch_.resize(nf_block_,0.0);
        for(int b=0;b<n_blocks_;++b) {
            const int target=block_target_[b];
            for(int qrow=block_row_offsets_[b];qrow<block_row_offsets_[b+1];++qrow) {
                const int r=block_rows_[qrow];
                block_active_row_[qrow]=active_of_row_[r];
                const int state=static_cast<int>(
                    target_states_[static_cast<size_t>(r)*state_width_+target]
                );
                block_row_local_state_[qrow]=
                    state_to_local_[static_cast<size_t>(b)*256+state];
            }
        }
    }
    double fun(double *w) override {
        double f=0.0;
        Xv(w,z_.data());
        for(int i=0;i<n_;++i) f+=w[i]*w[i];
        f/=2.0;
        for(int i=0;i<active_;++i) {
            const double yz=y_[i]*z_[i];
            if(yz>=0) f+=C_[i]*std::log(1+std::exp(-yz));
            else f+=C_[i]*(-yz+std::log(1+std::exp(yz)));
        }
        return f;
    }
    void grad(double*w,double*g) override {
        for(int i=0;i<active_;++i) {
            z_[i]=1/(1+std::exp(-y_[i]*z_[i]));
            D_[i]=z_[i]*(1-z_[i]);
            z_[i]=C_[i]*(z_[i]-1)*y_[i];
        }
        XTv(z_.data(),g);
        for(int i=0;i<n_;++i) g[i]=w[i]+g[i];
    }
    void Hv(double*s,double*Hs) override {
        std::vector<double> wa(active_);
        Xv(s,wa.data());
        for(int i=0;i<active_;++i) wa[i]=C_[i]*D_[i]*wa[i];
        XTv(wa.data(),Hs);
        for(int i=0;i<n_;++i) Hs[i]=s[i]+Hs[i];
    }
    int get_nr_variable() override { return n_; }
    double tol(double eps) const {
        return eps*std::max(std::min(pos_,neg_),1)/double(active_);
    }
private:
    void Xv(const double*v,double*out) {
        const int intercept=nf_base_+nf_block_;
        for(int q=0;q<active_;++q) {
            const int r=perm_[q];
            double sum=0.0;
            for(int k=indptr_[r];k<indptr_[r+1];++k) {
                sum+=v[indices_[k]]*data_[k];
            }
            out[q]=sum+v[intercept];
        }
        for(int b=0;b<n_blocks_;++b) {
            const int begin=block_coef_offsets_[b];
            const int end=block_coef_offsets_[b+1];
            double constant=0.0;
            for(int p=begin;p<end;++p) {
                constant+=v[nf_base_+p]*block_centers_[p]*block_scales_[p];
            }
            for(int qrow=block_row_offsets_[b];qrow<block_row_offsets_[b+1];++qrow) {
                const int q=block_active_row_[qrow];
                if(q<0) continue;
                const int local=block_row_local_state_[qrow];
                out[q]-=constant;
                if(local>=0) {
                    const int p=begin+local;
                    out[q]+=v[nf_base_+p]*block_scales_[p];
                }
            }
        }
    }
    void XTv(const double*v,double*out) {
        std::fill(out,out+n_,0.0);
        const int intercept=nf_base_+nf_block_;
        for(int q=0;q<active_;++q) {
            const int r=perm_[q];
            const double vi=v[q];
            for(int k=indptr_[r];k<indptr_[r+1];++k) {
                out[indices_[k]]+=vi*data_[k];
            }
            out[intercept]+=vi;
        }
        for(int b=0;b<n_blocks_;++b) {
            const int begin=block_coef_offsets_[b];
            const int end=block_coef_offsets_[b+1];
            std::fill(
                state_sum_scratch_.begin()+begin,
                state_sum_scratch_.begin()+end,
                0.0
            );
            double total=0.0;
            for(int qrow=block_row_offsets_[b];qrow<block_row_offsets_[b+1];++qrow) {
                const int q=block_active_row_[qrow];
                if(q<0) continue;
                const double vi=v[q];
                total+=vi;
                const int local=block_row_local_state_[qrow];
                if(local>=0) state_sum_scratch_[begin+local]+=vi;
            }
            for(int p=begin;p<end;++p) {
                out[nf_base_+p]=
                    block_scales_[p]*
                    (state_sum_scratch_[p]-block_centers_[p]*total);
            }
        }
    }

    const double*data_;
    const int32_t*indices_;
    const int32_t*indptr_;
    const uint8_t*target_states_;
    int state_width_;
    const int32_t*block_target_;
    const int32_t*block_coef_offsets_;
    const uint8_t*block_state_ids_;
    const double*block_centers_;
    const double*block_scales_;
    const int32_t*block_row_offsets_;
    const int32_t*block_rows_;
    int n_blocks_;
    int l_,nf_base_,nf_block_,n_,neg_=0,pos_=0,active_=0;
    std::vector<double> z_,D_,C_,y_;
    std::vector<int> perm_,active_of_row_;
    std::vector<int16_t> state_to_local_;
    std::vector<int32_t> block_active_row_;
    std::vector<int16_t> block_row_local_state_;
    std::vector<double> state_sum_scratch_;
};

extern "C" CERM_EXPORT int cerm_liblinear_train_csr_semantic_blocks_f64(
    const double*data,const int32_t*indices,const int32_t*indptr,
    int32_t n_samples,int32_t n_base_features,int32_t nnz,
    const uint8_t*target_states,int32_t state_width,
    const int32_t*block_target,const int32_t*block_coef_offsets,
    const uint8_t*block_state_ids,
    const double*block_centers,const double*block_scales,
    const int32_t*block_row_offsets,const int32_t*block_rows,
    int32_t n_blocks,int32_t n_block_features,
    const double*y,const double*sw,double C,uint32_t seed,int32_t max_iter,
    double*out_w,int32_t out_len,int32_t*out_n_iter) {
    (void)nnz;
    (void)seed;
    if(!g_ddot||!g_daxpy||!g_dscal||!g_dnrm2) return -20;
    if(!data||!indices||!indptr||!target_states||
       !block_coef_offsets||!block_row_offsets||
       !y||!sw||!out_w||!out_n_iter) return -21;
    if(n_samples<=0||n_base_features<0||state_width<=0||
       n_blocks<0||n_block_features<0) return -22;
    if(out_len!=n_base_features+n_block_features+1) return -23;
    if(block_coef_offsets[0]!=0||
       block_coef_offsets[n_blocks]!=n_block_features||
       block_row_offsets[0]!=0) return -24;
    if(n_blocks>0 && (!block_target||!block_state_ids||
       !block_centers||!block_scales||!block_rows)) return -25;
    int n0=0,n1=0;
    for(int i=0;i<n_samples;++i) {
        const int yi=(int)y[i];
        if(yi!=0&&yi!=1) return -26;
        if(sw[i]<0.0||!std::isfinite(sw[i])) return -27;
        if(sw[i]>0.0) {
            if(yi==0) ++n0;
            else ++n1;
        }
    }
    if(n0==0||n1==0) return -28;
    for(int b=0;b<n_blocks;++b) {
        if(block_target[b]<0||block_target[b]>=state_width) return -29;
        if(block_coef_offsets[b]>block_coef_offsets[b+1]||
           block_row_offsets[b]>block_row_offsets[b+1]) return -30;
        for(int q=block_row_offsets[b];q<block_row_offsets[b+1];++q) {
            if(block_rows[q]<0||block_rows[q]>=n_samples) return -31;
        }
    }
    csr_semantic_blocks_l2r_lr_fun fun(
        data,indices,indptr,n_samples,n_base_features,
        target_states,state_width,
        block_target,block_coef_offsets,block_state_ids,
        block_centers,block_scales,
        block_row_offsets,block_rows,
        n_blocks,n_block_features,y,sw,C
    );
    BlasFunctions blas{blas_dot,blas_axpy,blas_scal,blas_nrm2};
    TRON tron(&fun,fun.tol(1e-4),max_iter,&blas);
    tron.set_print_string(print_null);
    const int nit=tron.tron(out_w);
    *out_n_iter=nit;
    return 0;
}
