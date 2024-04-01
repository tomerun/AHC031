#include <sys/time.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#ifdef LOCAL
#ifndef NDEBUG
#define MEASURE_TIME
#define DEBUG
#endif
#else
#define NDEBUG
// #define DEBUG
#endif
#include <cassert>

using namespace std;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using ll = int64_t;
using ull = uint64_t;
using vi = vector<int>;
using vvi = vector<vi>;
using vvvi = vector<vvi>;
using pi = pair<int, int>;

namespace {

#ifdef LOCAL
constexpr ll TL = 500;
#else
constexpr ll TL = 2000;
#endif

inline ll get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  ll result = tv.tv_sec * 1000LL + tv.tv_usec / 1000LL;
  return result;
}

const ll start_time = get_time(); // msec

inline ll get_elapsed_msec() { return get_time() - start_time; }

struct Rand {
  uint32_t x, y, z, w;
  static const double TO_DOUBLE;

  Rand() {
    x = 123456789;
    y = 362436069;
    z = 521288629;
    w = 88675123;
  }

  template <class int_type> int next(int_type n) {
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    return w % n;
  }

  int next() {
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
  }

  double next(double max) { return (uint32_t)next() * TO_DOUBLE * max; }
};
const double Rand::TO_DOUBLE = 1.0 / (1LL << 32);

struct Counter {
  vector<ull> cnt;

  void add(int i) {
    if (i >= cnt.size()) {
      cnt.resize(i + 1);
    }
    ++cnt[i];
  }

  void print() {
    cerr << "counter:[";
    for (int i = 0; i < cnt.size(); ++i) {
      cerr << cnt[i] << ", ";
      if (i % 10 == 9) cerr << endl;
    }
    cerr << "]" << endl;
  }
};

struct Timer {
  vector<ull> at;
  vector<ull> sum;

  void start(int i) {
    if (i >= at.size()) {
      at.resize(i + 1);
      sum.resize(i + 1);
    }
    at[i] = get_time();
  }

  void stop(int i) { sum[i] += get_time() - at[i]; }

  void print() {
    cerr << "timer:[";
    for (int i = 0; i < at.size(); ++i) {
      cerr << sum[i] << ", ";
      if (i % 10 == 9) cerr << endl;
    }
    cerr << "]" << endl;
  }
};

} // namespace

#ifdef MEASURE_TIME
#define START_TIMER(i) (timer.start(i))
#define STOP_TIMER(i) (timer.stop(i))
#define PRINT_TIMER() (timer.print())
#define ADD_COUNTER(i) (counter.add(i))
#define PRINT_COUNTER() (counter.print())
#else
#define START_TIMER(i)
#define STOP_TIMER(i)
#define PRINT_TIMER()
#define ADD_COUNTER(i)
#define PRINT_COUNTER()
#endif

#ifdef DEBUG
#define debug(format, ...) fprintf(stderr, format, __VA_ARGS__)
#define debugStr(str) fprintf(stderr, str)
#define debugln() fprintf(stderr, "\n")
#else
#define debug(format, ...)
#define debugStr(str)
#define debugln()
#endif

template <class T> constexpr inline T sq(T v) { return v * v; }

void debug_vec(const vi& vec, const string& name = "") {
  if (name.empty()) {
    debugStr("[");
  } else {
    debug("%s: [", name.c_str());
  }
  for (int i = 0; i < vec.size(); ++i) {
    debug("%d ", vec[i]);
  }
  debugStr("]\n");
}

Rand rnd;
Timer timer;
Counter counter;

template <class T> void shuffle(vector<T>& v) {
  for (int i = 0; i + 1 < v.size(); ++i) {
    int pos = rnd.next(v.size() - i) + i;
    swap(v[i], v[pos]);
  }
}

//////// end of template ////////

struct Rect {
  int top, left, bottom, right;
  int area() const { return (bottom - top) * (right - left); }
};
bool operator<(const Rect& r1, const Rect& r2) { return r1.area() < r2.area(); }

constexpr int INF = 1 << 29;
constexpr int W = 1000;
constexpr vi EMPTY_VI;
constexpr vvi EMPTY_VVI;
int D;
int N;
double E;
array<array<int, 50>, 50> A;

struct Result {
  vector<vector<Rect>> rects;
  int64_t pena_area;
  int64_t pena_wall;
  int col;

  Result(vector<vector<Rect>> rects_, int64_t pena_area_, int64_t pena_wall_, int col_)
      : rects(rects_), pena_area(pena_area_), pena_wall(pena_wall_), col(col_) {}

  int64_t score() const { return pena_area + pena_wall + 1; }
};

const Result RESULT_EMPTY(vector<vector<Rect>>(), INF, INF, -1);

struct FreeSolution {
  vector<vector<Rect>> rects;
  vector<int64_t> pena_area;
  vector<int64_t> pena_wall;
  vvi wss;
  FreeSolution() : rects(D), pena_area(D, INF), pena_wall(D, INF), wss(D, vi(1, W)) {}

  int64_t score() const { return sum_pena_area() + sum_pena_wall() + 1; }

  int64_t sum_pena_area() const { return accumulate(pena_area.begin(), pena_area.end(), 0ll); }

  int64_t sum_pena_wall() const { return accumulate(pena_wall.begin(), pena_wall.end(), 0ll); }

  Result to_result() const {
    debugStr("------ to_result ------\n");
    int64_t sum = 0;
    for (int i = 0; i < D; ++i) {
      sum += pena_area[i] + pena_wall[i];
      debug("day:%d pena_area:%lld pena_wall:%lld sum:%lld\n", i, pena_area[i], pena_wall[i], sum);
      debug_vec(wss[i]);
    }
    return Result(rects, sum_pena_area(), sum_pena_wall(), -1);
  }
};

struct FixColumnSolution {
  vi ws;
  vvvi nis;
  vvvi hss;
  int64_t pena_area;
  int64_t pena_wall;
  FixColumnSolution(const vi& ws_, const vvvi& nis_, const vvvi& hss_, int64_t pena_area_, int64_t pena_wall_)
      : ws(ws_), nis(nis_), hss(hss_), pena_area(pena_area_), pena_wall(pena_wall_) {}

  int64_t score() const { return pena_area + pena_wall + 1; }

  Result to_result() const {
    debugStr("------ to_result ------\n");
    int64_t real_pena_area = 0;
    vector<vector<Rect>> rects(D);
    debug_vec(ws, "ws");
    for (int i = 0; i < D; ++i) {
      int x = 0;
      for (int j = 0; j < ws.size(); ++j) {
        debug("%d %d\n", i, j);
        debug_vec(hss[i][j], "hss");
        debug_vec(nis[i][j], "nis");
        int y = 0;
        for (int k = 0; k < hss[i][j].size(); ++k) {
          rects[i].emplace_back(y, x, y + hss[i][j][k], x + ws[j]);
          y += hss[i][j][k];
        }
        x += ws[j];
      }
      sort(rects[i].begin(), rects[i].end());
      for (int j = 0; j < N; ++j) {
        if (rects[i][j].area() < A[i][j]) {
          real_pena_area += (A[i][j] - rects[i][j].area()) * 100;
        }
      }
    }
    debug("pena_area:%lld pena_wall:%lld\n", real_pena_area, pena_wall);
    return Result(rects, real_pena_area, pena_wall, ws.size());
  }
};
const FixColumnSolution SOL_EMPTY(vi(), vvvi(), vvvi(), INF, INF);

bool accept(int64_t diff, double cooler) {
  if (diff <= 0) return true;
  double v = -diff * cooler;
  return rnd.next(1.0) < exp(v);
}

struct Solver {
  const ll timelimit;
  int64_t best_score;

  Solver(ll timelimit_) : timelimit(timelimit_), best_score(INF) {}

  Result solve() {
    // score=1 狙い
    vi max_areas(N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        max_areas[i] = max(max_areas[i], A[j][i]);
      }
    }
    int sum_max_area = accumulate(max_areas.begin(), max_areas.end(), 0);
    debug("sum_max_area:%d\n", sum_max_area);
    if (sum_max_area <= W * W) {
      START_TIMER(0);
      Result result = solve_nomove(max_areas);
      STOP_TIMER(0);
      if (result.score() == 1) return result;
    }

    Result best_result = RESULT_EMPTY;
    START_TIMER(1);
    // 壁のことは無視して面積ペナルティのみを最小化しようとする解
    FreeSolution noarea_sol;
    if (E < 0.02 || N <= 6) {
      solve_noarea(noarea_sol, get_time() + 50);
    }
    debug("noarea_score:%lld\n", noarea_sol.score());
    STOP_TIMER(1);
    START_TIMER(2);
    // 縦の壁だけ固定して面積ペナルティを最小化しようとする解
    ll time_left = timelimit - get_time();
    FixColumnSolution noarea_fix_sol = solve_noarea_fixed_column(get_time() + time_left / 3);
    STOP_TIMER(2);
    int col = noarea_fix_sol.ws.size();
    debug("fix_col:%d\n", col);
    debug_vec(noarea_fix_sol.ws, "ws");
    for (int i = 0; i < D; ++i) {
      debugStr("[");
      for (int j = 0; j < col; ++j) {
        debug("%lu ", noarea_fix_sol.nis[i][j].size());
      }
      debugStr("]\n");
    }

    if (noarea_sol.score() * 0.8 < noarea_fix_sol.score() * 0.6) {
      solve_noarea(noarea_sol, timelimit);
      best_result = noarea_sol.to_result();
    } else {
      noarea_fix_sol = improve(noarea_fix_sol, timelimit);
      best_result = noarea_fix_sol.to_result();
    }

    return best_result;
  }

  void set_sep_info(const vi& hs, vi& sep_cnt, vi& prev_sep, vi& next_sep) {
    fill(sep_cnt.begin(), sep_cnt.end(), 0);
    int y = 0;
    for (int h : hs) {
      y += h;
      sep_cnt[y]++;
    }
    if (hs.empty()) {
      sep_cnt[W] = 1;
    }
    // sep_cnt[W] = 1; // Wまで詰めないケースを考慮
    int prev = 0;
    int next = sep_cnt[W] == 1 ? W : -1;
    next_sep[W] = next;
    for (int i = 1; i <= W; ++i) {
      prev_sep[i] = prev;
      if (sep_cnt[i]) prev = i;
      next_sep[W - i] = next;
      if (sep_cnt[W - i]) next = W - i;
    }
    for (int i = 0; i < W; ++i) {
      sep_cnt[i + 1] += sep_cnt[i];
    }
  }

  FixColumnSolution improve(FixColumnSolution& sol, int64_t tl) {
    const int col = sol.ws.size();
    debug("start improve col:%d\n", col);
    vvvi sep_cnt(D, vvi(col, vi(W + 1)));
    vvvi prev_sep(D, vvi(col, vi(W + 1)));
    vvvi next_sep(D, vvi(col, vi(W + 1)));
    vvi pena_sep(D + 1, vi(col));
    vvi pena_area(D, vi(col));
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < col; ++j) {
        sort(sol.nis[i][j].begin(), sol.nis[i][j].end());
        set_sep_info(sol.hss[i][j], sep_cnt[i][j], prev_sep[i][j], next_sep[i][j]);
      }
    }
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < col; ++j) {
        int y = 0;
        for (int k = 0; k < sol.hss[i][j].size(); ++k) {
          int area_diff = A[i][sol.nis[i][j][k]] - sol.hss[i][j][k] * sol.ws[j];
          if (area_diff > 0) {
            pena_area[i][j] += area_diff * 100;
          }
          y += sol.hss[i][j][k];
          if (y == W) continue;
          if (i != 0) {
            if (sep_cnt[i - 1][j][y - 1] == sep_cnt[i - 1][j][y]) {
              pena_sep[i][j] += sol.ws[j];
            }
          }
          if (i != D - 1) {
            if (sep_cnt[i + 1][j][y - 1] == sep_cnt[i + 1][j][y]) {
              pena_sep[i + 1][j] += sol.ws[j];
            }
          }
        }
      }
    }
    sol.pena_area = 0;
    sol.pena_wall = 0;
    for (int i = 0; i < D; ++i) {
      sol.pena_area += accumulate(pena_area[i].begin(), pena_area[i].end(), 0);
      sol.pena_wall += accumulate(pena_sep[i].begin(), pena_sep[i].end(), 0);
    }
    debug("pena_area:%lld pena_wall:%lld\n", sol.pena_area, sol.pena_wall);

    auto update_col = [&](int day, int c) {
      vi new_hs = sol.hss[day][c];
      if (day != 0) {
        pena_sep[day][c] = (sol.nis[day][c].size() + sep_cnt[day - 1][c][W] - 2) * sol.ws[c];
        int y = 0;
        for (int h : new_hs) {
          y += h;
          if (y == W) break;
          if (sep_cnt[day - 1][c][y - 1] != sep_cnt[day - 1][c][y]) {
            pena_sep[day][c] -= 2 * sol.ws[c];
          }
        }
        if (y < W) {
          pena_sep[day][c] += sol.ws[c];
        }
        if (next_sep[day - 1][c][W] == -1) {
          pena_sep[day][c] += sol.ws[c];
        }
      }
      if (day != D - 1) {
        pena_sep[day + 1][c] = (sol.nis[day][c].size() + sep_cnt[day + 1][c][W] - 2) * sol.ws[c];
        int y = 0;
        for (int h : new_hs) {
          y += h;
          if (y == W) break;
          if (sep_cnt[day + 1][c][y - 1] != sep_cnt[day + 1][c][y]) {
            pena_sep[day + 1][c] -= 2 * sol.ws[c];
          }
        }
        if (y < W) {
          pena_sep[day + 1][c] += sol.ws[c];
        }
        if (next_sep[day + 1][c][W] == -1) {
          pena_sep[day + 1][c] += sol.ws[c];
        }
      }
      sort(new_hs.begin(), new_hs.end());
      sort(sol.nis[day][c].begin(), sol.nis[day][c].end());
      pena_area[day][c] = 0;
      for (int i = 0; i < sol.nis[day][c].size(); ++i) {
        int area_diff = A[day][sol.nis[day][c][i]] - new_hs[i] * sol.ws[c];
        if (area_diff > 0) {
          pena_area[day][c] += area_diff * 100;
        }
      }
      set_sep_info(sol.hss[day][c], sep_cnt[day][c], prev_sep[day][c], next_sep[day][c]);
    };

    FixColumnSolution best_sol = sol;
    int pena = sol.pena_area + sol.pena_wall;
    const int type_th_swap = 0x2F;
    for (int turn = 0;; ++turn) {
      auto cur_time = get_time();
      if (tl < cur_time) {
        debug("turn:%d\n", turn);
        break;
      }

      if ((turn & 0x3FFF) == 0) {
        for (int day = 0; day < D; ++day) {
          vector<pair<int, int>> as;
          for (int i = 0; i < col; ++i) {
            for (int h : sol.hss[day][i]) {
              as.emplace_back(h * sol.ws[i], i);
            }
          }
          sort(as.begin(), as.end());
          vvi snis(col);
          for (int i = 0; i < N; ++i) {
            snis[as[i].second].push_back(i);
          }
          sol.nis[day] = snis;
        }
      }

      int c0 = rnd.next(col);
      int c1;
      if (col == 1) {
        c1 = 0;
      } else {
        c1 = rnd.next(col - 1);
        if (c0 <= c1) c1++;
      }
      int day = rnd.next(D);
      if (sol.nis[day][c0].size() == 0) {
        continue;
      }
      int type = rnd.next() & 0x3F;
      int diff = 0;
      if (col > 1 && type <= 0) {
        // 行の幅を変更
        if (sol.ws[c1] == 1) continue;
        for (day = 0; day < D; ++day) {
          diff -= pena_sep[day][c0];
          diff -= pena_sep[day][c1];
          diff -= pena_area[day][c0];
          diff -= pena_area[day][c1];
        }
        for (day = 0; day < D; ++day) {
          assert(pena_sep[day][c0] % sol.ws[c0] == 0);
          assert(pena_sep[day][c1] % sol.ws[c1] == 0);
          diff += pena_sep[day][c0] * (sol.ws[c0] + 1) / sol.ws[c0];
          diff += pena_sep[day][c1] * (sol.ws[c1] - 1) / sol.ws[c1];
          vi hs = sol.hss[day][c0];
          sort(hs.begin(), hs.end());
          for (int i = 0; i < sol.nis[day][c0].size(); ++i) {
            int area_diff = A[day][sol.nis[day][c0][i]] - hs[i] * (sol.ws[c0] + 1);
            if (area_diff > 0) {
              diff += area_diff * 100;
            }
          }
          hs = sol.hss[day][c1];
          sort(hs.begin(), hs.end());
          for (int i = 0; i < sol.nis[day][c1].size(); ++i) {
            int area_diff = A[day][sol.nis[day][c1][i]] - hs[i] * (sol.ws[c1] - 1);
            if (area_diff > 0) {
              diff += area_diff * 100;
            }
          }
        }
        // debug("diff width:%d\n", diff);
        if (diff <= 0) {
          pena += diff;
          for (day = 0; day < D; ++day) {
            pena_sep[day][c0] = pena_sep[day][c0] * (sol.ws[c0] + 1) / sol.ws[c0];
            pena_sep[day][c1] = pena_sep[day][c1] * (sol.ws[c1] - 1) / sol.ws[c1];
            pena_area[day][c0] = 0;
            pena_area[day][c1] = 0;
            vi hs = sol.hss[day][c0];
            sort(hs.begin(), hs.end());
            for (int i = 0; i < sol.nis[day][c0].size(); ++i) {
              int area_diff = A[day][sol.nis[day][c0][i]] - hs[i] * (sol.ws[c0] + 1);
              if (area_diff > 0) {
                pena_area[day][c0] += area_diff * 100;
              }
            }
            hs = sol.hss[day][c1];
            sort(hs.begin(), hs.end());
            for (int i = 0; i < sol.nis[day][c1].size(); ++i) {
              int area_diff = A[day][sol.nis[day][c1][i]] - hs[i] * (sol.ws[c1] - 1);
              if (area_diff > 0) {
                pena_area[day][c1] += area_diff * 100;
              }
            }
          }
          sol.ws[c0]++;
          sol.ws[c1]--;
          if (diff < 0) {
            best_sol = sol;
            debug("best_sol change_width:%d turn:%d\n", pena, turn);
          }
        }
      } else if (col == 1 || (type < 0xF && sol.nis[day][c0].size() > 1)) {
        // 順序だけ変える
        // clang-format off
        diff -= pena_area[day][c0];
        diff -= pena_sep[day][c0];
        diff -= pena_sep[day + 1][c0];
        auto [new_pena, new_hs] = improve_col(
          day,
          sol.nis[day][c0],
          sol.ws[c0],
          day == 0 ? EMPTY_VI : sep_cnt[day - 1][c0],
          day == 0 ? EMPTY_VI : prev_sep[day - 1][c0],
          day == 0 ? EMPTY_VI : next_sep[day - 1][c0],
          day == D - 1 ? EMPTY_VI : sep_cnt[day + 1][c0],
          day == D - 1 ? EMPTY_VI : prev_sep[day + 1][c0],
          day == D - 1 ? EMPTY_VI : next_sep[day + 1][c0],
          -diff
        );
        // clang-format on
        if (new_hs == sol.hss[day][c0]) {
          continue;
        }
        diff += new_pena;
        // debug("diff shuffle:%d\n", diff);
        if (diff <= 0) {
          debug("shuffle day:%d col:%d w:%d diff:%d n:%lu\n", day, c0, sol.ws[c0], diff, sol.nis[day][c0].size());
          sol.hss[day][c0] = new_hs;
          update_col(day, c0);
          int real_pena = pena_sep[day][c0] + pena_sep[day + 1][c0] + pena_area[day][c0];
          // debug("penas: %d %d %d %d %d\n", new_pena, pena_sep[day][c0], pena_sep[day + 1][c0], pena_area[day][c0], real_pena);
          assert((new_pena - real_pena) % 100 == 0); // 面積順ソートで改善することがあるので面積ペナルティのみずれる
          diff += real_pena - new_pena;
          if (diff < 0) {
            pena += diff;
            best_sol = sol;
            debug("best_sol move:%d turn:%d\n", pena, turn);
          }
        }
      } else {
        int p0 = rnd.next(sol.nis[day][c0].size());
        int p1 = 0;
        if (type < type_th_swap) {
          // 2要素を交換
          if (sol.nis[day][c1].size() == 0) {
            continue;
          }
          p1 = rnd.next(sol.nis[day][c1].size());
          swap(sol.nis[day][c0][p0], sol.nis[day][c1][p1]);
          int sum_a = 0;
          for (int ni : sol.nis[day][c0]) {
            sum_a += A[day][ni];
          }
          if (sum_a > sol.ws[c0] * W) {
            swap(sol.nis[day][c0][p0], sol.nis[day][c1][p1]);
            continue;
          }
          sum_a = 0;
          for (int ni : sol.nis[day][c1]) {
            sum_a += A[day][ni];
          }
          if (sum_a > sol.ws[c1] * W) {
            swap(sol.nis[day][c0][p0], sol.nis[day][c1][p1]);
            continue;
          }
        } else {
          // c0->c1へ1要素を移動
          int ma = sol.nis[day][c0][p0];
          int sum_a = A[day][ma];
          for (int ni : sol.nis[day][c1]) {
            sum_a += A[day][ni];
          }
          if (sum_a > sol.ws[c1] * W) {
            continue;
          }
          sol.nis[day][c0].erase(sol.nis[day][c0].begin() + p0);
          sol.nis[day][c1].push_back(ma);
        }
        diff -= pena_area[day][c0];
        diff -= pena_sep[day][c0];
        diff -= pena_sep[day + 1][c0];
        // clang-format off
        auto [new_pena0, new_hs0] = improve_col(
          day,
          sol.nis[day][c0],
          sol.ws[c0],
          day == 0 ? EMPTY_VI : sep_cnt[day - 1][c0],
          day == 0 ? EMPTY_VI : prev_sep[day - 1][c0],
          day == 0 ? EMPTY_VI : next_sep[day - 1][c0],
          day == D - 1 ? EMPTY_VI : sep_cnt[day + 1][c0],
          day == D - 1 ? EMPTY_VI : prev_sep[day + 1][c0],
          day == D - 1 ? EMPTY_VI : next_sep[day + 1][c0],
          -diff
        );
        diff += new_pena0;
        diff -= pena_area[day][c1];
        diff -= pena_sep[day][c1];
        diff -=  pena_sep[day + 1][c1];
        if (diff > 0) {
          if (type < type_th_swap) {
            swap(sol.nis[day][c0][p0], sol.nis[day][c1][p1]);
          } else {
            sol.nis[day][c0].insert(sol.nis[day][c0].begin() + p0, sol.nis[day][c1].back());
            sol.nis[day][c1].pop_back();
          }
          continue;
        }
        auto [new_pena1, new_hs1] = improve_col(
          day,
          sol.nis[day][c1],
          sol.ws[c1],
          day == 0 ? EMPTY_VI : sep_cnt[day - 1][c1],
          day == 0 ? EMPTY_VI : prev_sep[day - 1][c1],
          day == 0 ? EMPTY_VI : next_sep[day - 1][c1],
          day == D - 1 ? EMPTY_VI : sep_cnt[day + 1][c1],
          day == D - 1 ? EMPTY_VI : prev_sep[day + 1][c1],
          day == D - 1 ? EMPTY_VI : next_sep[day + 1][c1],
          -diff
        );
        // clang-format on
        diff += new_pena1;
        // if (type < type_th_swap) {
        //   debug("diff swap:%d\n", diff);
        // } else {
        //   debug("diff move:%d\n", diff);
        // }
        if (diff <= 0) {
          debug("move day:%d col:%d w:%d diff:%d n:%lu\n", day, c0, sol.ws[c0], diff, sol.nis[day][c0].size());
          sol.hss[day][c0] = new_hs0;
          sol.hss[day][c1] = new_hs1;
          update_col(day, c0);
          update_col(day, c1);
          int real_pena = pena_sep[day][c0] + pena_sep[day + 1][c0] + pena_area[day][c0] + pena_sep[day][c1] + pena_sep[day + 1][c1] +
                          pena_area[day][c1];
          // debug("penas: %d %d %d %d %d\n", new_pena, pena_sep[day][c0], pena_sep[day + 1][c0], pena_area[day][c0], real_pena);
          assert((new_pena0 + new_pena1 - real_pena) % 100 == 0); // 面積順ソートで改善することがあるので面積ペナルティのみずれる
          diff += real_pena - new_pena0 - new_pena1;
          if (diff < 0) {
            pena += diff;
            best_sol = sol;
            debug("best_sol swap:%d turn:%d\n", pena, turn);
          }
        } else {
          if (type < type_th_swap) {
            swap(sol.nis[day][c0][p0], sol.nis[day][c1][p1]);
          } else {
            sol.nis[day][c0].insert(sol.nis[day][c0].begin() + p0, sol.nis[day][c1].back());
            sol.nis[day][c1].pop_back();
          }
        }
      }
    }
    best_sol.pena_area = 0;
    best_sol.pena_wall = 0;
    for (int i = 0; i < D; ++i) {
      best_sol.pena_area += accumulate(pena_area[i].begin(), pena_area[i].end(), 0);
      best_sol.pena_wall += accumulate(pena_sep[i].begin(), pena_sep[i].end(), 0);
    }
    debug("%lld %lld %lld %d\n", best_sol.pena_area, best_sol.pena_wall, best_sol.pena_area + best_sol.pena_wall, pena);
    assert(best_sol.pena_area + best_sol.pena_wall == pena);
    return best_sol;
  }

  pair<int, vi> improve_col(int day, vi nis, int w, const vi& sep_cnt_before, const vi& prev_sep_before, const vi& next_sep_before,
                            const vi& sep_cnt_after, const vi& prev_sep_after, const vi& next_sep_after, int threshold) {
    if (nis.empty()) {
      return make_pair(INF, vi());
    }
    shuffle(nis);
    static vvi dp(N + 1, vi(W + 1, INF));
    static vvi prev(N + 1, vi(W + 1, 0));
    dp[0][0] = 0;
    if (day != 0) {
      dp[0][0] += (nis.size() + sep_cnt_before[W] - 1) * w;
      if (next_sep_before[W] == -1) {
        dp[0][0] += w;
      }
    }
    if (day != D - 1) {
      dp[0][0] += (nis.size() + sep_cnt_after[W] - 1) * w;
      if (next_sep_after[W] == -1) {
        dp[0][0] += w;
      }
    }
    const int ub = threshold + dp[0][0];
    auto update_dp = [&](int i, int cy, int ny, int na, vi& next_vaild_pos) {
      int nv = dp[i][cy];
      int a = (ny - cy) * w;
      if (a < na) nv += (na - a) * 100;
      if (ny == W) {
        if (day != 0) nv -= w;
        if (day != D - 1) nv -= w;
      } else {
        if (day != 0 && sep_cnt_before[ny] != sep_cnt_before[ny - 1]) {
          nv -= w * 2;
        }
        if (day != D - 1 && sep_cnt_after[ny] != sep_cnt_after[ny - 1]) {
          nv -= w * 2;
        }
      }
      if (nv < dp[i + 1][ny] && nv < ub) {
        if (dp[i + 1][ny] == INF) {
          next_vaild_pos.push_back(ny);
        }
        dp[i + 1][ny] = nv;
        prev[i + 1][ny] = cy;
      }
    };

    static vi valid_pos;
    static vi next_valid_pos;
    valid_pos.assign(1, 0);
    for (int i = 0; i < nis.size(); ++i) {
      next_valid_pos.clear();
      int min_v = ub;
      int ai = nis[i];
      for (int y : valid_pos) {
        if (dp[i][y] >= min_v) {
          dp[i][y] = INF;
          continue;
        }
        min_v = dp[i][y];
        int jy = y + (A[day][ai] + w - 1) / w;
        if (jy < W) {
          update_dp(i, y, jy, A[day][ai], next_valid_pos);
          int ny = W;
          if (day != 0 && next_sep_before[jy] != -1) ny = next_sep_before[jy];
          if (day != D - 1 && next_sep_after[jy] != -1) ny = min(ny, next_sep_after[jy]);
          if (ny != W) update_dp(i, y, ny, A[day][ai], next_valid_pos);
        }
        if (jy != y + 1 && jy - 1 < W) {
          update_dp(i, y, jy - 1, A[day][ai], next_valid_pos);
        }
        if (i == nis.size() - 1) {
          update_dp(i, y, W, A[day][ai], next_valid_pos);
        }
        jy = min(jy, W);
        int py = 0;
        if (day != 0) py = prev_sep_before[jy];
        if (day != D - 1) py = max(py, prev_sep_after[jy]);
        if (py > y) {
          update_dp(i, y, py, A[day][ai], next_valid_pos);
        }
        dp[i][y] = INF;
      }
      swap(valid_pos, next_valid_pos);
      sort(valid_pos.begin(), valid_pos.end());
    }
    int ret = dp[nis.size()][W];
    if (ret >= ub) {
      return make_pair(ret, vi());
    }
    int best_y = W;
    if ((day == 0 || next_sep_before[W] == W) && ((day == D - 1 || next_sep_after[W] == W))) {
      // Wまで使うのが最適でないケースを考慮
      for (int y : valid_pos) {
        if (dp[nis.size()][y] < ret) {
          best_y = y;
          ret = dp[nis.size()][y];
        }
        dp[nis.size()][y] = INF;
      }
    } else {
      for (int y : valid_pos) {
        dp[nis.size()][y] = INF;
      }
    }
    int i = nis.size();
    vi hs;
    while (best_y != 0) {
      int py = prev[i][best_y];
      hs.push_back(best_y - py);
      best_y = py;
      i--;
    }
    reverse(hs.begin(), hs.end());
    return make_pair(ret, hs);
  }

  vi distribute_len(const vector<double>& ratio, int width) {
    double r_sum = accumulate(ratio.begin(), ratio.end(), 0.0);
    int len_sum = 0;
    vi ret(ratio.size());
    for (int i = 0; i < ratio.size(); ++i) {
      ret[i] = max(1, (int)(ratio[i] * width / r_sum));
      len_sum += ret[i];
    }
    for (int i = 0; len_sum < width; ++i) {
      ret[ratio.size() - 1 - i % ratio.size()]++;
      len_sum++;
    }
    for (int i = 0; len_sum > width; ++i) {
      if (ret[ratio.size() - 1 - i % ratio.size()] > 1) {
        ret[ratio.size() - 1 - i % ratio.size()]--;
        len_sum--;
      }
    }
    return ret;
  }

  int eval_height(int h) { return h < W ? W - h : pow((h - W) * 5, 2); }

  vvi distribute_area(const vector<int>& ws, const vector<int>& areas) {
    const int col = ws.size();
    const int n = areas.size();
    vi hs(col);
    vi ai(n);
    for (int i = n - 1; i >= 0; --i) {
      int pos = 0;
      int min_h = hs[0] + (areas[i] + ws[0] - 1) / ws[0];
      for (int j = 1; j < col; ++j) {
        int cur_h = hs[j] + (areas[i] + ws[j] - 1) / ws[j];
        if (cur_h < min_h || (cur_h == min_h && ws[pos] < ws[j])) {
          pos = j;
          min_h = cur_h;
        }
      }
      ai[i] = pos;
      hs[pos] += (areas[i] + ws[pos] - 1) / ws[pos];
    }
    if (*max_element(hs.begin(), hs.end()) > W) {
      vi vs(col);
      for (int i = 0; i < col; ++i) {
        vs[i] = eval_height(hs[i]);
      }
      // nisを高さが均等になるように詰める
      const int rep = (int)(5.0 / sqrt(E) * N);
      for (int i = 0; i < rep; ++i) {
        int p0 = rnd.next(n);
        int from = ai[p0];
        if ((i & 0x3) == 0) {
          int to = rnd.next(col - 1);
          if (from <= to) to++;
          int v0 = eval_height(hs[from] - (areas[p0] + ws[from] - 1) / ws[from]);
          int v1 = eval_height(hs[to] + (areas[p0] + ws[to] - 1) / ws[to]);
          int diff = v0 + v1 - vs[from] - vs[to];
          if (diff <= 0) {
            vs[from] = v0;
            vs[to] = v1;
            hs[from] -= (areas[p0] + ws[from] - 1) / ws[from];
            hs[to] += (areas[p0] + ws[to] - 1) / ws[to];
            ai[p0] = to;
          }
        } else {
          int p1 = rnd.next(n - 1);
          if (p0 <= p1) p1++;
          if (ai[p0] == ai[p1]) continue;
          int to = ai[p1];
          int h0_old = (areas[p0] + ws[from] - 1) / ws[from];
          int h0_new = (areas[p0] + ws[to] - 1) / ws[to];
          int h1_old = (areas[p1] + ws[to] - 1) / ws[to];
          int h1_new = (areas[p1] + ws[from] - 1) / ws[from];
          int v0 = eval_height(hs[from] + h1_new - h0_old);
          int v1 = eval_height(hs[to] + h0_new - h1_old);
          int diff = v0 + v1 - vs[from] - vs[to];
          if (diff <= 0) {
            vs[from] = v0;
            vs[to] = v1;
            hs[from] += h1_new - h0_old;
            hs[to] += h0_new - h1_old;
            ai[p0] = to;
            ai[p1] = from;
          }
        }
      }
    }
    vvi nis(col);
    for (int i = 0; i < n; ++i) {
      nis[ai[i]].push_back(i);
    }
    return nis;
  }

  Result solve_nomove(const vi& areas) {
    vi first_nis(N);
    iota(first_nis.begin(), first_nis.end(), 0);
    auto [best_pena, first_seps] = solve_nomove_column(first_nis, areas, W);
    debug("col:%d pena:%lld\n", 1, best_pena);
    vvi best_seps = {first_seps};
    vi best_ws = {W};
    for (int turn = 0; turn < 10 && best_pena > 1; ++turn) {
      for (int col = 2; col <= 4 && best_pena > 1; ++col) {
        vector<double> ratio(col);
        for (int i = 0; i < col; ++i) {
          ratio[i] = rnd.next(5.0) + 1.0;
        }
        vi ws = distribute_len(ratio, W);
        vvi nis = distribute_area(ws, areas);
        int64_t cur_pena = 0;
        vvi seps;
        for (int i = 0; i < col; ++i) {
          auto [pena, ss] = solve_nomove_column(nis[i], areas, ws[i]);
          cur_pena += pena;
          seps.push_back(ss);
        }
        debug("col:%d pena:%lld\n", col, cur_pena);
        if (cur_pena < best_pena) {
          best_pena = cur_pena;
          swap(best_seps, seps);
          swap(best_ws, ws);
        }
      }
    }
    vector<Rect> rects;
    int x = 0;
    for (int i = 0; i < best_seps.size(); ++i) {
      for (int j = 0; j < best_seps[i].size() - 1; ++j) {
        rects.emplace_back(best_seps[i][j], x, best_seps[i][j + 1], x + best_ws[i]);
      }
      x += best_ws[i];
    }
    sort(rects.begin(), rects.end());
    int64_t pena = 0;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < N; ++j) {
        if (rects[j].area() < A[i][j]) {
          pena += (A[i][j] - rects[j].area()) * 100;
        }
      }
    }
    debug("solve_nomove_score:%lld\n", pena);
    return Result(vector<vector<Rect>>(D, rects), pena, 0ll, best_ws.size());
  }

  pair<int64_t, vi> solve_nomove_column(const vi& nis, const vi& areas, int width) {
    int n = nis.size();
    if (n == 0) {
      vi sep = {0, W};
      return make_pair(0ll, sep);
    }
    if (n == 1) {
      vi sep = {0, W};
      int64_t pena = 0;
      for (int i = 0; i < D; ++i) {
        if (W * width < A[i][nis[0]]) {
          pena += A[i][nis[0]] - W * width;
        }
      }
      return make_pair(pena, sep);
    }
    vector<double> cur_areas;
    for (int i : nis) {
      cur_areas.push_back(areas[i]);
    }
    vi hs = distribute_len(cur_areas, W);
    vector<int64_t> pena(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < D; ++j) {
        if (hs[i] * width < A[j][nis[i]]) {
          pena[i] += A[j][nis[i]] - hs[i] * width;
        }
      }
    }
    int64_t best_pena = accumulate(pena.begin(), pena.end(), 0LL);
    debug("best_pena:%lld -> ", best_pena);
    vi best_hs = hs;
    for (int turn = 0; turn < n * 100; ++turn) {
      // 仕切りの位置を動かして山登り
      int p0 = rnd.next(n);
      int p1 = rnd.next(n - 1);
      if (p0 <= p1) p1++;
      if (hs[p1] == 1) continue;
      int64_t new_pena_0 = 0;
      int64_t new_pena_1 = 0;
      for (int i = 0; i < D; ++i) {
        if ((hs[p0] + 1) * width < A[i][nis[p0]]) {
          new_pena_0 += A[i][nis[p0]] - (hs[p0] + 1) * width;
        }
        if ((hs[p1] - 1) * width < A[i][nis[p1]]) {
          new_pena_1 += A[i][nis[p1]] - (hs[p1] - 1) * width;
        }
      }
      int64_t diff = new_pena_0 + new_pena_1 - pena[p0] - pena[p1];
      if (diff <= 0) {
        hs[p0]++;
        hs[p1]--;
        pena[p0] = new_pena_0;
        pena[p1] = new_pena_1;
        best_hs = hs;
        best_pena = accumulate(pena.begin(), pena.end(), 0LL);
        if (best_pena == 0) break;
      }
    }

    debug("%lld\n", best_pena);
    vi sep = {0};
    for (int i = 0; i < n; ++i) {
      sep.push_back(sep.back() + best_hs[i]);
    }
    assert(sep.back() == W);
    return make_pair(best_pena, sep);
  }

  tuple<vvi, int, int> pack_columns(int day, const vvi& nis, const vi& ws) {
    const int col = ws.size();
    vvi hss;
    int pena_area = 0;
    int pena_wall = 0;
    for (int i = 0; i < col; ++i) {
      if (nis[i].empty()) {
        hss.push_back(vi());
        continue;
      }
      int sum_h = 0;
      vi hs;
      for (int n : nis[i]) {
        int h = (A[day][n] + ws[i] - 1) / ws[i];
        sum_h += h;
        hs.push_back(h);
      }
      if (W < sum_h) {
        vector<pair<int, int>> rems;
        for (int j = 0; j < nis[i].size(); ++j) {
          int rem = A[day][nis[i][j]] % ws[i];
          if (rem == 0) rem = ws[i];
          rems.emplace_back(rem, j);
        }
        sort(rems.begin(), rems.end());
        for (int j = 0; sum_h > W; ++j) {
          if (hs[rems[j % hs.size()].second] > 1) {
            hs[rems[j % hs.size()].second]--;
            sum_h--;
          }
        }
        for (int j = 0; j < nis[i].size(); ++j) {
          int a = hs[j] * ws[i];
          if (a < A[day][nis[i][j]]) {
            pena_area += (A[day][nis[i][j]] - a) * 100;
          }
        }
      } else {
        hs.back() += W - sum_h;
      }
      hss.push_back(hs);
      pena_wall += (nis[i].size() - 1) * ws[i];
    }
    if (day != 0 && day != D - 1) pena_wall *= 2;
    return make_tuple(hss, pena_area, pena_wall);
  }

  void solve_noarea(FreeSolution& sol, ll tl) {
    const int max_col = clamp((int)sqrt(N * 3), 4, 9);
    debug("solve_noarea max_col:%d\n", max_col);
    for (int turn = 0;; ++turn) {
      for (int day = 0; day < D; ++day) {
        if (get_time() > tl) {
          debug("solve_noarea turn:%d pena_area:%lld pena_wall:%lld\n", turn, sol.sum_pena_area(), sol.sum_pena_wall());
          return;
        }
        for (int col = turn == 0 ? 1 : 2; col <= max_col; ++col) {
          solve_noarea_day(sol, day, col, turn);
        }
      }
    }
  }

  int count_match(const vi& ws0, const vi& ws1) {
    int p0 = 1;
    int p1 = 1;
    int x0 = ws0[0];
    int x1 = ws1[0];
    int match = 0;
    while (x0 < W && x1 < W) {
      if (x0 < x1) {
        x0 += ws0[p0++];
      } else if (x0 > x1) {
        x1 += ws1[p1++];
      } else {
        match++;
        x0 += ws0[p0++];
        x1 += ws1[p1++];
      }
    }
    return match;
  }

  void solve_noarea_day(FreeSolution& sol, int day, int col, int turn) {
    vi ws;
    vvi nis;
    if (col == 1) {
      ws.push_back(W);
      nis.push_back(vi(N));
      iota(nis[0].begin(), nis[0].end(), 0);
    } else {
      vector<double> ratio(col);
      for (int i = 0; i < col; ++i) {
        ratio[i] = rnd.next(5.0) + 1.0;
      }
      ws = distribute_len(ratio, W);
      nis = distribute_area(ws, vi(A[day].begin(), A[day].begin() + N));
    }
    auto [hss, pena_area, pena_wall] = pack_columns(day, nis, ws);
    pena_wall += W * (col - 1) * (day != 0 && day != D - 1 ? 2 : 1);
    int match_p0 = day == 0 ? 0 : count_match(sol.wss[day], sol.wss[day - 1]);
    int match_n0 = day == D - 1 ? 0 : count_match(sol.wss[day], sol.wss[day + 1]);
    int match_p1 = day == 0 ? 0 : count_match(ws, sol.wss[day - 1]);
    int match_n1 = day == D - 1 ? 0 : count_match(ws, sol.wss[day + 1]);
    pena_wall -= (match_p1 + match_n1) * W;
    int adj_diff = (match_p0 + match_n0 - match_p1 - match_n1) * W;
    if (pena_area + pena_wall + adj_diff < sol.pena_area[day] + sol.pena_wall[day]) {
      debug("turn:%d day:%d col:%d pena_area:%lld->%d pena_wall:%lld->%d pena_sum:%lld->%d\n", turn, day, col, sol.pena_area[day],
            pena_area, sol.pena_wall[day], pena_wall, sol.pena_area[day] + sol.pena_wall[day], pena_area + pena_wall);
      sol.pena_area[day] = pena_area;
      sol.pena_wall[day] = pena_wall;
      if (day != 0) {
        sol.pena_wall[day - 1] += (match_p0 - match_p1) * W;
      }
      if (day != D - 1) {
        sol.pena_wall[day + 1] += (match_n0 - match_n1) * W;
      }
      sol.wss[day] = ws;
      sol.rects[day].clear();
      int x = 0;
      for (int i = 0; i < col; ++i) {
        int y = 0;
        for (int j = 0; j < nis[i].size(); ++j) {
          sol.rects[day].emplace_back(y, x, y + hss[i][j], x + ws[i]);
          y += hss[i][j];
        }
        x += ws[i];
      }
      sort(sol.rects[day].begin(), sol.rects[day].end());
    }
  }

  FixColumnSolution solve_noarea_fixed_column(ll tl) {
    vector<vector<Rect>> best_rects;
    int best_pena_area = INF;
    int best_pena_wall = INF;
    vi best_ws;
    vvvi best_nis(D);
    vvvi best_hss(D);
    vvvi nis(D);
    vvvi hss(D);
    const int max_col = (N + 3) / 2;
    for (int t = 0;; ++t) {
      if (get_time() > tl - TL / 8) break;
      int lo_col = t < 10 ? 2 : max(2, (int)best_ws.size() - 1);
      int hi_col = t < 10 ? max_col : best_ws.size() + 1;
      for (int col = lo_col; col <= hi_col; ++col) {
        vector<double> ratio(col);
        for (int i = 0; i < col; ++i) {
          ratio[i] = rnd.next(18.0) + 1.0;
        }
        vi ws = distribute_len(ratio, W);
        int sum_pena_area = 0;
        int sum_pena_wall = 0;
        for (int day = 0; day < D && sum_pena_area + sum_pena_wall < best_pena_area + best_pena_wall; ++day) {
          nis[day] = distribute_area(ws, vi(A[day].begin(), A[day].begin() + N));
          auto [hs, pena_area, pena_wall] = pack_columns(day, nis[day], ws);
          hss[day] = hs;
          sum_pena_area += pena_area;
          sum_pena_wall += pena_wall;
        }
        if (sum_pena_area + sum_pena_wall < best_pena_area + best_pena_wall) {
          best_pena_area = sum_pena_area;
          best_pena_wall = sum_pena_wall;
          best_ws = ws;
          swap(best_nis, nis);
          swap(best_hss, hss);
          debug("pena_area:%d pena_wall:%d col:%d t:%d\n", best_pena_area, best_pena_wall, col, t);
        }
      }
    }
    const int col = best_ws.size();
    for (int t = 0;; ++t) {
      if (get_time() > tl) break;
      vi ws = best_ws;
      int c0 = rnd.next(col);
      int c1 = rnd.next(col - 1);
      if (c0 <= c1) c1++;
      if (ws[c0] <= 5) continue;
      int mv = rnd.next(ws[c0] / 2) + 1;
      ws[c0] -= mv;
      ws[c1] += mv;
      int sum_pena_area = 0;
      int sum_pena_wall = 0;
      for (int day = 0; day < D && sum_pena_area + sum_pena_wall < best_pena_area + best_pena_wall; ++day) {
        nis[day] = distribute_area(ws, vi(A[day].begin(), A[day].begin() + N));
        auto [hs, pena_area, pena_wall] = pack_columns(day, nis[day], ws);
        hss[day] = hs;
        sum_pena_area += pena_area;
        sum_pena_wall += pena_wall;
      }
      if (sum_pena_area + sum_pena_wall < best_pena_area + best_pena_wall) {
        best_pena_area = sum_pena_area;
        best_pena_wall = sum_pena_wall;
        best_ws = ws;
        swap(best_nis, nis);
        swap(best_hss, hss);
        debug("pena_area:%d pena_wall:%d col:%d t:%d\n", best_pena_area, best_pena_wall, col, t);
      }
    }
    debug("solve_noarea_fixed_column:%d %d %d\n", best_pena_area + best_pena_wall, best_pena_area, best_pena_wall);
    return FixColumnSolution(best_ws, best_nis, best_hss, best_pena_area, best_pena_wall);
  }
};

int main() {
  ll tl = TL;
#ifdef DEBUG
  char* env = getenv("TL");
  if (env) {
    stringstream ss(env);
    ss >> tl;
  }
#endif
  int _;
  scanf("%d %d %d", &_, &D, &N);
  int area_sum = 0;
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < N; ++j) {
      scanf("%d", &A[i][j]);
      area_sum += A[i][j];
    }
  }
  E = 1.0 * (W * W * D - area_sum) / (W * W * D);
  debug("D:%d N:%d E:%.4f\n", D, N, E);
  auto solver = make_unique<Solver>(start_time + tl);
  Result res = solver->solve();
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d %d %d %d\n", res.rects[i][j].top, res.rects[i][j].left, res.rects[i][j].bottom, res.rects[i][j].right);
    }
  }
  PRINT_TIMER();
  PRINT_COUNTER();
  debug("score:%8lld pena_area:%8lld pena_wall:%8lld col:%d\n", res.score(), res.pena_area, res.pena_wall, res.col);
}
