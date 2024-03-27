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
constexpr ll TL = 2500;
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

template <class T> using ar_t = array<array<T, 20>, 20>;

ar_t<uint64_t> cell_hash;

struct Point {
  int y, x;

  bool operator==(const Point& p) const { return y == p.y && x == p.x; }
  bool operator!=(const Point& p) const { return y != p.y || x != p.x; }
  bool operator<(const Point& p) const { return y == p.y ? x < p.x : y < p.y; }
};

struct Rect {
  int top, left, bottom, right;
  int area() const { return (bottom - top) * (right - left); }
};

struct Result {
  vector<vector<Rect>> rects;
  int64_t pena_area;
  int64_t pena_wall;

  Result(vector<vector<Rect>> rects_, int64_t pena_area_, int64_t pena_wall_)
      : rects(rects_), pena_area(pena_area_), pena_wall(pena_wall_) {}

  int64_t score() const { return pena_area + pena_wall + 1; }
};

Result RESULT_EMPTY(vector<vector<Rect>>(), 1 << 29, 1 << 29);

constexpr int INF = 1 << 29;
constexpr int W = 1000;
int D;
int N;
double E;
array<array<int, 50>, 50> A;

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
    Result best_result = RESULT_EMPTY;
    vi max_areas(N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        max_areas[i] = max(max_areas[i], A[j][i]);
      }
    }
    int sum_max_area = accumulate(max_areas.begin(), max_areas.end(), 0);
    debug("sum_max_area:%d\n", sum_max_area);
    if (sum_max_area <= W * W + 200) {
      START_TIMER(0);
      best_result = solve_nomove(max_areas);
      STOP_TIMER(0);
      if (best_result.score() == 1) return best_result;
    }

    START_TIMER(1);
    Result noarea_result = solve_noarea();
    STOP_TIMER(1);
    if (noarea_result.score() < best_result.score()) {
      best_result = noarea_result;
      best_score = best_result.score();
    }
    START_TIMER(2);
    Result noarea2_result = solve_noarea2();
    STOP_TIMER(2);
    if (noarea2_result.score() < best_result.score()) {
      best_result = noarea2_result;
      best_score = best_result.score();
    }

    int turn = 0;
    int max_col = ceil(sqrt(D) * 1.5);
    while (true) {
      for (int col = 1; col <= max_col; ++col) {
        if (timelimit < get_time()) {
          debug("turn:%d\n", turn * max_col + col - 1);
          return best_result;
        }
        if (turn > 0 && col == 1) continue;
        double max_ratio = rnd.next(40.0) + 10.0;
        double amp = rnd.next(1.0) + 1.0;
        vector<double> ratio = {1.0};
        for (int i = 1; i < col; ++i) {
          ratio.push_back(min(max_ratio, ratio.back() * amp));
        }
        vi lens = distribute_len(ratio, W);
        vi xs = {0};
        for (int i = 0; i < col; ++i) {
          xs.push_back(xs.back() + lens[i]);
        }
        debug_vec(xs, "xs");
        Result res = solve_cols(xs);
        debug("score:%lld col:%d max_ratio:%f amp:%f\n", res.score(), col, max_ratio, amp);
        if (res.score() < best_result.score()) {
          best_result = res;
          best_score = res.score();
        }
        // return best_result;
      }
      turn++;
    }
    return best_result;
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
      for (int j = 1; j < col; ++j) {
        if (hs[j] < hs[pos] || (hs[j] == hs[pos] && ws[pos] < ws[j])) {
          pos = j;
        }
      }
      ai[i] = pos;
      hs[pos] += (areas[i] + ws[pos] - 1) / ws[pos];
    }
    vi vs(col);
    for (int i = 0; i < col; ++i) {
      vs[i] = eval_height(hs[i]);
    }
    // nisを高さが均等になるように詰める
    const int rep = (int)(5.0 / sqrt(E) * N);
    for (int i = 0; i < rep; ++i) {
      int p0 = rnd.next(n);
      int from = ai[p0];
      if ((i & 3) == 0) {
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
    // debug_vec(hs, "hs");
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
    sort(rects.begin(), rects.end(), [](const Rect& r1, const Rect& r2) { return r1.area() < r2.area(); });
    int64_t pena = 0;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < N; ++j) {
        if (rects[j].area() < A[i][j]) {
          pena += (A[i][j] - rects[j].area()) * 100;
        }
      }
    }
    debug("solve_nomove_score:%lld\n", pena);
    return Result(vector<vector<Rect>>(D, rects), pena, 0ll);
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

  Result solve_noarea() {
    vector<vector<Rect>> rects;
    int64_t sum_pena_area = 0;
    int64_t sum_pena_wall = 0;
    for (int day = 0; day < D; ++day) {
      int best_pena_area = INF;
      int best_pena_wall = INF;
      int best_col = 0;
      vector<Rect> best_rect;
      for (int t = 0; t < 10; ++t) {
        for (int col = t == 0 ? 1 : 2; col <= clamp((int)sqrt(N * 3), 4, 9); ++col) {
          vector<double> ratio(col);
          for (int i = 0; i < col; ++i) {
            ratio[i] = t == 0 ? 1.0 : (rnd.next(5.0) + 1.0);
          }
          vi ws;
          vvi nis;
          if (col == 1) {
            ws.push_back(W);
            nis.assign(1, vi(N));
            iota(nis[0].begin(), nis[0].end(), 0);
          } else {
            ws = distribute_len(ratio, W);
            nis = distribute_area(ws, vi(A[day].begin(), A[day].begin() + N));
          }
          vvi hss;
          int pena_area = 0;
          int pena_wall = W * (col - 1);
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
            if (sum_h < W) {
              hs.back() += W - sum_h;
            }
            for (int j = 0; j < nis[i].size(); ++j) {
              int a = hs[j] * ws[i];
              // debug("%d %d %d\n", nis[i][j], a, A[day][nis[i][j]]);
              if (a < A[day][nis[i][j]]) {
                pena_area += (A[day][nis[i][j]] - a) * 100;
              }
            }
            hss.push_back(hs);
            pena_wall += (nis[i].size() - 1) * ws[i];
          }
          if (day != 0 && day != D - 1) pena_wall *= 2;
          if (pena_area + pena_wall < best_pena_area + best_pena_wall) {
            best_pena_area = pena_area;
            best_pena_wall = pena_wall;
            best_col = col;
            best_rect.clear();
            int x = 0;
            for (int i = 0; i < col; ++i) {
              int y = 0;
              for (int j = 0; j < nis[i].size(); ++j) {
                best_rect.emplace_back(y, x, y + hss[i][j], x + ws[i]);
                y += hss[i][j];
              }
              x += ws[i];
            }
            sort(best_rect.begin(), best_rect.end(), [](const Rect& r1, const Rect& r2) { return r1.area() < r2.area(); });
          }
        }
      }
      sum_pena_area += best_pena_area;
      sum_pena_wall += best_pena_wall;
      debug("day:%d pena_area:%d pena_wall:%d col:%d\n", day, best_pena_area, best_pena_wall, best_col);
      rects.push_back(best_rect);
    }
    debug("solve_noarea:%lld %lld %lld\n", sum_pena_area + sum_pena_wall, sum_pena_area, sum_pena_wall);
    return Result(rects, sum_pena_area, sum_pena_wall);
  }

  Result solve_noarea2() {
    vector<vector<Rect>> best_rects;
    int best_pena_area = INF;
    int best_pena_wall = INF;
    int best_col = 0;
    for (int t = 0; t < max(10, 100000 / (D * N)); ++t) {
      for (int col = 2; col <= clamp((int)sqrt(N * 3), 4, 7); ++col) {
        vector<double> ratio(col);
        for (int i = 0; i < col; ++i) {
          ratio[i] = rnd.next(5.0) + 1.0;
        }
        vi ws = distribute_len(ratio, W);
        int sum_pena_area = 0;
        int sum_pena_wall = 0;
        vector<vector<Rect>> rects(D);
        for (int day = 0; day < D && sum_pena_area + sum_pena_wall < best_pena_area + best_pena_wall; ++day) {
          vvi nis = distribute_area(ws, vi(A[day].begin(), A[day].begin() + N));
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
            if (sum_h < W) {
              hs.back() += W - sum_h;
            }
            for (int j = 0; j < nis[i].size(); ++j) {
              int a = hs[j] * ws[i];
              // debug("%d %d %d\n", nis[i][j], a, A[day][nis[i][j]]);
              if (a < A[day][nis[i][j]]) {
                pena_area += (A[day][nis[i][j]] - a) * 100;
              }
            }
            hss.push_back(hs);
            pena_wall += (nis[i].size() - 1) * ws[i];
          }
          if (day != 0 && day != D - 1) pena_wall *= 2;
          sum_pena_area += pena_area;
          sum_pena_wall += pena_wall;
          rects[day].clear();
          int x = 0;
          for (int i = 0; i < col; ++i) {
            int y = 0;
            for (int j = 0; j < nis[i].size(); ++j) {
              rects[day].emplace_back(y, x, y + hss[i][j], x + ws[i]);
              y += hss[i][j];
            }
            x += ws[i];
          }
        }
        if (sum_pena_area + sum_pena_wall < best_pena_area + best_pena_wall) {
          best_pena_area = sum_pena_area;
          best_pena_wall = sum_pena_wall;
          best_col = col;
          best_rects = rects;
          for (int day = 0; day < D; ++day) {
            sort(best_rects[day].begin(), best_rects[day].end(), [](const Rect& r1, const Rect& r2) { return r1.area() < r2.area(); });
          }
          debug("pena_area:%d pena_wall:%d col:%d t:%d\n", best_pena_area, best_pena_wall, best_col, t);
        }
      }
    }
    debug("solve_noarea2:%d %d %d\n", best_pena_area + best_pena_wall, best_pena_area, best_pena_wall);
    return Result(best_rects, best_pena_area, best_pena_wall);
  }
  Result solve_cols(const vi& xs) {
    const int col = xs.size() - 1;
    vector<vector<double>> dp(col, vector<double>(N + 1, 1e99));
    vector<vector<int>> prev(col, vector<int>(N + 1, 1 << 29));
    vector<int64_t> acc(N + 1, 0);
    for (int i = 0; i < N; ++i) {
      acc[i + 1] = acc[i] + A[0][i];
    }
    for (int i = 0; i <= N; ++i) {
      double h = acc[i] / xs[1];
      dp[0][i] = (h <= W ? W - h : W + pow(50 * (h - W), 2));
    }
    for (int i = 1; i < col; ++i) {
      for (int j = 0; j <= N; ++j) {
        for (int k = 0; k <= j; ++k) {
          double h = 1.0 * (acc[j] - acc[k]) / (xs[i + 1] - xs[i]);
          double nv = dp[i - 1][k] + (h <= W ? W - h : W + pow(50 * (h - W), 2));
          if (nv < dp[i][j]) {
            dp[i][j] = nv;
            prev[i][j] = k;
          }
        }
      }
    }
    vi nis = {N};
    for (int i = col - 1; i > 0; --i) {
      nis.push_back(prev[i][nis.back()]);
    }
    nis.push_back(0);
    reverse(nis.begin(), nis.end());
    for (int i = 0; i < col; ++i) {
      debug("%d %.1f\n", nis[i + 1], 1.0 * (acc[nis[i + 1]] - acc[nis[i]]) / (xs[i + 1] - xs[i]));
    }
    vector<int64_t> penas(D);
    vvvi seps(D, vvi(col));
    for (int i = 0; i < col; ++i) {
      if (nis[i] == nis[i + 1]) return RESULT_EMPTY;
      seps[0][i] = place_separator(vi(A[0].begin() + nis[i], A[0].begin() + nis[i + 1]));
    }
    penas[0] = initial_day_area_pena(seps[0], xs);
    for (int i = 1; i < D; ++i) {
      auto res = solve_single_day(i, seps[i - 1], xs);
      if (res.second == INF) {
        return RESULT_EMPTY;
      }
      seps[i] = res.first;
      penas[i] = res.second;
    }
    debug("sum_pena:%lld\n", accumulate(penas.begin(), penas.end(), 0LL));
    if (accumulate(penas.begin(), penas.end(), 0LL) < best_score * 3) {
      for (int t = 0; t < 2; ++t) {
        reverse(seps.begin(), seps.end());
        reverse(penas.begin(), penas.end());
        reverse(A.begin(), A.begin() + D);
        penas[0] = initial_day_area_pena(seps[0], xs);
        for (int i = 1; i < D; ++i) {
          auto res = solve_single_day(i, seps[i - 1], xs);
          if (res.second == INF) {
            if (t == 0) {
              reverse(seps.begin(), seps.end());
              reverse(penas.begin(), penas.end());
              reverse(A.begin(), A.begin() + D);
            }
            return RESULT_EMPTY;
          }
          seps[i] = res.first;
          penas[i] = res.second;
        }
        debug("sum_pena:%lld\n", accumulate(penas.begin(), penas.end(), 0LL));
      }
    }

    int64_t wall_cost = accumulate(penas.begin(), penas.end(), 0LL);
    int64_t area_cost = 0;
    vector<vector<Rect>> rects(D);
    for (int day = 0; day < D; ++day) {
      int ai = 0;
      for (int c = 0; c < col; ++c) {
        int left = xs[c];
        int right = xs[c + 1];
        for (int i = 0; i < seps[day][c].size() - 1; ++i) {
          int top = seps[day][c][i];
          int bottom = seps[day][c][i + 1];
          rects[day].emplace_back(top, left, bottom, right);
          int a = (bottom - top) * (right - left);
          if (a < A[day][ai]) {
            wall_cost -= (A[day][ai] - a) * 100;
          }
          ai++;
        }
      }
      sort(rects[day].begin(), rects[day].end(), [](const Rect& r1, const Rect& r2) { return r1.area() < r2.area(); });
      for (int i = 0; i < N; ++i) {
        if (rects[day][i].area() < A[day][i]) {
          area_cost += (A[day][i] - rects[day][i].area()) * 100;
        }
      }
    }
    return Result(rects, area_cost, wall_cost);
  }

  int64_t initial_day_area_pena(const vvi& sep, const vi& xs) {
    int64_t pena = 0;
    int ai = 0;
    for (int i = 0; i < sep.size(); ++i) {
      for (int j = 0; j < sep[i].size() - 1; ++j) {
        int area = (sep[i][j + 1] - sep[i][j]) * (xs[i + 1] - xs[i]);
        if (area < A[0][ai]) {
          pena += (A[0][ai] - area) * 100;
        }
        ai++;
      }
    }
    return pena;
  }

  pair<vvi, int> solve_single_day(int day, const vvi& prev_sep, const vi& xs) {
    // debug("solve_single_day:%d\n", day);
    const int FAIL = (int)min((int64_t)10000000ll, best_score);
    const int col = prev_sep.size();
    vector<vvi> dp(col, vvi(N + 1, vi(W + 1, INF)));
    vector<vvi> prev(col, vvi(N + 1, vi(W + 1, -1)));
    dp[0][0][0] = 0;
    static vi bottom_sep(W + 1);
    static vi top_sep(W + 1);
    static vi sep_idx(W + 1);
    for (int c = 0; c < col; ++c) {
      // debug("c:%d\n", c);
      for (int i = 1; i < prev_sep[c].size(); ++i) {
        fill(bottom_sep.begin() + prev_sep[c][i - 1] + 1, bottom_sep.begin() + prev_sep[c][i] + 1, prev_sep[c][i]);
        fill(top_sep.begin() + prev_sep[c][i - 1], top_sep.begin() + prev_sep[c][i], prev_sep[c][i - 1]);
        fill(sep_idx.begin() + prev_sep[c][i - 1], sep_idx.begin() + prev_sep[c][i], i - 1);
      }
      bottom_sep[0] = 0;
      top_sep[W] = W;
      sep_idx[W] = prev_sep[c].size() - 1;
      const int width = xs[c + 1] - xs[c];
      auto update = [&](int i, int y, int bottom) {
        // debug("update:%d %d %d %d\n", i, y, bottom, dp[c][i][y]);
        int nv = dp[c][i][y];
        const bool on_sep = bottom_sep[bottom] == bottom;
        if (!on_sep) nv += width;
        int over = sep_idx[bottom] - sep_idx[y] - on_sep;
        if (over > 0) nv += over * width;
        if ((bottom - y) * width < A[day][i]) {
          nv += (A[day][i] - (bottom - y) * width) * 100;
        }
        if (nv < dp[c][i + 1][bottom]) {
          // debug("up:%d -> %d\n", dp[c][i + 1][bottom], nv);
          dp[c][i + 1][bottom] = nv;
          prev[c][i + 1][bottom] = y;
        }
      };
      for (int i = 0; i < N; ++i) {
        int h = (A[day][i] + width - 1) / width;
        // debug("i:%d h:%d\n", i, h);
        for (int y = 0; y < W; ++y) {
          if (dp[c][i][y] >= FAIL) continue;
          // ぴったり
          if (y + h < W) {
            update(i, y, y + h);
          }
          // ぴったり-1
          if (h > 1 && y + h - 1 < W) {
            update(i, y, y + h - 1);
          }
          // 前の線
          if (y + h < W && top_sep[y + h] != y + h && y < top_sep[y + h]) {
            update(i, y, top_sep[y + h]);
          }
          // 次の線
          if (y + h < W && bottom_sep[y + h] != y + h && bottom_sep[y + h] != W) {
            update(i, y, bottom_sep[y + h]);
          }
          // 最後
          update(i, y, W);
        }
      }
      if (c < col - 1) {
        for (int i = 0; i < N; ++i) {
          dp[c + 1][i][0] = dp[c][i][W];
        }
      }
    }
    // debug("dp_value:%d\n", dp[col - 1][N][W]);
    if (dp[col - 1][N][W] >= FAIL) {
      debugStr("fail\n");
      return make_pair(vvi(), INF);
    }
    vvi ret(col);
    int n = N;
    for (int c = col - 1; c >= 0; --c) {
      ret[c].push_back(W);
      int y = W;
      while (y > 0) {
        y = prev[c][n][y];
        --n;
        ret[c].push_back(y);
      }
      reverse(ret[c].begin(), ret[c].end());
    }
    assert(n == 0);
    return make_pair(ret, dp[col - 1][N][W]);
  }

  vi place_separator(const vi& areas) {
    int sum = accumulate(areas.begin(), areas.end(), 0);
    vi ret = {0};
    int left = W;
    for (int a : areas) {
      ret.push_back(max(1, a * W / sum));
      left -= ret.back();
    }
    for (int i = 0; left > 0; ++i) {
      ret[areas.size() - i % areas.size()]++;
      left--;
    }
    for (int i = 0; left < 0; ++i) {
      if (ret[areas.size() - i % areas.size()] == 1) continue;
      ret[areas.size() - i % areas.size()]--;
      left++;
    }
    for (int i = 1; i < ret.size(); ++i) {
      ret[i] += ret[i - 1];
    }
    return ret;
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
  debug("score:%8lld pena_area:%8lld pena_wall:%8lld\n", res.score(), res.pena_area, res.pena_wall);
}
