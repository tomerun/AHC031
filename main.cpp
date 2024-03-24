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
// #define MEASURE_TIME
#define OUTPUT_FOR_VIS
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

void debug_vec(const vi& vec) {
  debugStr("[");
  for (int i = 0; i < vec.size(); ++i) {
    debug("%d ", vec[i]);
  }
  debugStr("]");
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

constexpr int INF = 1 << 30;
constexpr array<int, 4> DY = {1, 0, -1, 0};
constexpr array<int, 4> DX = {0, 1, 0, -1};
constexpr int W = 1000;
int D;
int N;
array<array<int, 50>, 50> A;

bool accept(int64_t diff, double cooler) {
  if (diff <= 0) return true;
  double v = -diff * cooler;
  return rnd.next(1.0) < exp(v);
}

struct Solver {
  const ll timelimit;

  Solver(ll timelimit_) : timelimit(timelimit_) {}

  Result solve() {
    Result best_result = RESULT_EMPTY;
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
        debug("col:%d max_ratio:%f amp:%f\n", col, max_ratio, amp);
        vector<double> ws = {1.0};
        for (int i = 1; i < col; ++i) {
          ws.push_back(min(max_ratio, ws.back() * amp));
        }
        double sum = accumulate(ws.begin(), ws.end(), 0.0);
        for (int i = 0; i < col; ++i) {
          ws[i] = floor(ws[i] * W / sum);
        }
        sum = accumulate(ws.begin(), ws.end(), 0.0);
        for (int i = 0; sum < W; ++i) {
          ws[col - 1 - i % col] += 1;
          sum += 1;
        }
        vi xs = {0};
        for (int i = 0; i < col; ++i) {
          xs.push_back(xs.back() + (int)ws[i]);
        }
        debugStr("xs:");
        debug_vec(xs);
        debugln();
        Result res = solve_cols(xs);
        debug("score:%lld cols:%d\n", res.score(), col);
        if (res.score() < best_result.score()) {
          best_result = res;
        }
        // return best_result;
      }
      turn++;
    }
    return best_result;
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
      dp[0][i] = (h <= W ? h : W + pow(50 * (h - W), 2));
    }
    for (int i = 1; i < col; ++i) {
      for (int j = 0; j <= N; ++j) {
        for (int k = 0; k <= j; ++k) {
          double h = 1.0 * (acc[j] - acc[k]) / (xs[i + 1] - xs[i]);
          double nv = dp[i - 1][k] + (h <= W ? h : W + pow(50 * (h - W), 2));
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
    int64_t pena = 0;
    vvvi seps(D, vvi(col));
    for (int i = 0; i < col; ++i) {
      if (nis[i] == nis[i + 1]) return RESULT_EMPTY;
      seps[0][i] = place_separator(vi(A[0].begin() + nis[i], A[0].begin() + nis[i + 1]));
      for (int j = nis[i]; j < nis[i + 1]; ++j) {
        int area = (seps[0][i][j - nis[i] + 1] - seps[0][i][j - nis[i]]) * (xs[i + 1] - xs[i]);
        if (area < A[0][j]) {
          pena += (A[0][j] - area) * 100;
        }
      }
    }
    debug("pena_first_day:%lld\n", pena);

    for (int i = 1; i < D; ++i) {
      auto res = solve_single_day(i, seps[i - 1], xs);
      if (res.second == INF) {
        return RESULT_EMPTY;
      }
      seps[i] = res.first;
      pena += res.second;
    }

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
            area_cost += (A[day][ai] - a) * 100;
          }
          ai++;
        }
      }
    }
    return Result(rects, area_cost, pena - area_cost);
  }

  pair<vvi, int> solve_single_day(int day, const vvi& prev_sep, const vi& xs) {
    debug("solve_single_day:%d\n", day);
    constexpr int FAIL = 10000000;
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
          if (dp[c][i][y] > FAIL) continue;
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
    debug("dp_value:%d\n", dp[col - 1][N][W]);
    if (dp[col - 1][N][W] > FAIL) {
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
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < N; ++j) {
      scanf("%d", &A[i][j]);
    }
  }
  debug("D:%d N:%d\n", D, N);
  auto solver = make_unique<Solver>(start_time + tl);
  Result res = solver->solve();
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%d %d %d %d\n", res.rects[i][j].top, res.rects[i][j].left, res.rects[i][j].bottom, res.rects[i][j].right);
    }
  }
  debug("score:%lld pena_area:%lld pena_wall:%lld\n", res.score(), res.pena_area, res.pena_wall);
  PRINT_TIMER();
  PRINT_COUNTER();
}
