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

constexpr int64_t INF = 1ll << 60;
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
        double amp = rnd.next(0.7) + 1.4;
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
      }
      turn++;
      // break;
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
    vvi seps(col);
    for (int i = 0; i < col; ++i) {
      if (nis[i] == nis[i + 1]) return RESULT_EMPTY;
      seps[i] = place_separator(vi(A[0].begin() + nis[i], A[0].begin() + nis[i + 1]));
    }

    vector<Rect> rects;
    for (int i = 0; i < col; ++i) {
      for (int j = nis[i]; j < nis[i + 1]; ++j) {
        int top = seps[i][j - nis[i]];
        int bottom = seps[i][j + 1 - nis[i]];
        rects.emplace_back(Rect{top, xs[i], bottom, xs[i + 1]});
      }
    }
    int64_t area_cost = 0;
    for (int i = 0; i < N; ++i) {
      int a = (rects[i].bottom - rects[i].top) * (rects[i].right - rects[i].left);
      for (int j = 0; j < D; ++j) {
        if (a < A[j][i]) {
          area_cost += 100 * (A[j][i] - a);
        }
      }
    }

    return Result(vector<vector<Rect>>(D, rects), area_cost, 0ll);
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
