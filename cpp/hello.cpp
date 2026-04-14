#include <iostream>
#include <algorithm>
#include <cstdint>
using namespace std;

bool solve(long long n, long long &a, long long &b, long long &c) {
    // 奇数直接无解
    if (n % 2 == 1) return false;
    // N=2 特殊处理
    if (n == 2) return false;

    // 方法1：模式匹配
    struct Pattern {
        long long a, b, c;
    };
    Pattern patterns[] = {
        {1, n, n-1},
        {n-1, n, n+1},
        {n/2, n, n/2},
        {n/2 - 1, n, n/2 + 1},
        {1, n-1, n},
        {2, n-2, n}
    };
    int patternCnt = sizeof(patterns) / sizeof(patterns[0]);
    for (int i = 0; i < patternCnt; ++i) {
        long long aa = patterns[i].a;
        long long bb = patterns[i].b;
        long long cc = patterns[i].c;
        if (aa > 0 && bb > 0 && cc > 0 && aa != bb && aa != cc && bb != cc) {
            if (aa + bb + cc == 2 * n && (aa ^ bb ^ cc) == n) {
                a = aa; b = bb; c = cc;
                return true;
            }
        }
    }

    // 方法2：智能枚举
    long long limit = min(100LL, n);
    for (long long aa = 1; aa <= limit; ++aa) {
        long long candidates_b[] = {n, n - aa, n + aa, n / 2, aa, 2 * aa};
        int cbCnt = sizeof(candidates_b) / sizeof(candidates_b[0]);
        for (int i = 0; i < cbCnt; ++i) {
            long long bb = candidates_b[i];
            if (bb > 0 && bb != aa && bb < 2 * n) {
                long long cc = 2 * n - aa - bb;
                if (cc > 0 && cc != aa && cc != bb) {
                    if ((aa ^ bb ^ cc) == n) {
                        a = aa; b = bb; c = cc;
                        return true;
                    }
                }
            }
        }
    }

    // 方法3：位运算构造
    if (n >= 4) {
        int bit_len = 64 - __builtin_clzll(static_cast<unsigned long long>(n));
        for (int i = 0; i < bit_len; ++i) {
            long long mask = 1LL << i;
            // 组合1
            long long aa = mask;
            long long bb = n;
            long long cc = 2 * n - aa - bb;
            if (cc > 0 && aa != bb && aa != cc && bb != cc) {
                if ((aa ^ bb ^ cc) == n) {
                    a = aa; b = bb; c = cc;
                    return true;
                }
            }
            // 组合2
            aa = n ^ mask;
            if (aa > 0) {
                bb = n;
                cc = 2 * n - aa - bb;
                if (cc > 0 && aa != bb && aa != cc && bb != cc) {
                    if ((aa ^ bb ^ cc) == n) {
                        a = aa; b = bb; c = cc;
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    while (T--) {
        long long n;
        cin >> n;
        long long a, b, c;
        if (solve(n, a, b, c)) {
            cout << a << " " << b << " " << c << "\n";
        } else {
            cout << -1 << "\n";
        }
    }

    return 0;
}