// Heuristic handwritten digit recognizer for 28x28 grayscale images in [0,1]
// Implements judge(IMAGE_T&) returning 0..9 using simple geometric features.

#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstdint>

typedef std::vector<std::vector<double>> IMAGE_T;

namespace nr_heur {

struct BBox { int r0, c0, r1, c1; int h() const { return std::max(0, r1 - r0 + 1); } int w() const { return std::max(0, c1 - c0 + 1); } };

static int H = 28, W = 28;

static double clamp01(double x){ return x < 0 ? 0 : (x > 1 ? 1 : x); }

static double otsu_threshold(const IMAGE_T &img){
    // Build 256-bin histogram over [0,1]
    const int bins = 256;
    std::vector<double> hist(bins, 0.0);
    double total = 0.0;
    for (const auto &row : img) {
        for (double v : row) {
            int b = (int)std::floor(clamp01(v) * (bins - 1));
            hist[b] += 1.0;
            total += 1.0;
        }
    }
    if (total <= 0.0) return 0.5;
    // Normalize
    for (auto &h : hist) h /= total;
    std::vector<double> omega(bins, 0.0), mu(bins, 0.0);
    omega[0] = hist[0];
    mu[0] = 0.0 * hist[0];
    for (int i = 1; i < bins; ++i) {
        omega[i] = omega[i-1] + hist[i];
        mu[i] = mu[i-1] + hist[i] * i;
    }
    double mu_t = mu[bins-1];
    double max_sigma = -1.0; int best_t = bins/2;
    for (int t = 0; t < bins-1; ++t) {
        if (omega[t] <= 1e-9 || (1 - omega[t]) <= 1e-9) continue;
        double mu0 = mu[t] / omega[t];
        double mu1 = (mu_t - mu[t]) / (1 - omega[t]);
        double sigma_b = omega[t] * (1 - omega[t]) * (mu0 - mu1) * (mu0 - mu1);
        if (sigma_b > max_sigma) { max_sigma = sigma_b; best_t = t; }
    }
    return (double)best_t / (bins - 1);
}

static std::vector<std::vector<uint8_t>> binarize(const IMAGE_T &img){
    H = (int)img.size();
    W = H ? (int)img[0].size() : 0;
    double t = otsu_threshold(img);
    // Slight bias to prefer foreground being bright digits
    double thr = std::max(0.3, std::min(0.7, t * 0.9));
    std::vector<std::vector<uint8_t>> bw(H, std::vector<uint8_t>(W, 0));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            bw[i][j] = (img[i][j] > thr) ? 1 : 0;
        }
    }
    return bw;
}

static BBox bounding_box(const std::vector<std::vector<uint8_t>> &bw){
    int r0 = H, c0 = W, r1 = -1, c1 = -1;
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) if (bw[i][j]) {
        r0 = std::min(r0, i); c0 = std::min(c0, j);
        r1 = std::max(r1, i); c1 = std::max(c1, j);
    }
    if (r1 < r0 || c1 < c0) return {0,0,-1,-1};
    return {r0,c0,r1,c1};
}

static int count_area(const std::vector<std::vector<uint8_t>> &bw, const BBox &b){
    if (b.r1 < b.r0) return 0;
    int a = 0; for (int i=b.r0;i<=b.r1;++i) for (int j=b.c0;j<=b.c1;++j) a+=bw[i][j]; return a;
}

static int count_holes(const std::vector<std::vector<uint8_t>> &bw, const BBox &b){
    if (b.r1 < b.r0) return 0;
    int h = b.h(), w = b.w();
    std::vector<std::vector<uint8_t>> seen(h, std::vector<uint8_t>(w, 0));
    auto inb = [&](int r,int c){return r>=0&&r<h&&c>=0&&c<w;};
    auto is_bg = [&](int r,int c){ return bw[b.r0+r][b.c0+c]==0; };
    int dr[4]={-1,1,0,0}; int dc[4]={0,0,-1,1};
    int components_bg = 0; int touch_border = 0;
    for (int i=0;i<h;++i) for (int j=0;j<w;++j) if (!seen[i][j] && is_bg(i,j)){
        components_bg++;
        bool touches = false;
        std::queue<std::pair<int,int>>q; q.push({i,j}); seen[i][j]=1;
        while(!q.empty()){
            auto [r,c]=q.front(); q.pop();
            if (r==0||c==0||r==h-1||c==w-1) touches = true;
            for (int k=0;k<4;++k){int nr=r+dr[k], nc=c+dc[k]; if (inb(nr,nc)&&!seen[nr][nc]&&is_bg(nr,nc)){seen[nr][nc]=1; q.push({nr,nc});}}
        }
        if (touches) touch_border++;
    }
    // background components not touching border are holes
    int holes = components_bg - touch_border;
    if (holes < 0) holes = 0;
    return holes;
}

static void projections(const std::vector<std::vector<uint8_t>> &bw, const BBox &b,
                        std::vector<int> &hp, std::vector<int> &vp){
    hp.assign(b.h(), 0); vp.assign(b.w(), 0);
    for (int i=b.r0;i<=b.r1;++i){
        for (int j=b.c0;j<=b.c1;++j){
            if (bw[i][j]){ hp[i-b.r0]++; vp[j-b.c0]++; }
        }
    }
}

static void center_of_mass(const std::vector<std::vector<uint8_t>> &bw, const BBox &b,
                           double &cy, double &cx){
    long long sum=0; long long sy=0, sx=0;
    for (int i=b.r0;i<=b.r1;++i){
        for (int j=b.c0;j<=b.c1;++j){ if (bw[i][j]){ sum++; sy+=i; sx+=j; } }
    }
    if (sum==0){ cy = (b.r0+b.r1)/2.0; cx=(b.c0+b.c1)/2.0; return; }
    cy = (double)sy / (double)sum; cx = (double)sx / (double)sum;
}

static int count_endpoints4(const std::vector<std::vector<uint8_t>> &bw, const BBox &b){
    int ep = 0; int dr[4]={-1,1,0,0}; int dc[4]={0,0,-1,1};
    for (int i=b.r0;i<=b.r1;++i){
        for (int j=b.c0;j<=b.c1;++j){
            if (!bw[i][j]) continue;
            int n=0; for (int k=0;k<4;++k){int r=i+dr[k], c=j+dc[k]; if (r>=b.r0&&r<=b.r1&&c>=b.c0&&c<=b.c1 && bw[r][c]) n++;}
            if (n==1) ep++;
        }
    }
    return ep;
}

} // namespace nr_heur

int judge(IMAGE_T &img){
    using namespace nr_heur;
    if (img.empty() || img[0].empty()) return 0;
    auto bw = binarize(img);
    BBox b = bounding_box(bw);
    if (b.r1 < b.r0) return 0;
    int area = count_area(bw, b);
    int h = b.h(), w = b.w();
    double fill = (h>0 && w>0) ? (double)area / (double)(h*w) : 0.0;
    int holes = count_holes(bw, b);
    std::vector<int> hp, vp; projections(bw, b, hp, vp);
    double cy, cx; center_of_mass(bw, b, cy, cx);
    double top_sum=0, bot_sum=0, left_sum=0, right_sum=0;
    for (int i=0;i<h;++i){ (i < h/2 ? top_sum : bot_sum) += hp[i]; }
    for (int j=0;j<w;++j){ (j < w/2 ? left_sum : right_sum) += vp[j]; }
    double tb = (bot_sum>1e-6) ? top_sum / bot_sum : 1.0;
    double lr = (right_sum>1e-6) ? left_sum / right_sum : 1.0;
    int endpoints = count_endpoints4(bw, b);

    // Rule 1: Distinguish by holes
    if (holes >= 2) return 8;
    if (holes == 1){
        // Distinguish 0 vs 6 vs 9 using vertical mass distribution and endpoints
        if (endpoints <= 2 && fill >= 0.25 && fill <= 0.65) return 0;
        // Top heavy -> 9, bottom heavy -> 6, else fallback to 0
        if (tb > 1.15) return 9;
        if (tb < 0.87) return 6;
        // Right vs left bias: 9 tends right-heavy, 6 left-heavy
        if (lr < 0.9) return 9; // right heavy
        if (lr > 1.1) return 6; // left heavy
        return 0;
    }

    // No holes: classify 1,7,4, others
    double aspect = (h>0)? (double)w / (double)h : 1.0;
    if (aspect < 0.45 && fill < 0.55) return 1;

    // Check strong top horizontal bar and sparse bottom -> 7
    int top_rows = std::max(2, h/4);
    long long top_bar=0; for (int i=0;i<top_rows && i<h; ++i) top_bar += hp[i];
    long long bottom_region=0; for (int i=h*2/3; i<h; ++i) if (i>=0) bottom_region += hp[i];
    if (top_bar > (long long)(0.35*area) && bottom_region < (long long)(0.25*area) && lr < 0.95) return 7;

    // Attempt to detect 4: strong vertical on left half + crossbar around mid
    int midr = h/2; long long mid_sum = (midr>=0&&midr<h)? hp[midr] : 0;
    long long left_vert=0; for (int i=0;i<h;++i){ int col = w/3; if (col>=0&&col<w) left_vert += (vp[col]); break; }
    (void)left_vert; // avoid unused in some compilers
    if (mid_sum > (long long)(0.25*area) && tb > 0.9 && tb < 1.2 && lr > 0.9 && lr < 1.1) {
        // Balanced with strong mid bar often 4/5/2; favor 4
        return 4;
    }

    // Fallback rules for 2,3,5: use center of mass and left/right bias
    if (lr < 0.85) return 3; // right heavy
    if (lr > 1.15 && tb < 1.0) return 5; // left heavy and bottom heavier
    if (tb > 1.1 && lr > 0.9 && lr < 1.1) return 2; // top heavy and centered

    // Final fallback: choose 3 for right bias, 2 otherwise
    return (lr < 1.0 ? 3 : 2);
}
