// Heuristic handwritten digit recognizer for 28x28 grayscale images in [0,1]
// Implements judge(IMAGE_T&) returning 0..9 using simple geometric features.

#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

// Do not redefine IMAGE_T; use explicit type to avoid typedef conflicts with OJ

namespace nr_heur {

struct BBox { int r0, c0, r1, c1; int h() const { return std::max(0, r1 - r0 + 1); } int w() const { return std::max(0, c1 - c0 + 1); } };

static int H = 28, W = 28;

static double clamp01(double x){ return x < 0 ? 0 : (x > 1 ? 1 : x); }

static double otsu_threshold(const std::vector<std::vector<double> > &img){
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

static std::vector<std::vector<unsigned char> > binarize(const std::vector<std::vector<double> > &img){
    H = (int)img.size();
    W = H ? (int)img[0].size() : 0;
    double t = otsu_threshold(img);
    // Slight bias to prefer foreground being bright digits
    double thr = std::max(0.3, std::min(0.7, t * 0.9));
    std::vector<std::vector<unsigned char> > bw(H, std::vector<unsigned char>(W, 0));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            bw[i][j] = (img[i][j] > thr) ? 1 : 0;
        }
    }
    return bw;
}

static BBox bounding_box(const std::vector<std::vector<unsigned char> > &bw){
    int r0 = H, c0 = W, r1 = -1, c1 = -1;
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) if (bw[i][j]) {
        r0 = std::min(r0, i); c0 = std::min(c0, j);
        r1 = std::max(r1, i); c1 = std::max(c1, j);
    }
    if (r1 < r0 || c1 < c0) return {0,0,-1,-1};
    return {r0,c0,r1,c1};
}

static int count_area(const std::vector<std::vector<unsigned char> > &bw, const BBox &b){
    if (b.r1 < b.r0) return 0;
    int a = 0; for (int i=b.r0;i<=b.r1;++i) for (int j=b.c0;j<=b.c1;++j) a+=bw[i][j]; return a;
}

static int count_holes(const std::vector<std::vector<unsigned char> > &bw, const BBox &b){
    if (b.r1 < b.r0) return 0;
    int h = b.h(), w = b.w();
    std::vector<std::vector<unsigned char> > seen(h, std::vector<unsigned char>(w, 0));
    struct Helper { static bool inb(int r,int c,int h,int w){return r>=0&&r<h&&c>=0&&c<w;} };
    int dr[4]={-1,1,0,0}; int dc[4]={0,0,-1,1};
    int components_bg = 0; int touch_border = 0;
    for (int i=0;i<h;++i) for (int j=0;j<w;++j) if (!seen[i][j] && (bw[b.r0+i][b.c0+j]==0)){
        components_bg++;
        bool touches = false;
        std::queue<std::pair<int,int> > q; q.push(std::make_pair(i,j)); seen[i][j]=1;
        while(!q.empty()){
            std::pair<int,int> p = q.front(); q.pop();
            int r = p.first, c = p.second;
            if (r==0||c==0||r==h-1||c==w-1) touches = true;
            for (int k=0;k<4;++k){int nr=r+dr[k], nc=c+dc[k]; if (Helper::inb(nr,nc,h,w) && !seen[nr][nc] && (bw[b.r0+nr][b.c0+nc]==0)){seen[nr][nc]=1; q.push(std::make_pair(nr,nc));}}
        }
        if (touches) touch_border++;
    }
    // background components not touching border are holes
    int holes = components_bg - touch_border;
    if (holes < 0) holes = 0;
    return holes;
}

static bool hole_centroid(const std::vector<std::vector<unsigned char> > &bw, const BBox &b,
                          double &hr, double &hc){
    if (b.r1 < b.r0) return false;
    int h = b.h(), w = b.w();
    std::vector<std::vector<unsigned char> > seen(h, std::vector<unsigned char>(w, 0));
    int dr[4]={-1,1,0,0}; int dc[4]={0,0,-1,1};
    bool found=false; long long best_cnt=0; long long best_sr=0, best_sc=0;
    for (int i=0;i<h;++i){
        for (int j=0;j<w;++j){
            if (seen[i][j]) continue;
            if (bw[b.r0+i][b.c0+j]!=0) { seen[i][j]=1; continue; }
            bool touches=false; long long cnt=0, sr=0, sc=0;
            std::queue<std::pair<int,int> > q; q.push(std::make_pair(i,j)); seen[i][j]=1;
            while(!q.empty()){
                std::pair<int,int> p=q.front(); q.pop();
                int r=p.first, c=p.second; cnt++; sr+=r; sc+=c;
                if (r==0||c==0||r==h-1||c==w-1) touches=true;
                for(int k=0;k<4;++k){int nr=r+dr[k], nc=c+dc[k];
                    if (nr>=0&&nr<h&&nc>=0&&nc<w && !seen[nr][nc] && bw[b.r0+nr][b.c0+nc]==0){ seen[nr][nc]=1; q.push(std::make_pair(nr,nc)); }
                }
            }
            if (!touches && cnt>best_cnt){ best_cnt=cnt; best_sr=sr; best_sc=sc; found=true; }
        }
    }
    if (found && best_cnt>0){ hr = (double)best_sr / (double)best_cnt; hc = (double)best_sc / (double)best_cnt; return true; }
    return false;
}

static void projections(const std::vector<std::vector<unsigned char> > &bw, const BBox &b,
                        std::vector<int> &hp, std::vector<int> &vp){
    hp.assign(b.h(), 0); vp.assign(b.w(), 0);
    for (int i=b.r0;i<=b.r1;++i){
        for (int j=b.c0;j<=b.c1;++j){
            if (bw[i][j]){ hp[i-b.r0]++; vp[j-b.c0]++; }
        }
    }
}

static void center_of_mass(const std::vector<std::vector<unsigned char> > &bw, const BBox &b,
                           double &cy, double &cx){
    long long sum=0; long long sy=0, sx=0;
    for (int i=b.r0;i<=b.r1;++i){
        for (int j=b.c0;j<=b.c1;++j){ if (bw[i][j]){ sum++; sy+=i; sx+=j; } }
    }
    if (sum==0){ cy = (b.r0+b.r1)/2.0; cx=(b.c0+b.c1)/2.0; return; }
    cy = (double)sy / (double)sum; cx = (double)sx / (double)sum;
}

static int count_endpoints4(const std::vector<std::vector<unsigned char> > &bw, const BBox &b){
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

int judge(std::vector<std::vector<double> > &img){
    using namespace nr_heur;
    if (img.empty() || img[0].empty()) return 0;
    std::vector<std::vector<unsigned char> > bw = binarize(img);
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
        // Prefer classifying 0/6/9 by hole position
        double hr_c=0.0, hc_c=0.0;
        if (hole_centroid(bw, b, hr_c, hc_c)){
            double rh = (b.h()>0) ? (hr_c / (double)b.h()) : 0.5;
            double rc = (b.w()>0) ? (hc_c / (double)b.w()) : 0.5;
            double dcent = std::sqrt((rh-0.5)*(rh-0.5) + (rc-0.5)*(rc-0.5));
            if (dcent < 0.20) return 0; // hole near center -> 0
            if (rh < 0.5) return 9;     // hole upper -> 9
            if (rh > 0.5) return 6;     // hole lower -> 6
        }
        // Fallbacks
        if (tb > 1.10) return 9;
        if (tb < 0.90) return 6;
        if (endpoints <= 2 && fill >= 0.22 && fill <= 0.70) return 0;
        return (lr < 1.0 ? 9 : 6);
    }

    // No holes: classify 1,7,4, others
    double aspect = (h>0)? (double)w / (double)h : 1.0;
    int nonempty_cols = 0; for (int j=0;j<w;++j) if (vp[j]>0) nonempty_cols++;
    if (aspect < 0.55 && nonempty_cols <= std::max(2, w/3)) return 1;

    // Check strong top horizontal bar and sparse bottom -> 7
    int top_rows = std::max(2, h/4);
    long long top_bar=0; for (int i=0;i<top_rows && i<h; ++i) top_bar += hp[i];
    long long bottom_region=0; for (int i=h*2/3; i<h; ++i) if (i>=0) bottom_region += hp[i];
    if (top_bar > (long long)(0.30*area) && bottom_region < (long long)(0.28*area) && lr < 0.98) return 7;

    // Attempt to detect 4: strong vertical on left half + crossbar around mid
    int midr = h/2; long long mid_sum = (midr>=0&&midr<h)? hp[midr] : 0;
    long long left_vert=0; for (int i=0;i<h;++i){ int col = w/3; if (col>=0&&col<w) left_vert += (vp[col]); break; }
    (void)left_vert; // avoid unused in some compilers
    if (mid_sum > (long long)(0.22*area) && tb > 0.85 && tb < 1.15) {
        // Balanced with strong mid bar often 4/5/2; favor 4
        return 4;
    }

    // Fallback rules for 2,3,5: use center of mass and left/right bias
    if (lr > 1.05 && tb < 1.02) return 5; // left heavy and slightly bottom-heavy
    if (tb > 1.05 && lr > 0.9 && lr < 1.1) return 2; // top heavy and centered
    if (lr < 0.90) return 3; // right heavy

    // Final fallback: choose 3 for right bias, 2 otherwise
    return (lr < 1.0 ? 3 : 2);
}
