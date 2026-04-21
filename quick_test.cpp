#include <iostream>
#include <vector>
#include src.hpp
int main(){
    IMAGE_T img(28, std::vector<double>(28, 0.0));
    for(int i=5;i<23;i++) img[i][14] = 1.0;
    std::cout << judge(img) << n;
    return 0;
}
