#include <iostream>

int main(int argc, char** argv) {
    char temp[5];
    temp[5] = 'a';  // Out-of-bounds write to trigger Address
    std::cout << "hello world!" << std::endl;
    return 0;
}
