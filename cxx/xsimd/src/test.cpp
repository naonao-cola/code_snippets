/**
 * @FilePath     : /test_xsimd/src/test.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-09-20 22:44:56
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-09-21 00:19:11
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#include "test.h"
using namespace std::chrono;

double work( double *a, double *b, size_t n )
{
    using batch_type = xsimd::batch<double, xsimd::default_arch>;
	std::size_t inc  = batch_type::size;
    std::size_t vec_size = n - n % inc;
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (long i=0; i < vec_size; i += inc) {
        batch_type v1 = xsimd::load_unaligned(a + i);
        batch_type v2 = xsimd::load_unaligned(b + i);
        batch_type batch_c = v1 + v2;
        sum += xsimd::reduce_add(batch_c);
    }
#pragma omp parallel for reduction(+:sum)
    for (long i = vec_size; i < n; ++i)
    {
        sum += (a[i] + b[i]);
    }

   return sum;
}

double work_no_omp( double *a, double *b, size_t n )
{
   size_t i;
   double tmp, sum;
   sum = 0.0;
   //#pragma omp simd private(tmp) reduction(+:sum)
   for (i = 0; i < n; i++) {
      tmp = a[i] + b[i];
      sum += tmp;
   }
   return sum;
}

void fill_random_1d_real(size_t n, double *arr)
{

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
    for (size_t i = 0; i < n; ++i)
    {
        arr[i] = distribution(generator);
    }

}

void test_01(){

     printf("Hello, from test_01 !\n");

    //数据量
    size_t n = 429496729;
    double* a = (double*)malloc(n *sizeof(double));
    double* b = (double*)malloc(n *sizeof(double));
    fill_random_1d_real(n, a);
    fill_random_1d_real(n, b);


    auto start = std::chrono::high_resolution_clock::now();
    double sum =0;
    for(int i =0;i<20;i++){
        sum += work_no_omp(a, b, n) /((double) RAND_MAX);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_std = end - start;
    std::cout << "test_01 took me " << sum <<"  "<<  elapsed_std.count() << " seconds." << std::endl;

    sum =0;
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum)
    for(int i =0;i<20;i++){
        sum += work(a, b, n)/((double) RAND_MAX);
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_std = end - start;
    std::cout << "test_02 took me " << sum <<"  "<< elapsed_std.count() << " seconds." << std::endl;

    free(a);
    free(b);
}
