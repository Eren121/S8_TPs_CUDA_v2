nvcc -o a.out farhenheit.cu
nvprof -o profile.timeline ./a.out
nvprof -o profile.metrics --analysis-metrics ./a.out