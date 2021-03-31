CC=nvcc -g

bin/sum_v3: sum_v3.cu
	$(CC) -o $@ $^

bin/sum_v2: sum_v2.cu
	$(CC) -o $@ $^

sum_v2 sum_v3 hypercube:
%: %.cu
	$(CC) -o $@ $^
