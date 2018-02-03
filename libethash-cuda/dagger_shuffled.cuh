#include "ethash_cuda_miner_kernel_globals.h"
#include "ethash_cuda_miner_kernel.h"
#include "cuda_helper.h"
#include <stdio.h>

template <uint32_t _PARALLEL_HASH>
__device__ __forceinline__ uint64_t compute_hash(
	uint64_t nonce
	)
{

	// Threads work together in this phase in groups of 8.
	const int thread_id  = threadIdx.x &  (THREADS_PER_HASH - 1);
	int tid = (threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)?0:1;
	const int mix_idx    = thread_id & 3;

	double ti1=0, ti2=0, ti3=0;
	double t1=0;
	clock_t s;
	if(tid == 0){
		s = clock();
	}


	// sha3_512(header .. nonce)
	uint2 state[12];
	state[4] = vectorize(nonce);
	keccak_f1600_init(state);

	if(tid == 0){
		t1 = (clock()-s);
	}

	// 0 4
	// _PARALLEL_HASH 4
	// THREADS_PER_HASH 8
	for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
	{
		clock_t s1, s2, s3;
		if(tid == 0){
			s1 = clock();
		}


		uint4 mix[_PARALLEL_HASH];
		uint32_t offset[_PARALLEL_HASH];
		uint32_t init0[_PARALLEL_HASH];
	
		// share init among threads
		for (int p = 0; p < _PARALLEL_HASH; p++)
		{
			uint2 shuffle[8];
			for (int j = 0; j < 8; j++) 
			{
#if CUDA_VERSION < SHUFFLE_DEPRECATED
				// int __shfl(int var, int srcLane, int width=warpSize);
				// A lane is a thread in a warp
				// __shfl broadcast var in thread srcLane to thread0 - threadWidth-1
				// THREADS_PER_HASH = 8 // _PARALLEL_HASH = 8
				// i+=8 per iter
				shuffle[j].x = __shfl(state[j].x, i+p, THREADS_PER_HASH);
				shuffle[j].y = __shfl(state[j].y, i+p, THREADS_PER_HASH);
#else
				// __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
				shuffle[j].x = __shfl_sync(0xFFFFFFFF,state[j].x, i+p, THREADS_PER_HASH);
				shuffle[j].y = __shfl_sync(0xFFFFFFFF,state[j].y, i+p, THREADS_PER_HASH);
#endif
			}
			switch (mix_idx)
			{
				// vectorize2 (x, y) => return {x: x.x, y: x.y, z: y.x, w: y.y}
				case 0: mix[p] = vectorize2(shuffle[0], shuffle[1]); break;
				case 1: mix[p] = vectorize2(shuffle[2], shuffle[3]); break;
				case 2: mix[p] = vectorize2(shuffle[4], shuffle[5]); break;
				case 3: mix[p] = vectorize2(shuffle[6], shuffle[7]); break;
			}
#if CUDA_VERSION < SHUFFLE_DEPRECATED
			init0[p] = __shfl(shuffle[0].x, 0, THREADS_PER_HASH);
#else
			init0[p] = __shfl_sync(0xFFFFFFFF,shuffle[0].x, 0, THREADS_PER_HASH);
#endif
		}

		if(tid == 0){
			ti1 += (clock()-s1);
			s2 = clock();
		}


		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = bfe(a, 2u, 3u);

			for (uint32_t b = 0; b < 4; b++)
			{
				for (int p = 0; p < _PARALLEL_HASH; p++)
				{
					offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t *)&mix[p])[b]) % d_dag_size;
#if CUDA_VERSION < SHUFFLE_DEPRECATED
					offset[p] = __shfl(offset[p], t, THREADS_PER_HASH);
#else
					offset[p] = __shfl_sync(0xFFFFFFFF,offset[p], t, THREADS_PER_HASH);
#endif
				}
				// Controls loop unrolling, for improved performance.
				// Just an optimization. No logic changes
				#pragma unroll
				for (int p = 0; p < _PARALLEL_HASH; p++)
				{
					//if(blockIdx.x == 0 && threadIdx.x==0 && offset[p] > (d_dag_size>>1)) //larger than half
					//    printf("d_dag_size = %d offset[p] = %d\n", d_dag_size, offset[p]);
					mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
				}

			}
		}

		if(tid == 0){
			ti2 += (clock()-s2);
			s3 = clock();
		}


		for (int p = 0; p < _PARALLEL_HASH; p++)
		{
			uint2 shuffle[4];
			uint32_t thread_mix = fnv_reduce(mix[p]);

			// update mix accross threads
#if CUDA_VERSION < SHUFFLE_DEPRECATED
			shuffle[0].x = __shfl(thread_mix, 0, THREADS_PER_HASH);
			shuffle[0].y = __shfl(thread_mix, 1, THREADS_PER_HASH);
			shuffle[1].x = __shfl(thread_mix, 2, THREADS_PER_HASH);
			shuffle[1].y = __shfl(thread_mix, 3, THREADS_PER_HASH);
			shuffle[2].x = __shfl(thread_mix, 4, THREADS_PER_HASH);
			shuffle[2].y = __shfl(thread_mix, 5, THREADS_PER_HASH);
			shuffle[3].x = __shfl(thread_mix, 6, THREADS_PER_HASH);
			shuffle[3].y = __shfl(thread_mix, 7, THREADS_PER_HASH);
#else
			shuffle[0].x = __shfl_sync(0xFFFFFFFF,thread_mix, 0, THREADS_PER_HASH);
			shuffle[0].y = __shfl_sync(0xFFFFFFFF,thread_mix, 1, THREADS_PER_HASH);
			shuffle[1].x = __shfl_sync(0xFFFFFFFF,thread_mix, 2, THREADS_PER_HASH);
			shuffle[1].y = __shfl_sync(0xFFFFFFFF,thread_mix, 3, THREADS_PER_HASH);
			shuffle[2].x = __shfl_sync(0xFFFFFFFF,thread_mix, 4, THREADS_PER_HASH);
			shuffle[2].y = __shfl_sync(0xFFFFFFFF,thread_mix, 5, THREADS_PER_HASH);
			shuffle[3].x = __shfl_sync(0xFFFFFFFF,thread_mix, 6, THREADS_PER_HASH);
			shuffle[3].y = __shfl_sync(0xFFFFFFFF,thread_mix, 7, THREADS_PER_HASH);
#endif
			if ((i+p) == thread_id) {
				//move mix into state:
				state[8] = shuffle[0];
				state[9] = shuffle[1];
				state[10] = shuffle[2];
				state[11] = shuffle[3];
			}
		}

		if(tid == 0){
			ti3 += (clock()-s3);
			//printf("_PARALLEL_HASH -> %d\n", _PARALLEL_HASH); //4
			printf("i = %d, nonce-> %llu;t1 -> %lf\n", i, nonce, t1);
			printf("i = %d, nonce-> %llu;ti1 -> %lf\n", i, nonce, ti1);
			printf("i = %d, nonce-> %llu;ti2 -> %lf\n", i, nonce, ti2);
			printf("i = %d, nonce-> %llu;ti3 -> %lf\n", i, nonce, ti3);
		}

	}


	// keccak_256(keccak_512(header..nonce) .. mix);
	return keccak_f1600_final(state);
}
