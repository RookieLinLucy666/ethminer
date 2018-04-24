#include "ethash_cuda_miner_kernel_globals.h"
#include "ethash_cuda_miner_kernel.h"
#include "cuda_helper.h"
#include <stdio.h>
#include <inttypes.h>

template <uint32_t _PARALLEL_HASH>
__device__ __forceinline__ uint64_t compute_hash(
	uint64_t nonce
	)
{

	// Threads work together in this phase in groups of 8.
	const int thread_id  = threadIdx.x &  (THREADS_PER_HASH - 1);// the last 3 bits
	//const int mix_idx    = thread_id & 3;

	// sha3_512(header .. nonce)
	uint2 state[12];
	state[4] = vectorize(nonce);
	keccak_f1600_init(state);

	// 0 4
	// _PARALLEL_HASH 8 ~
	// THREADS_PER_HASH 8
	for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
	{
		uint4 mix[_PARALLEL_HASH];
		uint32_t offset[_PARALLEL_HASH];
		uint32_t init0[_PARALLEL_HASH];
	
		// share init among threads
		// 1 iter per thread
		for (int p = 0; p < _PARALLEL_HASH; p++)
		{
			uint2 shuffle[8];

			// shuffle := state [0-7] of thread i+p
			for (int j = 0; j < 8; j++) 
			{
				// i+p the p th thread for i/_PARALLEL_HASH th hash
				shuffle[j].x = __shfl_sync(0xFFFFFFFF,state[j].x, i+p, THREADS_PER_HASH);
				shuffle[j].y = __shfl_sync(0xFFFFFFFF,state[j].y, i+p, THREADS_PER_HASH);
			}

			mix[p] = vectorize2(shuffle[p], shuffle[p+1]);
			init0[p] = __shfl_sync(0xFFFFFFFF,shuffle[0].x, 0, THREADS_PER_HASH);
		}

		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = bfe(a, 2u, 3u);



			for (uint32_t b = 0; b < 4; b++)
			{

				uint4 dag_val[_PARALLEL_HASH];

				offset[0] = fnv(init0[0] ^ (a + b), ((uint32_t *)&mix[0])[b]) % d_dag_size;
				offset[0] = __shfl_sync(0xFFFFFFFF,offset[0], t, THREADS_PER_HASH);

				dag_val[0] = d_dag[offset[0]].uint4s[thread_id]);

				offset[1] = fnv(init0[0] ^ (a + b), ((uint32_t *)&mix[0])[b]) % d_dag_size;
				offset[1] = __shfl_sync(0xFFFFFFFF,offset[1], t, THREADS_PER_HASH);

				// _PARALLEL_HASH = 8
				#pragma unroll
				for (int p = 0; p < (_PARALLEL_HASH-2); p+=1)
				{
					mix[p] = fnv4(mix[p], dag_val[p]);

					dag_val[p+1] = d_dag[offset[p+1]].uint4s[thread_id];

					offset[p+2] = fnv(init0[p+2] ^ (a + b), ((uint32_t *)&mix[p+2])[b]) % d_dag_size;
					offset[p+2] = __shfl_sync(0xFFFFFFFF,offset[p+2], t, THREADS_PER_HASH);

				}

				mix[_PARALLEL_HASH-2] = fnv4(mix[_PARALLEL_HASH-2], dag_val[_PARALLEL_HASH-2]);

				dag_val[_PARALLEL_HASH-1] = d_dag[offset[_PARALLEL_HASH-1]].uint4s[thread_id];

				mix[_PARALLEL_HASH-1] = fnv4(mix[_PARALLEL_HASH-1], dag_val[_PARALLEL_HASH-1]);

//				#pragma unroll
//				for (int p = 0; p < _PARALLEL_HASH; p++)
//				{
//					dag_val[p] = __ldg( &(d_dag[offset[p]].uint4s[thread_id]) );
//				}

//				// original ver
//				#pragma unroll
//				for (int p = 0; p < _PARALLEL_HASH; p++)
//				{
//					mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]); // bottleneck
//				}
			}
		}

		for (int p = 0; p < _PARALLEL_HASH; p++)
		{
			// uint2 => unsigned int x, y
			uint2 shuffle[4];
			uint32_t thread_mix = fnv_reduce(mix[p]);

			// update mix accross threads
			for(int k=0;k<4;k++) {
				shuffle[k].x = __shfl_sync(0xFFFFFFFF,thread_mix, 2*k, THREADS_PER_HASH);
				shuffle[k].y = __shfl_sync(0xFFFFFFFF,thread_mix, 2*k+1, THREADS_PER_HASH);
			}
//			shuffle[0].x = __shfl_sync(0xFFFFFFFF,thread_mix, 0, THREADS_PER_HASH);
//			shuffle[0].y = __shfl_sync(0xFFFFFFFF,thread_mix, 1, THREADS_PER_HASH);
//			shuffle[1].x = __shfl_sync(0xFFFFFFFF,thread_mix, 2, THREADS_PER_HASH);
//			shuffle[1].y = __shfl_sync(0xFFFFFFFF,thread_mix, 3, THREADS_PER_HASH);
//			shuffle[2].x = __shfl_sync(0xFFFFFFFF,thread_mix, 4, THREADS_PER_HASH);
//			shuffle[2].y = __shfl_sync(0xFFFFFFFF,thread_mix, 5, THREADS_PER_HASH);
//			shuffle[3].x = __shfl_sync(0xFFFFFFFF,thread_mix, 6, THREADS_PER_HASH);
//			shuffle[3].y = __shfl_sync(0xFFFFFFFF,thread_mix, 7, THREADS_PER_HASH);
			if ((i+p) == thread_id) {
				//move mix into state:
				for(int k=8;k<12;k++) {
					state[k] = shuffle[k-8];
				}
//				state[8] = shuffle[0];
//				state[9] = shuffle[1];
//				state[10] = shuffle[2];
//				state[11] = shuffle[3];
			}
		}
	}


	// keccak_256(keccak_512(header..nonce) .. mix);
	uint64_t final = keccak_f1600_final(state);
	//printf("nonce -> %"PRIu64" final -> %"PRIu64"\n", nonce, final);
	return final;
}
