//#include <thread>
//#include <fstream>
//#include <iostream>
//#include "MinerAux.h"
//#include "BuildInfo.h"
//#include <cuda_runtime.h>
//#include "libethash-cuda/ethash_cuda_miner_kernel.h"
//#include "libethash-cuda/ethash_cuda_miner_kernel_globals.h"
//#include "libethash-cuda/cuda_helper.h"
//
//using namespace std;
//using namespace dev;
//using namespace dev::eth;
//using namespace boost::algorithm;
//
//
//int main(int argc, char** argv)
//{
//	// Set env vars controlling GPU driver behavior.
//	setenv("GPU_MAX_HEAP_SIZE", "100");
//	setenv("GPU_MAX_ALLOC_PERCENT", "100");
//	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");
//
//	printf("fuck you new main.cpp\n");
//
//	cudaStream_t s;
//	//cudaStreamCreate(&s);
//	//ethash_generate_dag(1073739904, 8192, 128, s, 0);
//
//	printf("and fuck you again\n");
//	return 0;
//}

/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */

#include <thread>
#include <fstream>
#include <iostream>
#include "MinerAux.h"
#include "BuildInfo.h"
#include "cuda_profiler_api.h"
#include <numeric>

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace boost::algorithm;


void help()
{
	cout
		<< "Usage ethminer [OPTIONS]" << endl
		<< "Options:" << endl << endl;
	MinerCLI::streamHelp(cout);
	cout
		<< "General Options:" << endl
		<< "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (default: 8)." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl
		;
	exit(0);
}

void version()
{
	cout << "ethminer version " << ETH_PROJECT_VERSION << endl;
	cout << "Build: " << ETH_BUILD_PLATFORM << "/" << ETH_BUILD_TYPE << endl;
	exit(0);
}

int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

	//cudaProfilerStart();
	MinerCLI m(MinerCLI::OperationMode::Farm);

	for (int i = 1; i < argc; ++i)
	{
		// Mining options:
		if (m.interpretOption(i, argc, argv))
			continue;

		// Standard options:
		string arg = argv[i];
		if ((arg == "-v" || arg == "--verbosity") && i + 1 < argc)
			g_logVerbosity = atoi(argv[++i]);
		else if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "-V" || arg == "--version")
			version();
		else
		{
			cerr << "Invalid argument: " << arg << endl;
			exit(-1);
		}
	}

	m.execute();
	cudaDeviceReset();
	return 0;
}

//int main(int argc, char** argv) {
//
//	MinerType _m;
//	unsigned _warmupDuration = 15;
//	unsigned _trialDuration = 3;
//	unsigned _trials = 30;
//
//	BlockHeader genesis;
//	genesis.setNumber(0); // 0
//	genesis.setDifficulty(1 << 18);
//	cdebug << genesis.boundary();
//
//	Farm f; // Farm runs on another thread
//	map<string, Farm::SealerDescriptor> sealers;
//	sealers["cuda"] = Farm::SealerDescriptor{
//		&CUDAMiner::instances
//		, [](FarmFace& _farm, unsigned _index){ return new CUDAMiner(_farm, _index); }
//	};
//	f.setSealers(sealers);
//	f.onSolutionFound([&](Solution) { return false; });
//
//	string platformInfo = "CUDA";
//	cout << "Benchmarking on platform: " << platformInfo << endl;
//
//	cout << "Preparing DAG for block #" << 0 << endl;
//	//genesis.prep();
//
//	genesis.setDifficulty(u256(1) << 63);
//	f.start("cuda", false);
//	f.setWork(WorkPackage(genesis));
//
//	//this_thread::sleep_for(chrono::seconds(30));
//
//	list<uint64_t> results;
//	//uint64_t mean = 0;
//	//uint64_t innerMean = 0;
//	for (unsigned i = 0; i <= _trials; ++i) // _trials = 5
//	{
//		if (!i)
//			cout << "Warming up..." << endl;
//		else
//			cout << "Trial " << i << "... " << flush;
//		this_thread::sleep_for(chrono::seconds(i ? _trialDuration : _warmupDuration));
//
//		auto mp = f.miningProgress();
//		if (!i)
//			continue;
//		uint64_t rate = mp.rate();
//
//		cout << rate << endl;
//		results.push_back(rate);
//		//mean += rate;
//	}
//	f.stop();
////	int j = -1;
////	for (auto const& r: results)
////		if (++j > 0 && j < (int)_trials - 1) innerMean += r.second.rate();
////	innerMean /= (_trials - 2);
//
//	uint64_t avg = 0, max = 0, min = 9999999999;
//	double std;
//
//	for(list<uint64_t>::iterator it=results.begin();it!=results.end();++it) {
//		avg += *it;
//		if(*it < min) min = *it;
//		if(*it > max) max = *it;
//	}
//	avg /= results.size();
//	for(list<uint64_t>::iterator it=results.begin();it!=results.end();++it) {
//		std += (*it-avg) * (*it - avg);
//	}
//	std = sqrt(std/results.size());
//
//	//cout << "min/mean/max: " << results.begin() << "/" << mean << "/" << results.rbegin() << " H/s" << endl;
//	cout << "max: " << max << endl;
//	cout << "min: " << min << endl;
//	cout << "avg: " << avg << endl;
//	cout << "std: " << std << endl;
//	//cout << "std: " << innerMean <<  << endl;
//
//	exit(0);
//}
