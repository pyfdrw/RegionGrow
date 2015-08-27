
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"tgt\vector.h"
#include<helper_cuda.h>
#include<vector>

#include <stdio.h>
#include<time.h>


__global__ void kernel(unsigned char* vol, unsigned char* seedsVol, int3 volDim, size_t* numSeeds)
{
	int3 pos;
	pos.x = blockDim.x*blockIdx.x + threadIdx.x;
	pos.y = blockDim.y*blockIdx.y + threadIdx.y;
	pos.z = blockDim.z*blockIdx.z + threadIdx.z;

	if (pos.x > volDim.x /64)
	{
		return;
	}

	if (pos.y > volDim.y - 2 || pos.y <1)
	{
		return;
	}

	if (pos.z > volDim.z - 2 || pos.z <1)
	{
		return;
	}

	//每个线程处理 x 方向上连续的64个像素
	for (size_t i = 0; i < 64; i++)
	{
		size_t index = pos.z*volDim.x*volDim.y + pos.y*volDim.x + pos.x + i;

		if (vol[index] == unsigned char(0) || seedsVol[index] == unsigned char(255))
		{
			//如果某次迭代,所有线程都走这一个路径,说明生长结束

			return;
		}
		else
		{
			//判断种子数据 6 邻域

			size_t x_r = pos.z*volDim.x*volDim.y + pos.y*volDim.x + pos.x + 1 + i;
			size_t x_l = pos.z*volDim.x*volDim.y + pos.y*volDim.x + pos.x - 1 + i;
			size_t y_r = pos.z*volDim.x*volDim.y + (pos.y + 1)*volDim.x + pos.x + i;
			size_t y_l = pos.z*volDim.x*volDim.y + (pos.y - 1)*volDim.x + pos.x + i;
			size_t z_r = (pos.z + 1)*volDim.x*volDim.y + pos.y*volDim.x + pos.x + i;
			size_t z_l = (pos.z - 1)*volDim.x*volDim.y + pos.y*volDim.x + pos.x + i;

			if (seedsVol[x_r] | seedsVol[x_l] | seedsVol[y_r] | seedsVol[y_l] | seedsVol[z_r] | seedsVol[z_l])
			{
				atomicAdd(numSeeds, 1);
				seedsVol[index] = unsigned char(255);
			}
		}
	}

	

}

extern "C"
void RegionGrowGPU(unsigned char* inVol, unsigned char* outVol, tgt::svec3 volDim, std::vector<tgt::svec3>& seeds)
{
	size_t numVoxels = volDim.x*volDim.y*volDim.z;

	unsigned char* d_volume,*d_seedVolume;
	checkCudaErrors(cudaMalloc((void **)&d_volume, numVoxels*sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void **)&d_seedVolume, numVoxels*sizeof(unsigned char)));

	cudaError err;

	err = cudaMemcpy(d_volume, inVol, numVoxels*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	unsigned char* seedVol = new unsigned char[numVoxels];
	memset(seedVol, unsigned char(0), numVoxels*sizeof(unsigned char));

	for (size_t i = 0; i < seeds.size(); i++)
	{
		tgt::svec3 pos = seeds.at(i);
		seedVol[pos.z*volDim.x*volDim.y + pos.y*volDim.x + pos.x] = unsigned char(255);
	}

	err = cudaMemcpy(d_seedVolume, seedVol, numVoxels*sizeof(unsigned char), cudaMemcpyHostToDevice);

	delete[] seedVol;
	seedVol = nullptr;

	int3 vd = make_int3(volDim.x, volDim.y, volDim.z);

	dim3 dimGrid((volDim.x + 7) / 64, (volDim.y + 7) / 8, (volDim.z + 7) / 8);
	dim3 dimBlock(1, 8, 8);

	/*for (size_t i = 0; i < 500; i++)
	{
		kernel << <dimGrid, dimBlock >> >(d_volume, d_seedVolume, vd);
	}*/

	size_t *d_numSeeds;
	size_t numSeeds = 0, lastValue = 0;


	cudaMalloc((void**)&d_numSeeds, sizeof(size_t));

	cudaMemcpy(d_numSeeds, &numSeeds, sizeof(size_t), cudaMemcpyHostToDevice);


	int i = 0;

	while (true)
	{
		i++;

		kernel << <dimGrid, dimBlock >> >(d_volume, d_seedVolume, vd,d_numSeeds);

		//判断种子点是否增加
		cudaMemcpy(&numSeeds, d_numSeeds, sizeof(size_t), cudaMemcpyDeviceToHost);

		if (i>400)
		{
			break;
		}

		//if (numSeeds == lastValue)
		//{
		//	std::cout << "i " << i << std::endl;
		//	break;
		//}
		//else
		//{
		//	lastValue = numSeeds;
		//}
	}

	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	err = cudaMemcpy(outVol, d_seedVolume, numVoxels*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_volume);
	cudaFree(d_seedVolume);

}