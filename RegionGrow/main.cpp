
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include "tgt/vector.h"

#pragma comment(lib,"cudart.lib")

#define VOXEL_TYPE unsigned char


extern "C"
void RegionGrowGPU(unsigned char* inVol, unsigned char* outVol, tgt::svec3 volDim, std::vector<tgt::svec3>& seeds);


void preThreshold(VOXEL_TYPE* inVol, tgt::ivec3& volDim, float upThreshold, float dwThreshold, VOXEL_TYPE* outVol)
{
	size_t numVoxels = volDim.x*volDim.y*volDim.z;

	for (size_t i = 0; i < numVoxels; i++)
	{
		if (inVol[i] > VOXEL_TYPE(dwThreshold) && inVol[i] < VOXEL_TYPE(upThreshold))
		{
			outVol[i] = VOXEL_TYPE(255);
		}
		else
		{
			outVol[i] = VOXEL_TYPE(0);
		}
	}
}

int main()
{

	char* fn = "256x256x128.raw";
	VOXEL_TYPE* data = nullptr;
	VOXEL_TYPE* outLabels = nullptr;

	tgt::ivec3 volDim = tgt::svec3(256, 256, 128);

	data = new VOXEL_TYPE[volDim.x*volDim.y*volDim.z];
	outLabels = new VOXEL_TYPE[volDim.x*volDim.y*volDim.z];

	memset(outLabels, 0, volDim.x*volDim.y*volDim.z);

	std::ifstream inFile(fn, std::ios::binary | std::ios::in);

	if (inFile.is_open())
	{
		inFile.read((char*)data, volDim.x*volDim.y*volDim.z*sizeof(char));
		inFile.close();
	}
	else
	{
		inFile.close();
		return 0;
	}

	std::vector<tgt::svec3> seeds;
	seeds.push_back(tgt::svec3(224, 42,64));

	float upThres = 255, dwThres = 54;

	preThreshold(data, volDim, upThres, dwThres, data);

	clock_t s, e;

	s = clock();

	RegionGrowGPU(data, outLabels, volDim, seeds);

	e = clock();

	std::cout << "time " << float(e - s) / 1000.f << std::endl;

	std::ofstream ouFile("res.raw", std::ios::binary | std::ios::out);
	ouFile.write((char*)outLabels, volDim.x*volDim.y*volDim.z*sizeof(char));
	ouFile.close();

	return 0;
}