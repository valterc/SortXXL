#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include "HandleError.h"

#include "cudaEngine.h"
#include "demo.h"

__device__ void bubbleSort(int *values, int count);
__device__ void localSort(int *values, int *aux_bucket, int count, int digitCount);
__device__ int intNDigit(int number, int n);

__global__ void sort(int *A, int loop, int N)
{
	
	int i;
	int j;
	
	/* Stride is equal to the total number of threads per grid */
	//int stride = (blockDim.x * blockDim.y) * (gridDim.x * gridDim.y);

	/* tid is the thread working index */
	/*
	int tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y 
			+ blockIdx.x * blockDim.x  * blockDim.y 
			+ threadIdx.y * blockDim.x + threadIdx.x;
		*/	
			
	int tid = blockIdx.x * 1024 + threadIdx.x;
			
			
	int tid_origin = tid;
	
	// 256 - 4 values per thread
	__shared__ int sm_scat[1024]; 
	__shared__ int sm_pos[19]; 
	if (threadIdx.x < 19)
		sm_pos[threadIdx.x] = 0;
	
	__shared__ int sm_posd[19]; 
	if (threadIdx.x < 19)
		sm_posd[threadIdx.x] = 0;
	
	__shared__ int sm_post[19]; 
	if (threadIdx.x < 19)
		sm_post[threadIdx.x] = 0;
	
	
	// 256 threads - 4 values + 4 indexes
	__shared__ int sm_cache[256 * 8];

	int t_values[4];
	int t_pos[19];

	int count = 0;
	while(tid < N && count < 4)
	{
		t_values[count] = A[tid];
		sm_cache[threadIdx.x * 8 + count] = A[tid];
		
		tid += 256;
		count++;
	}
	tid = tid_origin;
	
	__syncthreads();
	
	
	//generate local histogram
	if (tid < N){
		for (i = 0; i < 4; i++)
		{
			int p = intNDigit(t_values[i], loop);
			p += 9; //negative numbers
			
			t_pos[p]++;
			sm_cache[threadIdx.x * 8 + 4 + i] = p;
		}
	}
	
	__syncthreads();	
	
	
	//generate start positions
	if (threadIdx.x == 0){
		
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 256; j++)
			{
				int pos = sm_cache[j * 8 + 4 + i];
				sm_pos[pos]++;
				sm_posd[pos] = 0;
			}
		}
		
		sm_post[0] = 0;
		for (i = 1; i < 19; i++)
		{
			sm_post[i] = sm_post[i-1] + sm_pos[i-1];
		}
		

		//place the values on the respected slots
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 256; j++)
			{
				
				int pos = sm_cache[j * 8 + 4 + i];
				
				int value = sm_cache[j * 8 + i];
				sm_scat[sm_post[pos] + sm_posd[pos]] = value;
				sm_posd[pos]++;
			}
		}

	}
	__syncthreads();
	
	//Put the scattered values in to gpu global memory
	if (tid < N) {
		for (i = 0; i < 4; i++)
		{
			//if (blockIdx.x == 1)
			//	cuPrintf("%d\n", sm_scat[threadIdx.x + 256 * i]);
				
			A[tid + 256 * i] = sm_scat[threadIdx.x + 256 * i];
		}
	}
	
	__syncthreads();
	
}


__device__ void bubbleSort(int *values, int count){
	int i;
	
	int dontExit = 1;
	while (dontExit)
	{
		dontExit = 0;
		for (i = 0; i < count - 1; i++)
		{
			if (values[i] > values[i + 1]){
				int aux = values[i + 1];
				values[i + 1] = values[i];
				values[i] = aux;
				dontExit = 1;
			}
		}
	}
}


__device__ void localSort(int *values, int *aux_bucket, int count, int digitCount){
	
	int i;
	int j;
	int k;
	
	//guarda o deslocamento de cada POS
	int s_values_aux[19]; 
		for (i = 0; i < 19; i++) s_values_aux[i] = 0;
	
	//0 = -9 
	//1 = -8
	//2 = -7
	//3 = -6
	//4 = -5
	//5 = -4
	//6 = -3
	//7 = -2
	//8 = -1
	//9 = 0
	//10 = 1
	//19 = 9
	
	for (i = 0; i < digitCount; i++)
	{
		for (j = 0; j < count; j++)
		{
			int value = values[j];
			int pos = intNDigit(value, i);
			
			//account for negative numbers
			int s_pos = pos + 9;
			
			//incrementar o deslocamento da posição
			s_values_aux[s_pos]++;
		}	
		
		int pd[19];
		for (k = 0; k < 19; k++) pd[k] = 0;
		
		//calculate global offset for each bucket, in the array
		int l;
		int ctx = 0;
		for (l = 0; l < 19; l++)
		{
			pd[l] = ctx;
			ctx += s_values_aux[l];
		}
		
		int sd[19];
		for (k = 0; k < 19; k++) sd[k] = 0;
		
		//Copy the random placed values to the correct sorted position
		for (l = 0; l < count; l++)
		{
			int value = values[l];
			int pos = intNDigit(value, i);
			
			//account for negative numbers
			int s_pos = pos + 9;

			aux_bucket[pd[s_pos] + sd[s_pos]] = value;
			sd[s_pos]++; //calculate bucket inner offset
		}

		//copy the sorted numbers back to the main array
		for (l = 0; l < count; l++)
		{
			values[l] = aux_bucket[l];
		}
		
		//reset values
		for (k = 0; k < 19; k++) s_values_aux[k] = 0;
	}
}


__device__ int intNDigit(int number, int n){
    
    while (n--) {
        number /= 10;
    }
    
    return (number % 10);
}


__global__ void scat_values(int *values, int *positions, int digitPos, int N)
{
	/* Stride is equal to the total number of threads per grid */
	//int stride = (blockDim.x * blockDim.y) * (gridDim.x * gridDim.y);
	
	/* tid is the thread working index */
	int tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y 
			+ blockIdx.x * blockDim.x * blockDim.y 
			+ threadIdx.y * blockDim.x + threadIdx.x;
	
	int i;
	
	//Calculates the global size of each buckets per thread
	for (i = 0; i < N; i++)
	{
		if ((intNDigit(values[i], digitPos) + 9) == tid){
			positions[tid]++;
		}
	}
	
	__syncthreads();
	
}


__global__ void sortByBucket(int *originalValues, int *blockSorted, int *positions, int digitPos, int N)
{
	/* Stride is equal to the total number of threads per grid */
	//int stride = (blockDim.x * blockDim.y) * (gridDim.x * gridDim.y);
	
	/* tid is the thread working index */
	int tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y 
			+ blockIdx.x * blockDim.x * blockDim.y 
			+ threadIdx.y * blockDim.x + threadIdx.x;

	int i;
	int j;
	
	int startPos = 0;
	int count = positions[tid];
	
	for (i = 0; i < tid; i++)
	{
		startPos += positions[i];
	}

	//This kernel places the buckets in order. 
	//Instead of having A1, A2, A3 | B1, B2, B3 | ...
	//We will have A1, B1, ... | B1, B2, ...

	int offset = 0;
	for (j = 0; j < N && offset < count; j++)
	{
		int bucket = intNDigit(originalValues[j], digitPos);
		bucket += 9;
		
		if (bucket == tid){
			blockSorted[startPos + offset] = originalValues[j];
			offset++;
		}
	}

	__syncthreads();
	
}


__global__ void sortEachBucket(int *blockSorted, int *aux, int *positions, int digitPos, int N)
{
	/* Stride is equal to the total number of threads per grid */
	//int stride = (blockDim.x * blockDim.y) * (gridDim.x * gridDim.y);
	
	/* tid is the thread working index */
	int tid = blockIdx.y * gridDim.x * blockDim.x * blockDim.y 
			+ blockIdx.x * blockDim.x * blockDim.y 
			+ threadIdx.y * blockDim.x + threadIdx.x;
	
	int i;

	if (positions[tid] > 0){
		int startPos = 0;
		
		for (i = 0; i < tid; i++)
		{
			startPos += positions[i];
		}
		
		//Sorts each bucket
		localSort((blockSorted + startPos), (aux + startPos), positions[tid], digitPos + 1);
		//bubbleSort((blockSorted + startPos), positions[tid]);
	}
	__syncthreads();
	
}



SortRunResult *radix_sort(int *numbers, int numbersCount, int digitCount){
	
	int n = numbersCount;
	
	int *resultNumbers = (int *)malloc(sizeof(int) * n);
	
	cudaEvent_t start, stop;
	float elapsedTime;
	
	dim3 Blocks(n/1024);
	dim3 Threads(256);
	
	int *dev_values, *dev_values_aux, *dev_sort_pos;
	
	/* allocate the memory on the GPU */
	cudaMalloc((void**)&dev_values, n * sizeof(int));
	cudaMalloc((void**)&dev_values_aux, n * sizeof(int));
	cudaMalloc((void**)&dev_sort_pos, 19 * sizeof(int));
	
		
	/* Copy numbers to GPU */
	cudaMemcpy(dev_values, numbers, n * sizeof(int),cudaMemcpyHostToDevice);
	
	//Init memory
	cudaMemset(dev_sort_pos, 0, 19 * sizeof(int));
	
	/* Create the timers */
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	/* Start the timer */
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	DEMO_startRun();
	
	int digit = digitCount;
	int i;
	for (i = 0; i < digit; i++)
	{
		sort<<<Blocks, Threads>>>(dev_values, i, n);
		CHECK_KERNEL_EXEC();
	}
	DEMO_alertNewStage(1);
	
	//No we have this data:
	//0 .. 10 | -1 .. 10 | 5 .. 9 | etc.
	
	//count the values for the buckets
	scat_values<<<19,1>>>(dev_values, dev_sort_pos, digit - 1, n);
	CHECK_KERNEL_EXEC();
	
	DEMO_alertNewStage(2);
	
	//place each number on the right bucket
	sortByBucket<<<19,1>>>(dev_values, dev_values_aux, dev_sort_pos, digit - 1, n);
	CHECK_KERNEL_EXEC();
	
	DEMO_alertNewStage(3);
	
	//Now we have this data sequence:
	//01 05 01 07 .. 09 | 18 16 .. 19 | etc.
	
	//lets sort each bucket independently
	sortEachBucket<<<19,1>>>(dev_values_aux, dev_values, dev_sort_pos, digit - 1, n);
	CHECK_KERNEL_EXEC();
	
	//The array should be sorted
	
	DEMO_alertNewStage(4);

	/* Terminate the timer */
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	
	HANDLE_ERROR( cudaThreadSynchronize() );
	HANDLE_ERROR( cudaMemcpy(resultNumbers, dev_values, n * sizeof(int), cudaMemcpyDeviceToHost) );


	HANDLE_ERROR( cudaFree(dev_values) );
	HANDLE_ERROR( cudaFree(dev_values_aux) );
	HANDLE_ERROR( cudaFree(dev_sort_pos) );
	
	SortRunResult *result = (SortRunResult *)malloc(sizeof(SortRunResult));
	result->elapsedTime = elapsedTime;
	result->sortedNumbers = resultNumbers;

	
	return result;
}

void printGPUInfo(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	 /*Based on code by nVidia on Cuda Toolkit documentation, docs.nvidia.com/cuda/index.html */
	int major = prop.major;
	int minor = prop.minor;
	
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};

	int index = 0;
	int coreCount = nGpuArchCoresPerSM[7].Cores; //If newer gpu than use the lastest gpu core count

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			coreCount = nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	printf( "Device ------------\n");
	printf( "Name: \t\t\t%s\n", prop.name );
	printf( "Compute capability: \t%d.%d\n", prop.major, prop.minor );
	printf( "Clock rate: \t\t%d\n", prop.clockRate );
	printf( "Total global mem: \t%zu\n", prop.totalGlobalMem );
	printf( "Total constant Mem: \t%zu\n", prop.totalConstMem );
	printf( "Multiprocessor count: \t%d\n", prop.multiProcessorCount );
	printf( "Cuda cores: \t\t%d\n", coreCount * prop.multiProcessorCount);
	printf( "-------------------\n");

}


void printGPUInfoToFile(FILE *f){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	 /*Based on code by nVidia on Cuda Toolkit documentation, docs.nvidia.com/cuda/index.html */
	int major = prop.major;
	int minor = prop.minor;
	
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};

	int index = 0;
	int coreCount = nGpuArchCoresPerSM[7].Cores; //If newer gpu than use the lastest gpu core count

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			coreCount = nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	fprintf(f, "#Device used--------\n");
	fprintf(f, "#Name: \t\t\t\t\t%s\n", prop.name );
	fprintf(f, "#Compute capability: \t%d.%d\n", prop.major, prop.minor );
	fprintf(f, "#Clock rate: \t\t\t%d\n", prop.clockRate );
	fprintf(f, "#Total global mem: \t\t%zu\n", prop.totalGlobalMem );
	fprintf(f, "#Total constant Mem: \t%zu\n", prop.totalConstMem );
	fprintf(f, "#Multiprocessor count: \t%d\n", prop.multiProcessorCount );
	fprintf(f, "#Cuda cores: \t\t\t%d\n", coreCount * prop.multiProcessorCount);
	fprintf(f, "#-------------------\n\n");

}


GPUInfo* queryGPUInfo(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	 /*Based on code by nVidia on Cuda Toolkit documentation, docs.nvidia.com/cuda/index.html */
	int major = prop.major;
	int minor = prop.minor;
	
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};

	int index = 0;
	int coreCount = nGpuArchCoresPerSM[7].Cores; //If newer gpu than use the lastest gpu core count

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			coreCount = nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	GPUInfo *info = (GPUInfo *)malloc(sizeof(GPUInfo));
	info->Name = strdup(prop.name);
	info->versionMajor = prop.major;
	info->versionMinor = prop.minor;
	info->clockRate = prop.clockRate;
	info->globalMemory = prop.totalGlobalMem;
	info->constantMemory = prop.totalConstMem;
	info->multiprocessorCount = prop.multiProcessorCount;
	info->cudaCores = coreCount * prop.multiProcessorCount;
	
	return info;
}
