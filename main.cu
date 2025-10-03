#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
__device__ float CalcDistance(float3 a, float3 b) {
    return (a.x - b.x) * (a.x - b.x)   + (a.y - b.y) * (a.y - b.y)  + (a.z  - b.z) * (a.z -b.z);
}
__global__ void FindClosest(float3* point,int* id,int count) {
    if (count <= 1)
        return;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count)
        return;
    float3 thispoint = point[idx];
    float smallestSoFar = 3.4e38;

    for (int i = 0;i < count;i++) {
        if (i == idx) continue;

        auto dis = CalcDistance(thispoint,point[i]);
        if (dis < smallestSoFar) {
            smallestSoFar = dis;
            id[idx] = i;
        }
    }
}
int main()
{
    float3* point;
    int *id;
    int count;


    count = 5;
    cudaMallocManaged(&point,sizeof(float3) * count);
    cudaMallocManaged(&id,sizeof(int) * count);

    for (int i = 0;i < count;i++) {
        point[i].x = rand() % 10;
        point[i].y = rand() % 10;
        point[i].z = rand() % 10;
    }
    FindClosest<<<count / 255 + 1,255>>>(point,id,count);
    cudaDeviceSynchronize();
    for (int i =0 ;i < count;i++)
        cout << "Point: " << point[i].x << ' ' << point[i].y << ' ' << point[i].z << " Neareast Neighbor index: " << id[i] << '\n';
}