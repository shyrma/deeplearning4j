/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//

#include <ops/declarable/helpers/histogramFixedWidth.h>
#include <cuda_exception.h>
#include <PointersManager.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename X, typename Z>
__global__ static void histogramFixedWidthCuda( const void* vx, const Nd4jLong* xShapeInfo,
                                                      void* vz, const Nd4jLong* zShapeInfo,
                                                const X leftEdge, const X rightEdge) {

    const auto x  = reinterpret_cast<const X*>(vx);
    auto z = reinterpret_cast<Z*>(vz);

    __shared__ Nd4jLong xLen, zLen, totalThreads, nbins;
    __shared__ X binWidth, secondEdge, lastButOneEdge;

    if (threadIdx.x == 0) {

        xLen  = shape::length(xShapeInfo);
        nbins = shape::length(zShapeInfo);          // nbins = zLen
        totalThreads = gridDim.x * blockDim.x;

        binWidth       = (rightEdge - leftEdge ) / nbins;
        secondEdge     = leftEdge + binWidth;
        lastButOneEdge = rightEdge - binWidth;
    }

    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < xLen; i += totalThreads) {

        const X value = x[shape::getIndexOffset(i, xShapeInfo)];

        Nd4jLong zIndex;

        if(value < secondEdge)
            zIndex = 0;
        else if(value >= lastButOneEdge)
            zIndex = nbins - 1;
        else
            zIndex = static_cast<Nd4jLong>((value - leftEdge) / binWidth);

        nd4j::math::atomics::nd4j_atomicAdd<Z>(&z[shape::getIndexOffset(zIndex, zShapeInfo)], 1);
    }
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Z>
__host__ static void histogramFixedWidthCudaLauncher(const cudaStream_t *stream, const NDArray& input, const NDArray& range, NDArray& output) {

    const X leftEdge  = range.e<X>(0);
    const X rightEdge = range.e<X>(1);

    histogramFixedWidthCuda<X, Z><<<256, 256, 1024, *stream>>>(input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftEdge, rightEdge);
}

////////////////////////////////////////////////////////////////////////
void histogramFixedWidth(nd4j::LaunchContext* context, const NDArray& input, const NDArray& range, NDArray& output) {

    // firstly initialize output with zeros
    output.nullify();

    PointersManager manager(context, "histogramFixedWidth");

    NDArray::prepareSpecialUse({&output}, {&input});
    BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), histogramFixedWidthCudaLauncher, (context->getCudaStream(), input, range, output), LIBND4J_TYPES, INDEXING_TYPES);
    NDArray::registerSpecialUse({&output}, {&input});

    manager.synchronize();
}


//     template <typename T>
//     __global__ static void copyBuffers(Nd4jLong* destination, void const* source, Nd4jLong* sourceShape, Nd4jLong bufferLength) {
//         const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
//         const auto step = gridDim.x * blockDim.x;
//         for (int t = tid; t < bufferLength; t += step) {
//             destination[t] = reinterpret_cast<T const*>(source)[shape::getIndexOffset(t, sourceShape)];
//         }
//     }

//     template <typename T>
//     __global__ static void returnBuffers(void* destination, Nd4jLong const* source, Nd4jLong* destinationShape, Nd4jLong bufferLength) {
//         const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
//         const auto step = gridDim.x * blockDim.x;
//         for (int t = tid; t < bufferLength; t += step) {
//             reinterpret_cast<T*>(destination)[shape::getIndexOffset(t, destinationShape)] = source[t];
//         }
//     }

//     template <typename T>
//     static __global__ void histogramFixedWidthKernel(void* outputBuffer, Nd4jLong outputLength, void const* inputBuffer, Nd4jLong* inputShape, Nd4jLong inputLength, double const leftEdge, double binWidth, double secondEdge, double lastButOneEdge) {

//         __shared__ T const* x;
//         __shared__ Nd4jLong* z; // output buffer

//         if (threadIdx.x == 0) {
//             z = reinterpret_cast<Nd4jLong*>(outputBuffer);
//             x = reinterpret_cast<T const*>(inputBuffer);
//         }
//         __syncthreads();
//         auto tid = blockIdx.x * gridDim.x + threadIdx.x;
//         auto step = blockDim.x * gridDim.x;

//         for(auto i = tid; i < inputLength; i += step) {

//             const T value = x[shape::getIndexOffset(i, inputShape)];
//             Nd4jLong currInd = static_cast<Nd4jLong>((value - leftEdge) / binWidth);

//             if(value < secondEdge)
//                 currInd = 0;
//             else if(value >= lastButOneEdge)
//                 currInd = outputLength - 1;
//             nd4j::math::atomics::nd4j_atomicAdd(&z[currInd], 1LL);
//         }
//     }


//     template <typename T>
//     void histogramFixedWidth_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& range, NDArray& output) {
//         const int nbins = output.lengthOf();
//         auto stream = context->getCudaStream();
//         // firstly initialize output with zeros
//         //if(output.ews() == 1)
//         //    memset(output.buffer(), 0, nbins * output.sizeOfT());
//         //else
//         output.assign(0);
//         if (!input.isActualOnDeviceSide())
//             input.syncToDevice();

//         const double leftEdge  = range.e<double>(0);
//         const double rightEdge = range.e<double>(1);

//         const double binWidth       = (rightEdge - leftEdge ) / nbins;
//         const double secondEdge     = leftEdge + binWidth;
//         double lastButOneEdge = rightEdge - binWidth;
//         Nd4jLong* outputBuffer;
//         cudaError_t err = cudaMalloc(&outputBuffer, output.lengthOf() * sizeof(Nd4jLong));
//         if (err != 0)
//             throw cuda_exception::build("helpers::histogramFixedWidth: Cannot allocate memory for output", err);
//         copyBuffers<Nd4jLong ><<<256, 512, 8192, *stream>>>(outputBuffer, output.getSpecialBuffer(), output.getSpecialShapeInfo(), output.lengthOf());
//         histogramFixedWidthKernel<T><<<256, 512, 8192, *stream>>>(outputBuffer, output.lengthOf(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), input.lengthOf(), leftEdge, binWidth, secondEdge, lastButOneEdge);
//         returnBuffers<Nd4jLong><<<256, 512, 8192, *stream>>>(output.specialBuffer(), outputBuffer, output.specialShapeInfo(), output.lengthOf());
//         //cudaSyncStream(*stream);
//         err = cudaFree(outputBuffer);
//         if (err != 0)
//             throw cuda_exception::build("helpers::histogramFixedWidth: Cannot deallocate memory for output buffer", err);
//         output.tickWriteDevice();
// //#pragma omp parallel for schedule(guided)
// //        for(Nd4jLong i = 0; i < input.lengthOf(); ++i) {
// //
// //            const T value = input.e<T>(i);
// //
// //            if(value < secondEdge)
// //#pragma omp critical
// //                output.p<Nd4jLong>(0, output.e<Nd4jLong>(0) + 1);
// //            else if(value >= lastButOneEdge)
// //#pragma omp critical
// //                output.p<Nd4jLong>(nbins-1, output.e<Nd4jLong>(nbins-1) + 1);
// //            else {
// //                Nd4jLong currInd = static_cast<Nd4jLong>((value - leftEdge) / binWidth);
// //#pragma omp critical
// //                output.p<Nd4jLong>(currInd, output.e<Nd4jLong>(currInd) + 1);
// //            }
// //        }
//     }

//     void histogramFixedWidth(nd4j::LaunchContext * context, const NDArray& input, const NDArray& range, NDArray& output) {
//         BUILD_SINGLE_SELECTOR(input.dataType(), histogramFixedWidth_, (context, input, range, output), LIBND4J_TYPES);
//     }
//     BUILD_SINGLE_TEMPLATE(template void histogramFixedWidth_, (nd4j::LaunchContext * context, const NDArray& input, const NDArray& range, NDArray& output), LIBND4J_TYPES);

}
}
}