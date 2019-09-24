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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/image_suppression.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// needToSuppressWithThreshold - predicate for suppression
//      boxes - boxes tensor buffer
//      boxesShape boxes tensor shape
//      previousIndex - index for current pos value
//      nextIndex - index for neighbor pos value
//      threshold - threashold value to suppress
//
//      return value: true, if threshold is overcome, false otherwise
//
    template <typename T>
    static __device__ bool needToSuppressWithThreshold(T* boxes, Nd4jLong* boxesShape, int previousIndex, int nextIndex, T threshold) {
        Nd4jLong previous0[] = {previousIndex, 0};
        Nd4jLong previous1[] = {previousIndex, 1};
        Nd4jLong previous2[] = {previousIndex, 2};
        Nd4jLong previous3[] = {previousIndex, 3};
        Nd4jLong next0[] = {nextIndex, 0};
        Nd4jLong next1[] = {nextIndex, 1};
        Nd4jLong next2[] = {nextIndex, 2};
        Nd4jLong next3[] = {nextIndex, 3};

        // we have rectangle with given max values. Compute vexes of rectangle first

        T minYPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(boxesShape, previous0)], boxes[shape::getOffset(boxesShape, previous2)]);
        T minXPrev = nd4j::math::nd4j_min(boxes[shape::getOffset(boxesShape, previous1)], boxes[shape::getOffset(boxesShape, previous3)]);
        T maxYPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(boxesShape, previous0)], boxes[shape::getOffset(boxesShape, previous2)]);
        T maxXPrev = nd4j::math::nd4j_max(boxes[shape::getOffset(boxesShape, previous1)], boxes[shape::getOffset(boxesShape, previous3)]);
        T minYNext = nd4j::math::nd4j_min(boxes[shape::getOffset(boxesShape, next0)],     boxes[shape::getOffset(boxesShape, next2)]);
        T minXNext = nd4j::math::nd4j_min(boxes[shape::getOffset(boxesShape, next1)],     boxes[shape::getOffset(boxesShape, next3)]);
        T maxYNext = nd4j::math::nd4j_max(boxes[shape::getOffset(boxesShape, next0)],     boxes[shape::getOffset(boxesShape, next2)]);
        T maxXNext = nd4j::math::nd4j_max(boxes[shape::getOffset(boxesShape, next1)],     boxes[shape::getOffset(boxesShape, next3)]);

        // compute areas for comparation
        T areaPrev = (maxYPrev - minYPrev) * (maxXPrev - minXPrev);
        T areaNext = (maxYNext - minYNext) * (maxXNext - minXNext);

        // of course, areas should be positive
        if (areaNext <= T(0.f) || areaPrev <= T(0.f)) return false;

        // compute intersection of rectangles
        T minIntersectionY = nd4j::math::nd4j_max(minYPrev, minYNext);
        T minIntersectionX = nd4j::math::nd4j_max(minXPrev, minXNext);
        T maxIntersectionY = nd4j::math::nd4j_min(maxYPrev, maxYNext);
        T maxIntersectionX = nd4j::math::nd4j_min(maxXPrev, maxXNext);
        T intersectionArea =
                nd4j::math::nd4j_max(T(maxIntersectionY - minIntersectionY), T(0.0f)) *
                nd4j::math::nd4j_max(T(maxIntersectionX - minIntersectionX), T(0.0f));
        T intersectionValue = intersectionArea / (areaPrev + areaNext - intersectionArea);
        // final check
        return intersectionValue > threshold;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// shouldSelectKernel - compute status for all selected rectangles (boxes)
//
// we compute boolean flag as shared uint32 and return it on final only for the first thread
//
    template <typename T, typename I>
    static __global__ void shouldSelectKernel(T* boxesBuf, Nd4jLong* boxesShape, I* indexBuf, I* selectedIndicesData, double threshold, int numSelected, int i, bool* shouldSelect) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = gridDim.x * blockDim.x;
        __shared__ unsigned int shouldSelectShared;
        if (threadIdx.x == 0) {
            shouldSelectShared = (unsigned int)shouldSelect[0];
        }
        __syncthreads();
        for (int j = numSelected - 1 - tid; j >= 0; j -= step) {
            if (shouldSelectShared) {
                if (needToSuppressWithThreshold(boxesBuf, boxesShape, indexBuf[i],
                                                                  indexBuf[selectedIndicesData[j]], T(threshold)))
                    atomicCAS(&shouldSelectShared, 1, 0); // exchange only when need to suppress
            }
        }
        __syncthreads();

        // final move: collect result
        if (threadIdx.x == 0) {
            *shouldSelect = shouldSelectShared > 0;
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// indices - type depended, indicesLong - type defined (only 64bit integers)
//
    template <typename I>
    static __global__ void copyIndices(void* indices,  void* indicesLong, Nd4jLong len) {
        I* indexBuf = reinterpret_cast<I*>(indices);
        Nd4jLong* srcBuf = reinterpret_cast<Nd4jLong*>(indicesLong);;

        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for (auto i = tid; i < len; i += step)
            indexBuf[i] = (I)srcBuf[i];
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// nonMaxSuppressionV2 algorithm - given from TF NonMaxSuppressionV2 implementation
//
    template <typename T, typename I>
    static void nonMaxSuppressionV2_(nd4j::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {boxes, scales});
        std::unique_ptr<NDArray> indices(NDArrayFactory::create_<I>('c', {scales->lengthOf()})); // - 1, scales->lengthOf()); //, scales->getContext());
        indices->linspace(0);
        indices->syncToDevice(); // linspace only on CPU, so sync to Device as well

        NDArray scores(*scales);
        Nd4jPointer extras[2] = {nullptr, stream};

        sortByValue(extras, indices->buffer(), indices->shapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(), scores.buffer(), scores.shapeInfo(), scores.specialBuffer(), scores.specialShapeInfo(), true);

        auto indexBuf = reinterpret_cast<I*>(indices->specialBuffer());

        NDArray selectedIndices = NDArrayFactory::create<I>('c', {output->lengthOf()});
        int numSelected = 0;
        int numBoxes = boxes->sizeAt(0);
        auto boxesBuf = reinterpret_cast<T*>(boxes->specialBuffer());

        auto selectedIndicesData = reinterpret_cast<I*>(selectedIndices.specialBuffer());
        auto outputBuf = reinterpret_cast<I*>(output->specialBuffer());

        bool* shouldSelectD;
        auto err = cudaMalloc(&shouldSelectD, sizeof(bool));
        if (err) {
            throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot allocate memory for bool flag", err);
        }
        for (I i = 0; i < boxes->sizeAt(0); ++i) {
            bool shouldSelect = numSelected < output->lengthOf();
            if (shouldSelect) {
                err = cudaMemcpy(shouldSelectD, &shouldSelect, sizeof(bool), cudaMemcpyHostToDevice);
                if (err) {
                    throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot set up bool flag to device", err);
                }

                shouldSelectKernel<T,I><<<128, 256, 1024, *stream>>>(boxesBuf, boxes->specialShapeInfo(), indexBuf, selectedIndicesData, threshold, numSelected, i, shouldSelectD);
                err = cudaMemcpy(&shouldSelect, shouldSelectD, sizeof(bool), cudaMemcpyDeviceToHost);
                if (err) {
                    throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot set up bool flag to host", err);
                }
            }

            if (shouldSelect) {
                cudaMemcpy(reinterpret_cast<I*>(output->specialBuffer()) + numSelected, indexBuf + i, sizeof(I), cudaMemcpyDeviceToDevice);
                cudaMemcpy(selectedIndicesData + numSelected, &i, sizeof(I), cudaMemcpyHostToDevice);
                numSelected++;
            }
        }

        err = cudaFree(shouldSelectD);
        if (err) {
            throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot deallocate memory for bool flag", err);
        }

    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void nonMaxSuppressionV2(nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        BUILD_DOUBLE_SELECTOR(boxes->dataType(), output->dataType(), nonMaxSuppressionV2_, (context, boxes, scales, maxSize, threshold, output), FLOAT_TYPES, INDEXING_TYPES);
    }

}
}
}
