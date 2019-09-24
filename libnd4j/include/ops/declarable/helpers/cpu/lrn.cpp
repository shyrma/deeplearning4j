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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/lrn.h>
#include <Status.h>
#include <ConstantTadHelper.h>

namespace nd4j {
namespace ops {
namespace helpers {

template <typename T>
static int lrnFunctor_(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, float bias, float alpha, float beta) {

    nd4j_debug("MKL-DNN is not used for lrn!\n", 0);

    const int rank = input->rankOf();

    TadPack inTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {rank - 1});
    TadPack outTadPack;

    if(shape::haveSameShapeAndStrides(input->getShapeInfo(), output->getShapeInfo()))
        outTadPack = inTadPack;
    else
        outTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {rank - 1});

    const Nd4jLong numOfTads = inTadPack.numberOfTads();
    const Nd4jLong tadLen    = input->sizeAt(-1); 
    
    const Nd4jLong* inTadOffsets    = inTadPack.primaryOffsets();        
    const Nd4jLong* outTadOffsets = outTadPack.primaryOffsets();

    const Nd4jLong inTadEws    = shape::elementWiseStride(inTadPack.primaryShapeInfo());
    const Nd4jLong outTadEws = shape::elementWiseStride(outTadPack.primaryShapeInfo());
    
    const T* inBuff  = reinterpret_cast<T*>(input->getBuffer());
          T* outBuff = reinterpret_cast<T*>(output->getBuffer());

    const T tbias  = static_cast<T>(bias);
    const T tbeta  = static_cast<T>(beta);
    const T talpha = static_cast<T>(alpha);    

    if(inTadEws == 1 && outTadEws == 1) {
        
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const T* x = inBuff    + inTadOffsets[i];
                  T* y = outBuff + outTadOffsets[i];

            T prev = 0;

            // calculate squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s)
                        prev = prev + x[s] * x[s];
                    y[j] = prev;
                }
                else if (begin == 0 && last <= tadLen)
                    y[j] = prev + x[end - 1] * x[end - 1];
                else if (begin > 0 && last <= tadLen)
                    y[j] = prev + x[end - 1] * x[end - 1] - x[begin - 1] * x[begin - 1];
                else if (begin > 0 && last > tadLen)
                    y[j] = prev - x[begin - 1] * x[begin - 1];
                else
                    y[j] = prev;

                  if(j != 0)
                    prev = y[j];
                
                y[j] = x[j] / nd4j::math::nd4j_pow<T, T, T>(tbias + alpha * prev, tbeta); 
            }          
        }
    }
    else {
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const T* x = inBuff    + inTadOffsets[i];
                  T* y = outBuff + outTadOffsets[i];

            T prev = 0;

            // calculate squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s)
                        prev = prev + x[s*inTadEws] * x[s*inTadEws];
                    y[j*outTadEws] = prev;
                }
                else if (begin == 0 && last <= tadLen)
                    y[j*outTadEws] = prev + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws];
                else if (begin > 0 && last <= tadLen)
                    y[j*outTadEws] = prev + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws] - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else if (begin > 0 && last > tadLen)
                    y[j*outTadEws] = prev - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else
                    y[j*outTadEws] = prev;

                  if(j != 0)
                    prev = y[j*outTadEws];
                
                y[j*outTadEws] = x[j*inTadEws] / nd4j::math::nd4j_pow<T, T, T>(tbias + alpha * prev, tbeta); 
            }          
        }
    }    
    return Status::OK();
}
    
BUILD_SINGLE_TEMPLATE(template int lrnFunctor_, (nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, float bias, float alpha, float beta), FLOAT_TYPES);

int lrnFunctor(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {
    BUILD_SINGLE_SELECTOR(input->dataType(), return lrnFunctor_, (block, input, output, depth, bias, alpha, beta), FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void lrnBP_(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
    
    const int rank = input.rankOf();

    TadPack inTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), {rank - 1});
    TadPack gradITadPack;

    if(shape::haveSameShapeAndStrides(input.getShapeInfo(), gradI.getShapeInfo()))
        gradITadPack = inTadPack;
    else
        gradITadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(gradI.getShapeInfo(), {rank - 1});

    const Nd4jLong numOfTads = inTadPack.numberOfTads();
    const Nd4jLong tadLen    = input.sizeAt(-1); 
    
    const Nd4jLong* inTadOffsets    = inTadPack.primaryOffsets();        
    const Nd4jLong* gradITadOffsets = gradITadPack.primaryOffsets();

    const Nd4jLong inTadEws    = shape::elementWiseStride(inTadPack.primaryShapeInfo());
    const Nd4jLong gradITadEws = shape::elementWiseStride(gradITadPack.primaryShapeInfo());
    
    const X* inBuff    = reinterpret_cast<X*>(input.getBuffer());
          Y* gradIBuff = reinterpret_cast<Y*>(gradI.getBuffer());    

    const Y tbias  = static_cast<Y>(bias);
    const Y tbeta  = static_cast<Y>(beta);
    const Y talpha = static_cast<Y>(alpha);
    const Y coeff  = talpha * tbeta; 

    if(inTadEws == 1 && gradITadEws == 1) {
        
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const X* x = inBuff    + inTadOffsets[i];
                  Y* y = gradIBuff + gradITadOffsets[i];

            // this loop calculates squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {
                    y[0] = 0;
                    for (uint s = begin; s < end; ++s)
                        y[0] = y[0] + x[s] * x[s];
                }
                else if (begin == 0 && last <= tadLen)
                    y[j] = y[j - 1] + x[end - 1] * x[end - 1];
                else if (begin > 0 && last <= tadLen)
                    y[j] = y[j - 1] + x[end - 1] * x[end - 1] - x[begin - 1] * x[begin - 1];
                else if (begin > 0 && last > tadLen)
                    y[j] = y[j - 1] - x[begin - 1] * x[begin - 1];
                else
                    y[j] = y[j - 1];                
            }

            Y* factor = new Y[tadLen];

            Y prev = 0;
            // second loop calculates derivatives using information gained in first loop above
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);

                Y init = tbias + talpha * y[j];

                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s) {
                        factor[s] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[s], -tbeta - 1);
                        prev = prev + x[s] * factor[s];
                    }
                    y[0] = prev;
                }
                else if(begin == 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[end - 1], -tbeta - 1);
                    y[j] = prev + x[end - 1] * factor[end - 1];
                }
                else if (begin > 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[end - 1], -tbeta - 1);
                    y[j] = prev + x[end - 1] * factor[end - 1] - x[begin - 1] * factor[begin - 1];
                }
                else if (begin > 0 && last > tadLen)
                    y[j] = prev - x[begin - 1] * factor[begin - 1];
                else 
                    y[j] = prev;
                
                if(j != 0)
                    prev = y[j];

                y[j] = factor[j] * init - 2 * x[j] * coeff * prev;                
            }
            
            delete []factor;
        }
    }
    else {

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (uint i = 0; i < numOfTads; ++i) {
            const X* x = inBuff    + inTadOffsets[i];
                  Y* y = gradIBuff + gradITadOffsets[i];

            // this loop calculates squared sum of elements per each j-th element range [j - depth, j + depth + 1]
            // we store each squared sum in corresponding element of y array
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;           
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);
                
                if (j == 0) {
                    y[0] = 0;
                    for (uint s = begin; s < end; ++s)
                        y[0] = y[0] + x[s*inTadEws] * x[s*inTadEws];
                }
                else if (begin == 0 && last <= tadLen)
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws] + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws];
                else if (begin > 0 && last <= tadLen)
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws] + x[(end - 1)*inTadEws] * x[(end - 1)*inTadEws] - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else if (begin > 0 && last > tadLen)
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws] - x[(begin - 1)*inTadEws] * x[(begin - 1)*inTadEws];
                else
                    y[j*gradITadEws] = y[(j - 1)*gradITadEws];              
            }

            Y* factor = new Y[tadLen];

            Y prev = 0;
            // second loop calculates derivatives using information gained in first loop above
            for (uint j = 0; j < tadLen; ++j) {
                const uint begin = nd4j::math::nd4j_max<int>(0, j - depth);
                const uint last  = depth + j + 1;
                const uint end   = nd4j::math::nd4j_min<int>(last, tadLen);

                Y init = tbias + talpha * y[j*gradITadEws];

                if (j == 0) {                    
                    for (uint s = begin; s < end; ++s) {
                        factor[s] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[s*gradITadEws], -tbeta - 1);
                        prev = prev + x[s*inTadEws] * factor[s];
                    }
                    y[0] = prev;
                }
                else if(begin == 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[(end - 1)*gradITadEws], -tbeta - 1);
                    y[j*gradITadEws] = prev + x[(end - 1)*inTadEws] * factor[end - 1];
                }
                else if (begin > 0 && last <= tadLen) {
                    factor[end - 1] = nd4j::math::nd4j_pow<Y, Y, Y>(tbias + talpha * y[(end - 1)*gradITadEws], -tbeta - 1);
                    y[j*gradITadEws] = prev + x[(end - 1)*inTadEws] * factor[end - 1] - x[(begin - 1)*inTadEws] * factor[begin - 1];
                }
                else if (begin > 0 && last > tadLen)
                    y[j*gradITadEws] = prev - x[(begin - 1)*inTadEws] * factor[begin - 1];
                else 
                    y[j*gradITadEws] = prev;
                
                if(j != 0)
                    prev = y[j*gradITadEws];

                y[j*gradITadEws] = factor[j] * init - 2 * x[j*inTadEws] * coeff * prev;                
            }
            
            delete []factor;
        }
    }    
    gradI *= gradO;
}


void lrnBP(nd4j::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
    BUILD_DOUBLE_SELECTOR(input.dataType(), gradO.dataType(), lrnBP_, (input, gradO, gradI, depth, bias, alpha, beta), FLOAT_TYPES, FLOAT_TYPES);
}

}
}
}
