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
// @author Yurii Shyrma, created on 26.02.2018
//


#include<ops/declarable/helpers/addBias.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void addBias_(const NDArray& input, const NDArray& bias, NDArray &output, const bool isNCHW) {

    // bias [oC]

    // if(input_rank == 4)
        // input and output have same shapes: [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    // if(input_rank == 5)
        // input and output have same shapes: [bS, oD, oH, oW, oC] (NHWC) or [bS, oD, oC, oH, oW] (NCHW)
    // else
        // apply applyBroadCast


    const X* x = input.bufferAsT<X>();
    const Y* y = bias.bufferAsT<Y>();
          X* z = output.bufferAsT<X>();

    const bool inOutAreSame = x == z;

    const uint bS           = output.sizeAt(0);              // batch size
    const Nd4jLong yStrideC = bias.stridesOf()[0];
    const Nd4jLong zStrideB = output.stridesOf()[0];

    if(output.rankOf() == 4) {

        const uint C  = isNCHW ? output.sizeAt(1) : output.sizeAt(3);     // channels
        const uint oH = isNCHW ? output.sizeAt(2) : output.sizeAt(1);     // height
        const uint oW = isNCHW ? output.sizeAt(3) : output.sizeAt(2);     // width

        const Nd4jLong zStrideC = isNCHW ? output.stridesOf()[1] : output.stridesOf()[3];
        const Nd4jLong zStrideH = isNCHW ? output.stridesOf()[2] : output.stridesOf()[1];
        const Nd4jLong zStrideW = isNCHW ? output.stridesOf()[3] : output.stridesOf()[2];

        if(inOutAreSame) {

            PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(4))
            for(uint b = 0; b < bS; ++b)
                for(uint c = 0; c < C; ++c)
                    for(uint h = 0; h < oH ; ++h)
                        for(uint w = 0; w < oW ; ++w)
                            z[b*zStrideB + c*zStrideC + h*zStrideH + w*zStrideW] += static_cast<X>(y[c*yStrideC]);
        }
        else {

            const Nd4jLong xStrideB = input.stridesOf()[0];
            const Nd4jLong xStrideC = isNCHW ? input.stridesOf()[1] : input.stridesOf()[3];
            const Nd4jLong xStrideH = isNCHW ? input.stridesOf()[2] : input.stridesOf()[1];
            const Nd4jLong xStrideW = isNCHW ? input.stridesOf()[3] : input.stridesOf()[2];

            PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(4))
            for(uint b = 0; b < bS; ++b)
                for(uint c = 0; c < C; ++c)
                    for(uint h = 0; h < oH ; ++h)
                        for(uint w = 0; w < oW ; ++w)
                            z[b*zStrideB + c*zStrideC + h*zStrideH + w*zStrideW] = x[b*xStrideB + c*xStrideC + h*xStrideH + w*xStrideW] +  static_cast<X>(y[c*yStrideC]);
        }
    }
    else if(output.rankOf() == 5) {

        const uint C  = isNCHW ? output.sizeAt(1) : output.sizeAt(4);     // channels
        const uint oD = isNCHW ? output.sizeAt(2) : output.sizeAt(1);     // depth
        const uint oH = isNCHW ? output.sizeAt(3) : output.sizeAt(2);     // height
        const uint oW = isNCHW ? output.sizeAt(4) : output.sizeAt(3);     // width

        const Nd4jLong zStrideC = isNCHW ? output.stridesOf()[1] : output.stridesOf()[4];
        const Nd4jLong zStrideD = isNCHW ? output.stridesOf()[2] : output.stridesOf()[1];
        const Nd4jLong zStrideH = isNCHW ? output.stridesOf()[3] : output.stridesOf()[2];
        const Nd4jLong zStrideW = isNCHW ? output.stridesOf()[4] : output.stridesOf()[3];

        if(inOutAreSame) {

            PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(5))
            for(uint b = 0; b < bS; ++b)
                for(uint c = 0; c < C; ++c)
                    for(uint d = 0; d < oD ; ++d)
                        for(uint h = 0; h < oH ; ++h)
                            for(uint w = 0; w < oW ; ++w)
                                z[b*zStrideB + c*zStrideC + d*zStrideD + h*zStrideH + w*zStrideW] += static_cast<X>(y[c*yStrideC]);
        }
        else {

            const Nd4jLong xStrideB = input.stridesOf()[0];
            const Nd4jLong xStrideC = isNCHW ? input.stridesOf()[1] : input.stridesOf()[4];
            const Nd4jLong xStrideD = isNCHW ? input.stridesOf()[2] : input.stridesOf()[1];
            const Nd4jLong xStrideH = isNCHW ? input.stridesOf()[3] : input.stridesOf()[2];
            const Nd4jLong xStrideW = isNCHW ? input.stridesOf()[4] : input.stridesOf()[3];

            PRAGMA_OMP_PARALLEL_FOR_ARGS(collapse(5))
            for(uint b = 0; b < bS; ++b)
                for(uint c = 0; c < C; ++c)
                    for(uint d = 0; d < oD ; ++d)
                        for(uint h = 0; h < oH ; ++h)
                            for(uint w = 0; w < oW ; ++w)
                                z[b*zStrideB + c*zStrideC + d*zStrideD + h*zStrideH + w*zStrideW] = x[b*xStrideB + c*xStrideC + d*xStrideD + h*xStrideH + w*xStrideW] + static_cast<X>(y[c*yStrideC]);
        }
    }
    else {
         const int channelDim = isNCHW ? 1 : input.rankOf() - 1;      // second or last
         const_cast<NDArray&>(input).applyBroadcast(nd4j::broadcast::Add, {channelDim}, &bias, &output);
    }
}

//////////////////////////////////////////////////////////////////////////
void addBias(nd4j::graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW) {

    // bias.rankOf() == 1 ? bias : bias.reshape(bias.ordering(), {bias.lengthOf()})
    BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBias_, (input, bias, output, isNCHW), FLOAT_TYPES, FLOAT_TYPES);
}


BUILD_DOUBLE_TEMPLATE(template void addBias_, (const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW), FLOAT_TYPES, FLOAT_TYPES);

}
}
}

