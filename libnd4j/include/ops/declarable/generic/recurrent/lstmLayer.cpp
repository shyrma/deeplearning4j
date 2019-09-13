/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstm)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmLayer.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmLayer, 3, 1, false, 1, 4) {

    // equations (no peephole connections)
    // it  =    σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  =    σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // ot  =    σ(Wxo * xt  +  Wro * ht-1  +  bo)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct = ft ◦ ct-1 + it ◦ c't
    // ht = ot ◦ tanh(ct)

    // equations (peephole connections are present)
    // it  =    σ(Wxi * xt  +  Wri * ht-1  +  Wpi * ct-1  +  bi)
    // ft  =    σ(Wxf * xt  +  Wrf * ht-1  +  Wpf * ct-1  +  bf)
    // ot  =    σ(Wxo * xt  +  Wro * ht-1  +  Wpo * ct-1  +  bo)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  Wpc * ct-1  +  bc)
    // ct = ft ◦ ct-1 + it ◦ c't
    // ht = ot ◦ tanh(ct)

    // bS - batch size
    // sL - sequence length, number of time steps

    const auto dataFormat  = INT_ARG(0)   // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto direction   = INT_ARG(1);  // direction - left to right, or right to left: 0 = fwd, 1 = bwd, 2 = bidirectional

    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct     = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct     = INT_ARG(3);    // activation for cell state (c)
    const auto outAct      = INT_ARG(4);    // activation for output (h)

    const auto bidirMode   = block.getIArguments() > 5 ? INT_ARG(5) : -1; // mode for bidirectional: 0=concat, 1=sum, 2=extra output dim (in conjunction with format arg 3+), -1 = no bidirectional mode

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole h {h_0, h_1, ... , h_sL-1}, if true, format would be [sL,bS,nOut] (exact shape depends on dataFormat argument)
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto gateActHasAlpha = gateAct == 3 || gateAct == 4 || gateAct == 5 || gateAct == 6 || gateAct == 7 || gateAct == 8;
    const auto cellActHasAlpha = cellAct == 3 || cellAct == 4 || cellAct == 5 || cellAct == 6 || cellAct == 7 || cellAct == 8;
    const auto outActHasAlpha  = outAct  == 3 || outAct  == 4 || outAct  == 5 || outAct  == 6 || outAct  == 7 || outAct  == 8;

    const auto gateActHasBeta = gateAct == 3 || gateAct == 6 || gateAct == 7;
    const auto cellActHasBeta = cellAct == 3 || cellAct == 6 || cellAct == 7;
    const auto outActHasBeta  = outAct  == 3 || outAct  == 6 || outAct  == 7;

    uint count = 1;
    const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping
    const auto gateAlpha = gateActHasAlpha ? T_ARG(count++) : 0;
    const auto gateBeta  = gateActHasBeta  ? T_ARG(count++) : 0;
    const auto cellAlpha = cellActHasAlpha ? T_ARG(count++) : 0;
    const auto cellBeta  = cellActHasBeta  ? T_ARG(count++) : 0;
    const auto outAlpha  = outActHasAlpha  ? T_ARG(count++) : 0;
    const auto outBeta   = outActHasBeta   ? T_ARG(count++) : 0;

    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER operation: cell clipping value should be nonnegative (>=0) !");

    const auto x  = INPUT_VARIABLE(0);          // input, for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto Wx = INPUT_VARIABLE(1);          // input weights [nIn,4*nOut] for unidirectional or [2,nIn,4*nOut] for bidirectional
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights [nOut,4*nOut] for unidirectional or [2,nOut,4*nOut] for bidirectional

    count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases, [4*nOut] for unidirectional or [2, 4*nOut] for bidirectional
    const auto seqLen = hasSeqLen ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector, [bS], contains integer values within [0,sL), each element of this vector set max time step per each input in batch, this means there are no calculations for time step > seqLen[index]
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output [bS, nOut] for unidirectional and [2,bS,nOut] for bidirectional
    const auto cI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state [bS, nOut] for unidirectional and [2,bS,nOut] for bidirectional
    const auto Wp     = hasPH     ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights, [3*nOut] for unidirectional, [2, 3*nOut] for bidirectional

    REQUIRE_TRUE(retFullSeq || retLastH || retLastC, 0, "LSTM_LAYER operation: please specify what output arrays to produce !");

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output, [sL,bS,nOut] for unidirectional or [sL,2,bS,nOut] for bidirectional
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step, [bS,nOut] for unidirectional or [2,bS,nOut] for bidirectional
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step, [bS,nOut] for unidirectional or [2,bS,nOut] for bidirectional

    // evaluate dimensions
    const int sL   = dataFormat == 0 || dataFormat == 3 ? x->sizeAt(0) : ( dataFormat == 1 ? x->sizeAt(1) : x->sizeAt(2) );
    const int bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(-2);
    const int nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(-1);
    const int nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    if(bidirectional != 2) {

        // Wx validation
        if(Wx->rankOf(0) != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx));
        // Wr validation
        if(Wr->rankOf(0) != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr));
        // biases validation
        if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b));
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 2 || hI->sizeAt(0) != bS || hI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI));
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 2 || cI->sizeAt(0) != bS || cI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI));
        // peephole weights validation
        if(Wp != nullptr && (Wp->rankOf() != 1 || Wp->sizeAt(0) != 3*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp));
    }
    else {
         // Wx validation
        if(Wx->rankOf(0) != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx));
        // Wr validation
        if(Wr->rankOf(0) != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr));
        // biases validation
        if(b != nullptr && (b->rankOf() != 2 || b->sizeAt(0) != 2 || b->sizeAt(1) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 4*nOut}).c_str(), ShapeUtils::shapeAsString(b));
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 3 || hI->sizeAt(0) != 2 || hI->sizeAt(1) != bS || hI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI));
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 3 || cI->sizeAt(0) != 2 || cI->sizeAt(1) != bS || cI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI));
        // peephole weights validation
        if(Wp != nullptr && (Wp->rankOf() != 2 || Wp->sizeAt(0) != 2 || Wp->sizeAt(1) != 3*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp));
    }

    return Status::OK();
}

        DECLARE_TYPES(lstmLayer) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }


DECLARE_SHAPE_FN(lstmLayer) {

    auto xShapeInfo  = inputShape->at(0);                    // input [time x bS x nIn]
    auto h0ShapeInfo = inputShape->at(1);                    // initial cell output (at time step = 0) [bS x nOut], in case of projection=false -> nOut == nOut !!!
    auto c0ShapeInfo = inputShape->at(2);                    // initial cell state  (at time step = 0) [bS x nOut],

    auto WxShapeInfo = inputShape->at(3);                   // input-to-hidden  weights, [nIn  x 4*nOut]
    auto WhShapeInfo = inputShape->at(4);                   // hidden-to-hidden weights, [nOut x 4*nOut]
    auto WcShapeInfo = inputShape->at(5);                   // diagonal weights for peephole connections [3*nOut]
    auto WpShapeInfo = inputShape->at(6);                   // projection weights [nOut x nOut]
    auto bShapeInfo  = inputShape->at(7);                   // biases, [4*nOut]

    const int rank     = xShapeInfo[0];
    const int time     = xShapeInfo[1];
    const int bS       = xShapeInfo[2];
    const int nIn   = xShapeInfo[3];
    const int nOut  = h0ShapeInfo[2];
    const int nOut = c0ShapeInfo[2];

    // input shapes validation
    const std::string h0Shape        = ShapeUtils::shapeAsString(h0ShapeInfo);
    const std::string correctH0Shape = ShapeUtils::shapeAsString({bS, nOut});
    const std::string c0Shape        = ShapeUtils::shapeAsString(c0ShapeInfo);
    const std::string correctC0Shape = ShapeUtils::shapeAsString({bS, nOut});
    const std::string WxShape        = ShapeUtils::shapeAsString(WxShapeInfo);
    const std::string correctWxShape = ShapeUtils::shapeAsString({nIn, 4*nOut});
    const std::string WhShape        = ShapeUtils::shapeAsString(WhShapeInfo);
    const std::string correctWhShape = ShapeUtils::shapeAsString({nOut, 4*nOut});
    const std::string WcShape        = ShapeUtils::shapeAsString(WcShapeInfo);
    const std::string correctWcShape = ShapeUtils::shapeAsString({3*nOut});
    const std::string WpShape        = ShapeUtils::shapeAsString(WpShapeInfo);
    const std::string correctWpShape = ShapeUtils::shapeAsString({nOut, nOut});
    const std::string bShape         = ShapeUtils::shapeAsString(bShapeInfo);
    const std::string correctBShape  = ShapeUtils::shapeAsString({4*nOut});

    REQUIRE_TRUE(correctH0Shape == h0Shape, 0, "LSTM operation: wrong shape of initial cell output, expected is %s, but got %s instead !", correctH0Shape.c_str(), h0Shape.c_str());
    REQUIRE_TRUE(correctC0Shape == c0Shape, 0, "LSTM operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", correctC0Shape.c_str(), c0Shape.c_str());
    REQUIRE_TRUE(correctWxShape == WxShape, 0, "LSTM operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", correctWxShape.c_str(), WxShape.c_str());
    REQUIRE_TRUE(correctWhShape == WhShape, 0, "LSTM operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", correctWhShape.c_str(), WhShape.c_str());
    REQUIRE_TRUE(correctWcShape == WcShape, 0, "LSTM operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", correctWcShape.c_str(), WcShape.c_str());
    REQUIRE_TRUE(correctWpShape == WpShape, 0, "LSTM operation: wrong shape of projection weights, expected is %s, but got %s instead !", correctWpShape.c_str(), WpShape.c_str());
    REQUIRE_TRUE(correctBShape  == bShape,  0, "LSTM operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str());


    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [time x bS x nOut]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [time x bS x nOut]

    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = time;
    hShapeInfo[2] = cShapeInfo[2] = bS;
    hShapeInfo[3] = nOut;
    cShapeInfo[3] = nOut;

    ShapeUtils::updateStridesAndType(hShapeInfo, xShapeInfo, shape::order(h0ShapeInfo));
    ShapeUtils::updateStridesAndType(cShapeInfo, xShapeInfo, shape::order(c0ShapeInfo));

    return SHAPELIST(CONSTANT(hShapeInfo), CONSTANT(cShapeInfo));
}





}
}

#endif