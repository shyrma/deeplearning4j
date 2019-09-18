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
#if NOT_EXCLUDED(OP_lstmLayer)

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

    // notations:
    // bS - batch size
    // sL - sequence length, number of time steps
    // nIn - input size
    // nOut - output size (hidden size)

    const auto dataFormat  = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto direction   = INT_ARG(1);    // direction - left to right, or right to left: 0 = fwd, 1 = bwd, 2 = bidirectional
    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct     = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct     = INT_ARG(3);    // activation for cell state (c)
    const auto outAct      = INT_ARG(4);    // activation for output (h)
    const auto bidirMode   = block.getIArguments()->size() > 5 ? INT_ARG(5) : -1; // mode for bidirectional: 0=concat, 1=sum, 2=extra output dim (in conjunction with format arg 3+), -1 = no bidirectional mode

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole h {h_0, h_1, ... , h_sL-1}, if true, format would be [sL,bS,nOut] (exact shape depends on dataFormat argument)
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto gateActHasAlpha = gateAct == 3 || gateAct == 4 || gateAct == 5 || gateAct == 6 || gateAct == 8;
    const auto cellActHasAlpha = cellAct == 3 || cellAct == 4 || cellAct == 5 || cellAct == 6 || cellAct == 8;
    const auto outActHasAlpha  = outAct  == 3 || outAct  == 4 || outAct  == 5 || outAct  == 6 || outAct  == 8;
    const auto gateActHasBeta  = gateAct == 3 || gateAct == 6;
    const auto cellActHasBeta  = cellAct == 3 || cellAct == 6;
    const auto outActHasBeta   = outAct  == 3 || outAct  == 6;

    uint count = 1;
    const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping
    const auto gateAlpha = gateActHasAlpha ? T_ARG(count++) : 0;
    const auto gateBeta  = gateActHasBeta  ? T_ARG(count++) : 0;
    const auto cellAlpha = cellActHasAlpha ? T_ARG(count++) : 0;
    const auto cellBeta  = cellActHasBeta  ? T_ARG(count++) : 0;
    const auto outAlpha  = outActHasAlpha  ? T_ARG(count++) : 0;
    const auto outBeta   = outActHasBeta   ? T_ARG(count++) : 0;

    const auto x  = INPUT_VARIABLE(0);          // input, for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto Wx = INPUT_VARIABLE(1);          // input weights [nIn,4*nOut] for unidirectional or [2,nIn,4*nOut] for bidirectional
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights [nOut,4*nOut] for unidirectional or [2,nOut,4*nOut] for bidirectional

    count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases, [4*nOut] for unidirectional or [2, 4*nOut] for bidirectional
    const auto seqLen = hasSeqLen ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector, [bS], contains integer values within [0,sL), each element of this vector set max time step per each input in batch, this means there are no calculations for time step > seqLen[index]
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output [bS, nOut] for unidirectional and [2,bS,nOut] for bidirectional
    const auto cI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state [bS, nOut] for unidirectional and [2,bS,nOut] for bidirectional
    const auto Wp     = hasPH     ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights, [3*nOut] for unidirectional, [2, 3*nOut] for bidirectional

    REQUIRE_TRUE(dataFormat < 3 || (dataFormat == 3 && direction == 2 && bidirMode == 2), 0, "LSTM_LAYER operation: if argument dataFormat = 3, then following arguments should be: direction == 2 and bidirMode == 2, but got dataFormat = %i, direction = %i, bidirMode = %i instead !", dataFormat, direction, bidirMode);
    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER operation: cell clipping value should be nonnegative (>=0) !");
    REQUIRE_TRUE(retFullSeq || retLastH || retLastC, 0, "LSTM_LAYER operation: please specify what output arrays to produce !");

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output, [sL,bS,nOut] for unidirectional or [sL,2,bS,nOut] for bidirectional
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step, [bS,nOut] for unidirectional or [2,bS,nOut] for bidirectional
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step, [bS,nOut] for unidirectional or [2,bS,nOut] for bidirectional

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 0 || dataFormat == 3 ? x->sizeAt(0) : ( dataFormat == 1 ? x->sizeAt(1) : x->sizeAt(2) );
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(-2);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(-1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    if(direction != 2) {

        // Wx validation
        if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx));
        // Wr validation
        if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
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
        if(Wx->rankOf() != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx));
        // Wr validation
        if(Wr->rankOf() != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
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

    const auto dataFormat = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto direction  = INT_ARG(1);    // direction - left to right, or right to left: 0 = fwd, 1 = bwd, 2 = bidirectional
    // const auto gateAct     = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    // const auto cellAct     = INT_ARG(3);    // activation for cell state (c)
    // const auto outAct      = INT_ARG(4);    // activation for output (h)
    const auto bidirMode  = block.getIArguments()->size() > 5 ? INT_ARG(5) : -1; // mode for bidirectional: 0=concat, 1=sum, 2=extra output dim (in conjunction with format arg 3+), -1 = no bidirectional mode

    const auto retFullSeq = B_ARG(5);           // indicates whether to return whole h {h_0, h_1, ... , h_sL-1}, if true, format would be [sL,bS,nOut] (exact shape depends on dataFormat argument)
    const auto retLastH   = B_ARG(6);           // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);           // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto x  = INPUT_VARIABLE(0);          // input, for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto Wx = INPUT_VARIABLE(1);          // input weights [nIn,4*nOut] for unidirectional or [2,nIn,4*nOut] for bidirectional
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights [nOut,4*nOut] for unidirectional or [2,nOut,4*nOut] for bidirectional

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 0 || dataFormat == 3 ? x->sizeAt(0) : ( dataFormat == 1 ? x->sizeAt(1) : x->sizeAt(2) );
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(-2);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(-1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    std::vector<Nd4jLong*> shapes;

    // evaluate h shape (output)
    if(retFullSeq) {

        std::vector<Nd4jLong> hShape;

        if(direction != 2 || bidirMode == 1) {      // single direction or bidirectional with sum
            if(dataFormat == 0)
                hShape = {sL, bS, nOut};
            else if(dataFormat == 1)
                hShape = {bS, sL, nOut};
            else if(dataFormat == 2)
                hShape = {bS, nOut, sL};
        }
        else if(bidirMode == 0) {                   // bidirectional with concat

            if(dataFormat == 0)
                hShape = {sL, bS, 2*nOut};
            else if(dataFormat == 1)
                hShape = {bS, sL, 2*nOut};
            else if(dataFormat == 2)
                hShape = {bS, 2*nOut, sL};
        }
        else {                                      // bidirectional with extra output dimension equal to 2
            hShape = {sL, 2, bS, nOut};
        }

        shapes.push_back(ConstantShapeHelper::getInstance()->createShapeInfo(x->dataType(), x->ordering(), hShape));
    }

    // evaluate hL shape (output at last step)
    if(retLastH) {

        std::vector<Nd4jLong> hLShape;

        if(direction != 2 || bidirMode == 1)        // single direction or bidirectional with sum
            hLShape = {bS, nOut};
        else if(bidirMode == 0)                     // bidirectional with concat
            hLShape = {bS, 2*nOut};
        else                                        // bidirectional with extra output dimension equal to 2
            hLShape = {2, bS, nOut};

        shapes.push_back(ConstantShapeHelper::getInstance()->createShapeInfo(x->dataType(), x->ordering(), hLShape));

        if(retLastC)                                // cL and hL have same shapes
            shapes.push_back(shapes[0]);
    }

    // evaluate cL shape (cell state at last step)
    if(retLastC && !retLastH) {

        std::vector<Nd4jLong> cLShape;

        if(direction != 2 || bidirMode == 1)        // single direction or bidirectional with sum
            cLShape = {bS, nOut};
        else if(bidirMode == 0)                     // bidirectional with concat
            cLShape = {bS, 2*nOut};
        else                                        // bidirectional with extra output dimension equal to 2
            cLShape = {2, bS, nOut};

        shapes.push_back(ConstantShapeHelper::getInstance()->createShapeInfo(x->dataType(), x->ordering(), cLShape));
    }

    return new ShapeList(shapes);
}





}
}

#endif