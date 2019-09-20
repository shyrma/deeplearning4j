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

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.


#include<ops/declarable/helpers/lstm.h>
#include <VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <array/NDArrayList.h>
#include <iterator>
#include <MmulHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                   NDArray* h, NDArray* c) {


    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** the objective is to provide math-readable code **/

    // equations (no peephole connections)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  bo)
    // ht  = ot ◦ tanh(ct)

    // equations (peephole connections are present)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  Wpi ◦ ct-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  Wpf ◦ ct-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  Wpo ◦ ct   +  bo)
    // ht  = ot ◦ tanh(ct)


    // IDs for activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus

    // params[2] - cell clipping value, if it = 0 then do not apply clipping

    // params[3]  - activation ID for input (i), forget (f) and output (o) gates
    // params[4]  - alpha value for gates activation
    // params[5]  - beta value for gates activation

    // params[6]  - activation ID for cell state (c)
    // params[7]  - alpha value for cell state activation
    // params[8]  - beta value for cell state activation

    // params[9]  - activation ID for output (h)
    // params[10] - alpha value for output activation
    // params[11] - beta value for output activation

    // INPUTS:
    // x  - current input [bS, nIn] at time t
    // Wx - input weights [nIn, 4*nOut]
    // Wr - recurrent weights [nOut, 4*nOut]
    // b  - biases [4*nOut], optional, may be nullptr
    // hI - previous (initial) output at time t-1 [bS, nOut], optional, may be nullptr
    // cI - previous (initial) cell state at time t-1 [bS, nOut], optional, may be nullptr
    // Wp - peephole weights [3*nOut], optional, may be nullptr

    // OUTPUTS:
    // h - current output [bS, nOut], that is at current time step t
    // c - current cell state [bS, nOut], that is at current time step t

    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    auto z = mmul(*x, *Wx) + mmul(*hI, *Wr);       //  [bs, nIn] * [nIn, 4*nOut] + [bs, nOut] * [nOut, 4*nOut] = [bS, 4*nOut]

    // add biases if they are given
    if(b != nullptr)
        z += *b;                                    // broadcast [bS, 4*nOut] + [4*nOut] = [bS, 4*nOut]

    auto zi = z({0,0, 0,        nOut});         // input gate it, [bS, nOut]
    auto zf = z({0,0, nOut,   2*nOut});         // forget gate ft, [bS, nOut]
    auto zc = z({0,0, 2*nOut, 3*nOut});         // cell gate c't, [bS, nOut]
    auto zo = z({0,0, 3*nOut, 4*nOut});         // output gate ot, [bS, nOut]

    // peephole connections for input and forget gates
    if(Wp != nullptr) {
        zi += *cI * Wp({0,      nOut});      // broadcast: [bS, nOut] + [bS, nOut] ◦ [nOut] = [bS, nOut]
        zf += *cI * Wp({nOut, 2*nOut});      // broadcast: [bS, nOut] + [bS, nOut] ◦ [nOut] = [bS, nOut]
    }

    applyActivation(zi, params[3], params[4], params[5], zi);   // inplace
    applyActivation(zf, params[3], params[4], params[5], zf);   // inplace
    applyActivation(zc, params[6], params[7], params[8], zc);   // inplace

    c->assign(zf * *cI + zi * zc);          // [bS, nOut] ◦ [bS, nOut] + [bS, nOut] ◦ [bS, nOut] = [bS, nOut]

    // if clipping value is non-zero then cell state is clipped by this value prior to the cell output activation
    if(params[0] != 0)
        c->applyScalar(scalar::LstmClip, params[2]);

    // peephole connections for output gate
    if(Wp != nullptr) {
        zo += *c * Wp({2*nOut, 3*nOut});    // broadcast: [bS, nOut] + [nOut] ◦ [bS, nOut] = [bS, nOut]

    applyActivation(zo, params[3], params[4], params[5], zo);

    applyActivation(*c, params[9], params[10], params[11], *h);
    *h *= zo;                               // [bS, nOut] ◦ [bS, nOut]
}



//////////////////////////////////////////////////////////////////////////
void lstmLayerTimeLoop(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                       const NDArray* b, const NDArray* seqLen, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                       const std::vector<float>& params,
                       const bool forward,
                       NDArray* h, NDArray* hL, NDArray* cL) {

    // INPUTS:
    // x  - current input  [sL, bS, nIn],  [bS, sL, nIn],  [bS, nIn, sL],
    // Wx - input weights [nIn, 4*nOut]
    // Wr - recurrent weights [nOut, 4*nOut]
    // b  - biases [4*nOut], optional, may be nullptr
    // seqLen - [bS], optional, may be nullptr
    // hI - initial output  [bS, nOut], optional, may be nullptr
    // cI - initial cell state at time t-1 [bS, nOut], optional, may be nullptr
    // Wp - peephole weights [3*nOut], optional, may be nullptr

    // OUTPUTS:
    // h - output [sL, bS, nOut],  [bS, sL, nOut],  [bS, nOut, sL], optional, may be nullptr
    // hL - output at last step [bS, nOut], optional, may be nullptr
    // cL - cell state at last step [bS, nOut], optional, may be nullptr

    // params = {dataFormat, directionMode, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};

    // time
    const Nd4jLong sL   = x->sizeAt(params[0]);
    const Nd4jLong bS   = params[0] == 1 || params[0] == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;


    int t, stop, step;

    if(forward) {
        t = 0; stop = sL; step = 1;
    }
    else {
        t = sL - 1; stop = -1; step = -1;
    }

    // 1) [sL, bS, nOut]    when directionMode <= 2 && dataFormat == 0
    // 2) [bS, sL, nOut]    when directionMode <= 2 && dataFormat == 1
    // 3) [bS, nOut, sL]    when directionMode <= 2 && dataFormat == 2

    NDArray* ht = const_cast<NDArray*>(hI);
    if(hI == nullptr) {
         ht = hL;
         if(hL == nullptr)
             ht = new NDArray(x->ordering(), {bS, nOut}, x->dataType(), x->getContext());
    }
    NDArray* ct(*c0);

    if(seqLen == nullptr) {

        // loop through time steps
        while (t != stop) {

            auto xt = lstmLayertimeSubset(*x, t, params[0]) {
            auto ht = (*h)({t,t+1, 0,0, 0,0});
            auto ct = (*c)({t,t+1, 0,0, 0,0});

            helpers::lstmLayerCell(context, &xt,&currentH,&currentC, Wx,Wh,Wc,Wp, b,   &ht, &ct,   params);
            currentH.assign(ht);
            currentC.assign(ct);

            t += step;
        }
    }


}



}
}
}

