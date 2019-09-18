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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_LSTMLAYER_H
#define LIBND4J_LSTMLAYER_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/activations.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray applyActivation(const NDArray& arr, const int opId, const float alpha = 0, const float beta = 0) {

    NDArray result(&arr, arr.getContext());

    switch (opId) {
        case 0:
            (const_cast<NDArray&>(arr)).applyTransform(transform::Tanh, &result);
            break;
        case 1:
            (const_cast<NDArray&>(arr)).applyScalar<float>(scalar::RELU, 0, &result);
            break;
        case 2:
            (const_cast<NDArray&>(arr)).applyTransform(transform::Sigmoid, &result);
            break;
        case 3: {
            ExtraArguments args({ static_cast<double>(alpha), static_cast<double>(beta)});
            (const_cast<NDArray&>(arr)).applyTransform(transform::Affine, &result, &args);
            break;
        }
        case 4:
            (const_cast<NDArray&>(arr)).applyScalar<float>(scalar::LeakyRELU, alpha, &result);
            break;
        case 5:
            helpers::thresholdRelu(arr.getContext(), arr, alpha, result);
            break;
        case 6: {
            ExtraArguments args({ static_cast<double>(alpha), static_cast<double>(beta)});
            (const_cast<NDArray&>(arr)).applyTransform(transform::ScaledTanh, &result, &args);
            break;
        }
        case 7:
            (const_cast<NDArray&>(arr)).applyTransform(transform::HardSigmoid, &result);
            break;
        case 8:
            (const_cast<NDArray&>(arr)).applyScalar<float>(scalar::ELU, alpha, &result);
            break;
        case 9:
            (const_cast<NDArray&>(arr)).applyTransform(transform::SoftSign, &result);
            break;
        case 10:
            (const_cast<NDArray&>(arr)).applyTransform(transform::SoftPlus, &result);
            break;
        default:
            throw std::invalid_argument("LSTM_LAYER operation: wrong id number of activation !");
    }

    return result;
}



}
}
}


#endif //LIBND4J_LSTMLAYER_H
