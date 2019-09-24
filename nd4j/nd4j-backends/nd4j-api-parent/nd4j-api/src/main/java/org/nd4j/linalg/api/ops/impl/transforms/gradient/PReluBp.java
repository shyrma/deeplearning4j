/*
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
 */

package org.nd4j.linalg.api.ops.impl.transforms.gradient;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * PRelu backpropagation op - dL/dIn from in and dL/dOut
 */
@NoArgsConstructor
public class PReluBp extends DynamicCustomOp {

    @Getter
    protected int[] sharedAxes;

    public PReluBp(SameDiff sd, SDVariable input, SDVariable alpha, SDVariable gradient, int... sharedAxes){
        super(sd, new SDVariable[]{input, alpha, gradient});
        this.sharedAxes = sharedAxes;
        addIArgument(sharedAxes);
    }

    public PReluBp(@NonNull INDArray input, @NonNull INDArray alpha, @NonNull INDArray gradient, INDArray dLdI, INDArray dLdA, int... sharedAxes){
        super(new INDArray[]{input, alpha, gradient}, wrapFilterNull(dLdI, dLdA));
        this.sharedAxes = sharedAxes;
        addIArgument(sharedAxes);
    }

    @Override
    public String opName(){
        return "prelu_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions
                .checkArgument(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatypes, got %s", dataTypes);
        Preconditions.checkArgument(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType() && dataTypes.get(2).isFPType(), "Input datatypes must be floating point, got %s", dataTypes);

        return Arrays.asList(dataTypes.get(0), dataTypes.get(1));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");
    }
}
