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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <MmulHelper.h>
#include <NDArrayFactory.h>
#include <Status.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void swapRows_(NDArray* matrix, int theFirst, int theSecond) {

        if (theFirst != theSecond)
            for (int i = 0; i < matrix->columns(); i++) {
                T e0 = matrix->e<T>(theFirst, i);
                T e1 = matrix->e<T>(theSecond, i);

                matrix->p<T>(theFirst, i, e1);
                matrix->p<T>(theSecond, i, e0);
            }
    }
    BUILD_SINGLE_TEMPLATE(template void swapRows_, (NDArray* matrix, int theFirst, int theSecond), FLOAT_TYPES);

    void swapRows(NDArray* matrix, int theFirst, int theSecond) {
        BUILD_SINGLE_SELECTOR(matrix->dataType(), swapRows_, (matrix, theFirst, theSecond), FLOAT_TYPES);
    }

    template <typename T>
    static void invertLowerMatrix_(NDArray* inputMatrix, NDArray* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->assign(0.f);

       // PRAGMA_OMP_PARALLEL_FOR_IF(n > Environment::getInstance()->elementwiseThreshold())
        for (int i = 0; i < n; i++)
            invertedMatrix->p(i, i, 1.0f);

        if (inputMatrix->isIdentityMatrix()) return;

        //PRAGMA_OMP_PARALLEL_FOR_IF(n > Environment::getInstance()->elementwiseThreshold())
        for (int i = 1; i < n; i++)
            invertedMatrix->t<T>(i, i - 1) = -inputMatrix->t<T>(i, i - 1);

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i = 2; i < n; i++) {
            for (int j = i - 2; j > -1; --j)
                for (int k = 0; k < i; k++)
                    invertedMatrix->t<T>(i, j) -= (invertedMatrix->t<T>(k, j) * inputMatrix->t<T>(i, k));
        }
    }

    BUILD_SINGLE_TEMPLATE(template void invertLowerMatrix_, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_TYPES);

    void invertLowerMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertLowerMatrix_, (inputMatrix, invertedMatrix), FLOAT_TYPES);
    }

    template <typename T>
    static void _invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->setIdentity();

        if (inputMatrix->isIdentityMatrix()) { // the inverse for I is I
            return;
        }

        //PRAGMA_OMP_PARALLEL_FOR_IF(n > Environment::getInstance()->elementwiseThreshold())
        for (int i = 0; i < n; i++)
            invertedMatrix->t<T>(i, i) /= inputMatrix->t<T>(i, i);

        //PRAGMA_OMP_PARALLEL_FOR_IF(n > Environment::getInstance()->elementwiseThreshold())
        for (int i = 0; i < n - 1; i++)
            invertedMatrix->t<T>(i, i + 1) -= (inputMatrix->t<T>(i, i + 1) * invertedMatrix->t<T>(i + 1, i + 1) / inputMatrix->t<T>(i, i));

//        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i = n - 2; i > - 1; i--) {
            for (int j = i + 2; j < n; j++)
                for (int k = i; k < n; k++)
                    invertedMatrix->t<T>(i, j) -= ((invertedMatrix->t<T>(k, j) * inputMatrix->t<T>(i, k) / inputMatrix->t<T>(i, i)));
        }
    }

    BUILD_SINGLE_TEMPLATE(template void _invertUpperMatrix, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_TYPES);

    void invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), _invertUpperMatrix, (inputMatrix, invertedMatrix), FLOAT_TYPES);
    }


    template <typename T>
    static NDArray lup_(LaunchContext *context, NDArray* input, NDArray* compound, NDArray* permutation) {

        const int rowNum = input->rows();
        const int columnNum = input->columns();

        NDArray determinant = NDArrayFactory::create<T>(1.f);
        NDArray compoundMatrix = *input; // copy
        NDArray permutationMatrix(input, false, context); // has same shape as input and contiguous strides
        permutationMatrix.setIdentity();

        T pivotValue; // = T(0.0);
        int pivot; // = -1;
        int swapCount = 0;

        for(int i = 0; i < rowNum; i++ ) {
            pivotValue = T(0.0);
            pivot = -1;
            //PRAGMA_OMP_PARALLEL_FOR //_ARGS(firstprivate(pivot,pivotValue))
            for(int rowCounter = i; rowCounter < rowNum; rowCounter++ ) {
                if (nd4j::math::nd4j_abs(compoundMatrix.t<T>(rowCounter, i)) > pivotValue) {
                    pivotValue = nd4j::math::nd4j_abs(compoundMatrix.t<T>(rowCounter, i));
                    pivot = rowCounter;
                }
            }

            if( pivotValue > T(0.00001)) {
                swapRows(&compoundMatrix, pivot, i);
                swapRows(&permutationMatrix, pivot, i);
                if (pivot != i)
                    swapCount++;

                for( int j = i + 1; j < rowNum; j++ ) {
                    compoundMatrix.t<T>(j, i) /= compoundMatrix.t<T>(i, i);
                    //PRAGMA_OMP_PARALLEL_FOR
                    for( int k = i + 1; k < rowNum; k++ ) {
                        compoundMatrix.t<T>(j, k) -= compoundMatrix.t<T>(j, i) * compoundMatrix.t<T>(i, k);
                    }
                }
            }
        }

        for (int e = 0; e < rowNum; e++) {
            // nd4j_printf("Compound matrix diag %i %f.\n", e, (*compoundMatrix)(e, e));
            determinant *= compoundMatrix.e<T>(e, e);
        }
        if (swapCount % 2) determinant = -determinant;
        if (compound != nullptr)
            compound->assign(compoundMatrix);
        if (permutation != nullptr)
            permutation->assign(permutationMatrix);
        return determinant;
    }

    BUILD_SINGLE_TEMPLATE(template NDArray lup_, (LaunchContext *context, NDArray* input, NDArray* output, NDArray* permutation), FLOAT_TYPES);



    template <typename T>
    static int determinant_(LaunchContext *context, NDArray* input, NDArray* output) {

        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;

        auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), context); //, block.getWorkspace());

        for (int e = 0; e < output->lengthOf(); e++) {
            for (int k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row)
                matrix.p(row, input->e<T>(k));
            output->p(e, lup_<T>(context, &matrix, (NDArray*)nullptr, (NDArray*)nullptr));
        }

        return Status::OK();
    }

    int determinant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), FLOAT_TYPES);
    }

template <typename T>
    int logAbsDeterminant_(LaunchContext *context, NDArray* input, NDArray* output) {

        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;

        NDArray matrix = NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), context); //, block.getWorkspace());
        for (int e = 0; e < output->lengthOf(); e++) {
            for (int k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row) {
                matrix.p(row, input->e<T>(k));
            }
	    NDArray det = lup_<T>(context, &matrix, (NDArray*)nullptr, (NDArray*)nullptr);
	    if (det.e<T>(0) != 0.f)
             	output->p(e, nd4j::math::nd4j_log<T,T>(nd4j::math::nd4j_abs(det.t<T>(0))));
        }

        return ND4J_STATUS_OK;
    }

    int logAbsDeterminant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), FLOAT_TYPES);
    }

    template <typename T>
    static int inverse_(LaunchContext *context, NDArray* input, NDArray* output) {

        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        auto totalCount = output->lengthOf() / n2;

        output->assign(0.f); // fill up output tensor with zeros
        auto matrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
        auto compound = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
        auto permutation = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto lowerMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto upperMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);

        for (int e = 0; e < totalCount; e++) {
            if (e)
                matrix.assign(0.f);

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                matrix.p(row++, input->e<T>(k));
            }
            T det = lup_<T>(context, &matrix, &compound, &permutation).template e<T>(0);

            // FIXME: and how this is going to work on float16?
            if (nd4j::math::nd4j_abs<T>(det) < T(0.000001)) {
                nd4j_printf("matrix_inverse: The matrix %i has no inverse due determinant is %lf. Quiting...\n", e, det);
                matrix.printIndexedBuffer("Wrong matrix");
                return ND4J_STATUS_VALIDATION;
            }
            lowerMatrix.setIdentity(); // set up U to identity matrix
            for (int k = 1; k < n; k++) {  // and then put all values under main diagonal on to it
                for (int j = 0; j < k; j++)
                    lowerMatrix.template t<T>(k, j) = compound.template t<T>(k, j);
            }
            upperMatrix.setIdentity(); // set up U to identity matrix
            for (int k = 0; k < n; k++) {  // and then put all values under main diagonal on to it
                for (int j = k; j < n; j++)
                    upperMatrix.template t<T>(k, j) = compound.template e<T>(k, j);
            }
            invertUpperMatrix(&upperMatrix, &matrix);

            invertLowerMatrix(&lowerMatrix, &upperMatrix);

            nd4j::MmulHelper::mmul(&matrix, &upperMatrix, &compound, 1.0, 0.0);
            nd4j::MmulHelper::mmul(&compound, &permutation, &matrix, 1.0, 0.0);
            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                output->t<T>(k) = matrix.template t<T>(row++);
            }
        }

        return Status::OK();
    }

    int inverse(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), FLOAT_TYPES);
    }

    template <typename T>
    static bool checkCholeskyInput_(nd4j::LaunchContext * context, NDArray const* input) {
        //std::unique_ptr<NDArray> matrix(NDArrayFactory::create_('c', {n, n}, input->dataType())); //, block.getWorkspace());
        std::unique_ptr<ResultSet> lastMatrixList(input->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf()-1}));
        for (size_t i = 0; i < lastMatrixList->size(); i++) {
            auto thisMatrix = lastMatrixList->at(i);
            // check for symmetric
            for (Nd4jLong r = 0; r < thisMatrix->rows(); r++)
                for (Nd4jLong c = 0; c < thisMatrix->columns(); c++)
                    if (nd4j::math::nd4j_abs(thisMatrix->e<T>(r, c) - lastMatrixList->at(i)->e<T>(c,r)) > T(1.e-6f)) return false;

            NDArray output = NDArrayFactory::create<T>(0., context);
            if (ND4J_STATUS_OK != determinant(context, thisMatrix, &output)) return false;
            if (output.e<T>(0) <= T(0)) return 0;
            NDArray reversedMatrix(*thisMatrix);
            if (ND4J_STATUS_OK != inverse(context, thisMatrix, &reversedMatrix)) return false;
            if (ND4J_STATUS_OK != determinant(context, &reversedMatrix, &output)) return false;
            if (output.e<T>(0) <= T(0)) return 0;

        }


        return true;
    }

    bool checkCholeskyInput(nd4j::LaunchContext * context, NDArray const* input) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return checkCholeskyInput_, (context, input), FLOAT_TYPES);
    }

    template <typename T>
    int cholesky_(LaunchContext *context, NDArray* input, NDArray* output, bool inplace) {

        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        auto totalCount = output->lengthOf() / n2;
        if (!inplace)
             output->assign(0.f); // fill up output tensor with zeros only inplace=false

        std::unique_ptr<NDArray> matrix(NDArrayFactory::create_('c', {n, n}, input->dataType(), context)); //, block.getWorkspace());
        std::unique_ptr<NDArray> lowerMatrix(NDArrayFactory::create_('c',{n, n}, input->dataType(), context));

        for (int e = 0; e < totalCount; e++) {

            // fill up matrix
            for (int k = e * n2, l = 0; k < (e + 1) * n2; k++) {
                matrix->p(l++, input->e<T>(k));
            }
            //if (e) // from the second loop need to zero matrix
            lowerMatrix->assign(0.f);

            for (Nd4jLong col = 0; col < n; col++) {
                for (Nd4jLong row = 0; row < col; row++) {
                    T rowSum = 0;
                    for (Nd4jLong k = 0; k < row; ++k)
                        rowSum += (lowerMatrix->e<T>(col, k) * lowerMatrix->e<T>(row, k));
                    lowerMatrix->p(col, row, (matrix->e<T>(row, col) - rowSum) / lowerMatrix->e<double>(row, row));
                }
                T diagonalSum = 0;
                for (Nd4jLong k = 0; k < col;  ++k)
                    diagonalSum += lowerMatrix->e<T>(col, k) * lowerMatrix->e<T>(col, k);
                lowerMatrix->p(col, col, nd4j::math::nd4j_sqrt<T, T>(matrix->e<T>(col, col) - diagonalSum));
                //nd4j_printf("%i: ", col);
                //lowerMatrix->printIndexedBuffer("Lower matrix");
            }
            for (int k = e * n2, l = 0; k < (e + 1) * n2; k++) {
                output->p(k, lowerMatrix->e<T>(l++));
            }
        }

        return ND4J_STATUS_OK;
    }

    int cholesky(nd4j::LaunchContext * context, NDArray* input, NDArray* output, bool inplace) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return cholesky_, (context, input, output, inplace), FLOAT_TYPES);
    }

    template <typename T>
    int logdetFunctor_(LaunchContext *context, NDArray* input, NDArray* output) {
        std::unique_ptr<NDArray> tempOutput(input->dup());
        int res = cholesky_<T>(context, input, tempOutput.get(), false);
        if (res != ND4J_STATUS_OK)
            return res;
        auto n = input->sizeAt(-1);
        auto totalCount = output->lengthOf();
        std::vector<T> d(n);
        std::unique_ptr<ResultSet> matricies(tempOutput->allTensorsAlongDimension({input->rankOf()-2, input->rankOf() - 1}));
        std::unique_ptr<ResultSet> inputMatricies(input->allTensorsAlongDimension({input->rankOf()-2, input->rankOf() - 1}));
        for (Nd4jLong e = 0; e < totalCount; e++) {

            //d[0] = inputMatricies->at(e)->t<T>(0, 0);
            for (size_t i = 0; i < n; ++i) {
                output->t<T>(e) += nd4j::math::nd4j_log<T,T>(nd4j::math::nd4j_pow<T,T,T>(matricies->at(e)->t<T>(i, i), T(2)));
            }
        }
        return ND4J_STATUS_OK;
    }

    int logdetFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return logdetFunctor_, (context, input, output), FLOAT_TYPES);
    }

}
}
}
