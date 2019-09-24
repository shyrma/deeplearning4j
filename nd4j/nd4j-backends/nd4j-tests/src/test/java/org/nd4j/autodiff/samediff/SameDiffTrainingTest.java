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

package org.nd4j.autodiff.samediff;

import static org.junit.Assert.assertTrue;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.weightinit.impl.XavierInitScheme;

@Slf4j
public class SameDiffTrainingTest extends BaseNd4jTest {

    public SameDiffTrainingTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void irisTrainingSanityCheck() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        for (String u : new String[]{"adam", "nesterov"}) {
            Nd4j.getRandom().setSeed(12345);
            log.info("Starting: " + u);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.placeHolder("input", DataType.FLOAT,  -1, 4);
            SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

            SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), DataType.FLOAT, 4, 10);
            SDVariable b0 = sd.zero("b0", DataType.FLOAT, 1, 10);

            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), DataType.FLOAT, 10, 3);
            SDVariable b1 = sd.zero("b1", DataType.FLOAT, 1, 3);

            SDVariable z0 = in.mmul(w0).add(b0);
            SDVariable a0 = sd.math().tanh(z0);
            SDVariable z1 = a0.mmul(w1).add("prediction", b1);
            SDVariable a1 = sd.nn().softmax(z1);

            SDVariable diff = sd.f().squaredDifference(a1, label);
            SDVariable lossMse = diff.mul(diff).mean();

            IUpdater updater;
            switch (u) {
                case "sgd":
                    updater = new Sgd(3e-1);
                    break;
                case "adam":
                    updater = new Adam(1e-2);
                    break;
                case "nesterov":
                    updater = new Nesterovs(1e-1);
                    break;
                default:
                    throw new RuntimeException();
            }

            TrainingConfig conf = new TrainingConfig.Builder()
                    .l2(1e-4)
                    .updater(updater)
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .build();

            sd.setTrainingConfig(conf);

            sd.setListeners(new ScoreListener(1));

            sd.fit(iter, 50);

            Evaluation e = new Evaluation();
            Map<String, List<IEvaluation>> evalMap = new HashMap<>();
            evalMap.put("prediction", Collections.singletonList(e));

            sd.evaluateMultiple(iter, evalMap);

            System.out.println(e.stats());

            double acc = e.accuracy();
            assertTrue(u + " - " + acc, acc >= 0.75);
        }
    }


    @Test
    public void irisTrainingEvalTest() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), DataType.FLOAT, 4, 10);
        SDVariable b0 = sd.zero("b0", DataType.FLOAT, 1, 10);

        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), DataType.FLOAT, 10, 3);
        SDVariable b1 = sd.zero("b1", DataType.FLOAT, 1, 3);

        SDVariable z0 = in.mmul(w0).add(b0);
        SDVariable a0 = sd.math().tanh(z0);
        SDVariable z1 = a0.mmul(w1).add("prediction", b1);
        SDVariable a1 = sd.nn().softmax(z1);

        SDVariable diff = sd.f().squaredDifference(a1, label);
        SDVariable lossMse = diff.mul(diff).mean();

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .trainEvaluation("prediction", 0, new Evaluation())
                .build();

        sd.setTrainingConfig(conf);

        History hist = sd.fit().train(iter, 50).exec();

        Evaluation e = hist.finalTrainingEvaluations().evaluation("prediction");

        System.out.println(e.stats());

        double acc = e.accuracy();

        assertTrue("Accuracy bad: " + acc, acc >= 0.75);
    }


    @Test
    public void irisTrainingValidationTest() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        DataSetIterator valIter = new IrisDataSetIterator(30, 60);
        NormalizerStandardize valStd = new NormalizerStandardize();
        valStd.fit(valIter);
        valIter.setPreProcessor(std);

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), DataType.FLOAT, 4, 10);
        SDVariable b0 = sd.zero("b0", DataType.FLOAT, 1, 10);

        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), DataType.FLOAT, 10, 3);
        SDVariable b1 = sd.zero("b1", DataType.FLOAT, 1, 3);

        SDVariable z0 = in.mmul(w0).add(b0);
        SDVariable a0 = sd.math().tanh(z0);
        SDVariable z1 = a0.mmul(w1).add("prediction", b1);
        SDVariable a1 = sd.nn().softmax(z1);

        SDVariable diff = sd.f().squaredDifference(a1, label);
        SDVariable lossMse = diff.mul(diff).mean();

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .validationEvaluation("prediction", 0, new Evaluation())
                .build();

        sd.setTrainingConfig(conf);

        History hist = sd.fit().train(iter, 50).validate(valIter, 5).exec();

        Evaluation e = hist.finalValidationEvaluations().evaluation("prediction");

        System.out.println(e.stats());

        double acc = e.accuracy();

        assertTrue("Accuracy bad: " + acc, acc >= 0.75);
    }


    @Test
    public void testTrainingMixedDtypes(){

        for (String u : new String[]{"adam", "nesterov", "adamax", "amsgrad"}) {

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);

            SDVariable inHalf = in.castTo(DataType.HALF);
            SDVariable inDouble = in.castTo(DataType.DOUBLE);

            SDVariable wFloat = sd.var("wFloat", Nd4j.rand(DataType.FLOAT, 4, 3));
            SDVariable wDouble = sd.var("wDouble", Nd4j.rand(DataType.DOUBLE, 4, 3));
            SDVariable wHalf = sd.var("wHalf", Nd4j.rand(DataType.HALF, 4, 3));

            SDVariable outFloat = in.mmul(wFloat);
            SDVariable outDouble = inDouble.mmul(wDouble);
            SDVariable outHalf = inHalf.mmul(wHalf);

            SDVariable sum = outFloat.add(outDouble.castTo(DataType.FLOAT)).add(outHalf.castTo(DataType.FLOAT));

            SDVariable loss = sum.std(true);

            IUpdater updater;
            switch (u) {
                case "sgd":
                    updater = new Sgd(1e-2);
                    break;
                case "adam":
                    updater = new Adam(1e-2);
                    break;
                case "nesterov":
                    updater = new Nesterovs(1e-2);
                    break;
                case "adamax":
                    updater = new AdaMax(1e-2);
                    break;
                case "amsgrad":
                    updater = new AMSGrad(1e-2);
                    break;
                default:
                    throw new RuntimeException();
            }

            TrainingConfig conf = new TrainingConfig.Builder()
                    .l2(1e-4)
                    .updater(updater)
                    .dataSetFeatureMapping("in")
                    .markLabelsUnused()
                    .build();

            sd.setTrainingConfig(conf);

            DataSet ds = new DataSet(Nd4j.rand(DataType.FLOAT, 3, 4), null);

            for( int i=0; i<10; i++ ){
                sd.fit(ds);
            }
        }

    }

    @Test
    public void simpleClassification() {
        double learning_rate = 0.001;
        int seed = 7;
        org.nd4j.linalg.api.rng.Random rng = Nd4j.getRandom();
        rng.setSeed(seed);
        INDArray x1_label1 = Nd4j.randn(3.0, 1.0, new long[]{1000}, rng);
        INDArray x2_label1 = Nd4j.randn(2.0, 1.0, new long[]{1000}, rng);
        INDArray x1_label2 = Nd4j.randn(7.0, 1.0, new long[]{1000}, rng);
        INDArray x2_label2 = Nd4j.randn(6.0, 1.0, new long[]{1000}, rng);

        INDArray x1s = Nd4j.concat(0, x1_label1, x1_label2);
        INDArray x2s = Nd4j.concat(0, x2_label1, x2_label2);

        SameDiff sd = SameDiff.create();
        INDArray ys = Nd4j.scalar(0.0).mul(x1_label1.length()).add(Nd4j.scalar(1.0).mul(x1_label2.length()));

        SDVariable X1 = sd.placeHolder("x1", DataType.DOUBLE, 2000);
        SDVariable X2 = sd.placeHolder("x2", DataType.DOUBLE, 2000);
        SDVariable y = sd.placeHolder("y", DataType.DOUBLE);
        SDVariable w = sd.var("w", DataType.DOUBLE, 3);

        // TF code:
        //cost = tf.reduce_mean(-tf.log(y_model * Y + (1 — y_model) * (1 — Y)))
        SDVariable y_model =
                sd.nn.sigmoid(w.get(SDIndex.point(2)).mul(X2).add(w.get(SDIndex.point(1)).mul(X1)).add(w.get(SDIndex.point(0))));
        SDVariable cost_fun =
                (sd.math.neg(sd.math.log(y_model.mul(y).add((sd.math.log(sd.constant(1.0).minus(y_model)).mul(sd.constant(1.0).minus(y)))))));
        SDVariable loss = sd.mean("loss", cost_fun);

        val updater = new Sgd(learning_rate);

        sd.setLossVariables("loss");
        sd.createGradFunction();
        val conf = new TrainingConfig.Builder()
                .updater(updater)
                .minimize("loss")
                .dataSetFeatureMapping("x1", "x2", "y")
                .markLabelsUnused()
                .build();

        MultiDataSet mds = new MultiDataSet(new INDArray[]{x1s, x2s, ys},null);

        sd.setTrainingConfig(conf);
        History history = sd.fit(new SingletonMultiDataSetIterator(mds), 1);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
