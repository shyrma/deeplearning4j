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

package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import lombok.Getter;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public class AsyncNStepQLearningThreadDiscrete<O extends Encodable> extends AsyncThreadDiscrete<O, IDQN> {

    @Getter
    final protected AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration conf;
    @Getter
    final protected IAsyncGlobal<IDQN> asyncGlobal;
    @Getter
    final protected int threadNumber;

    final private Random random;

    public AsyncNStepQLearningThreadDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IAsyncGlobal<IDQN> asyncGlobal,
                                             AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration conf,
                                             TrainingListenerList listeners, int threadNumber, int deviceNum) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        this.conf = conf;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;
        mdp.getActionSpace().setSeed(conf.getSeed() + threadNumber);
        random = new Random(conf.getSeed() + threadNumber);
    }

    public Policy<O, Integer> getPolicy(IDQN nn) {
        return new EpsGreedy(new DQNPolicy(nn), getMdp(), conf.getUpdateStart(), conf.getEpsilonNbStep(),
                        random, conf.getMinEpsilon(), this);
    }



    //calc the gradient based on the n-step rewards
    public Gradient[] calcGradient(IDQN current, Stack<MiniTrans<Integer>> rewards) {

        MiniTrans<Integer> minTrans = rewards.pop();

        int size = rewards.size();

        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape()
                        : getHistoryProcessor().getConf().getShape();
        int[] nshape = Learning.makeShape(size, shape);
        INDArray input = Nd4j.create(nshape);
        INDArray targets = Nd4j.create(size, getMdp().getActionSpace().getSize());

        double r = minTrans.getReward();
        for (int i = size - 1; i >= 0; i--) {
            minTrans = rewards.pop();

            r = minTrans.getReward() + conf.getGamma() * r;
            input.putRow(i, minTrans.getObs());
            INDArray row = minTrans.getOutput()[0];
            row = row.putScalar(minTrans.getAction(), r);
            targets.putRow(i, row);
        }

        return current.gradient(input, targets);
    }
}
