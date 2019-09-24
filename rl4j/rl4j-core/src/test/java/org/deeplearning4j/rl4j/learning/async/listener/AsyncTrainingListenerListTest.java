package org.deeplearning4j.rl4j.learning.async.listener;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.*;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AsyncTrainingListenerListTest {
    @Test
    public void when_listIsEmpty_expect_notifyTrainingStartedReturnTrue() {
        // Arrange
        TrainingListenerList sut = new TrainingListenerList();

        // Act
        boolean resultTrainingStarted = sut.notifyTrainingStarted();
        boolean resultNewEpoch = sut.notifyNewEpoch(null);
        boolean resultEpochTrainingResult = sut.notifyEpochTrainingResult(null, null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultNewEpoch);
        assertTrue(resultEpochTrainingResult);
    }

    @Test
    public void when_firstListerStops_expect_othersListnersNotCalled() {
        // Arrange
        MockTrainingListener listener1 = new MockTrainingListener();
        listener1.onTrainingResultResponse = TrainingListener.ListenerResponse.STOP;
        MockTrainingListener listener2 = new MockTrainingListener();
        TrainingListenerList sut = new TrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        sut.notifyEpochTrainingResult(null, null);

        // Assert
        assertEquals(1, listener1.onEpochTrainingResultCallCount);
        assertEquals(0, listener2.onEpochTrainingResultCallCount);
    }

    @Test
    public void when_allListenersContinue_expect_listReturnsTrue() {
        // Arrange
        MockTrainingListener listener1 = new MockTrainingListener();
        MockTrainingListener listener2 = new MockTrainingListener();
        TrainingListenerList sut = new TrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        boolean resultTrainingProgress = sut.notifyEpochTrainingResult(null, null);

        // Assert
        assertTrue(resultTrainingProgress);
    }

    private static class MockTrainingListener implements TrainingListener {

        public int onEpochTrainingResultCallCount = 0;
        public ListenerResponse onTrainingResultResponse = ListenerResponse.CONTINUE;
        public int onTrainingProgressCallCount = 0;
        public ListenerResponse onTrainingProgressResponse = ListenerResponse.CONTINUE;

        @Override
        public ListenerResponse onTrainingStart() {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public void onTrainingEnd() {

        }

        @Override
        public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, IDataManager.StatEntry statEntry) {
            ++onEpochTrainingResultCallCount;
            return onTrainingResultResponse;
        }

        @Override
        public ListenerResponse onTrainingProgress(ILearning learning) {
            ++onTrainingProgressCallCount;
            return onTrainingProgressResponse;
        }
    }

}
