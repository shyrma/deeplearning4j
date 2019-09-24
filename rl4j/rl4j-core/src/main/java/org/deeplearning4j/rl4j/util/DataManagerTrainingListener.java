package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;

/**
 * DataManagerSyncTrainingListener can be added to the listeners of SyncLearning so that the
 * training process can be fed to the DataManager
 */
@Slf4j
public class DataManagerTrainingListener implements TrainingListener {
    private final IDataManager dataManager;

    private int lastSave = -Constants.MODEL_SAVE_FREQ;

    public DataManagerTrainingListener(IDataManager dataManager) {
        this.dataManager = dataManager;
    }

    @Override
    public ListenerResponse onTrainingStart() {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {

    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
        IHistoryProcessor hp = trainer.getHistoryProcessor();
        if(hp != null) {
            int[] shape = trainer.getMdp().getObservationSpace().getShape();
            String filename = dataManager.getVideoDir() + "/video-";
            if (trainer instanceof AsyncThread) {
                filename += ((AsyncThread) trainer).getThreadNumber() + "-";
            }
            filename += trainer.getEpochCounter() + "-" + trainer.getStepCounter() + ".mp4";
            hp.startMonitor(filename, shape);
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, IDataManager.StatEntry statEntry) {
        IHistoryProcessor hp = trainer.getHistoryProcessor();
        if(hp != null) {
            hp.stopMonitor();
        }
        try {
            dataManager.appendStat(statEntry);
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onTrainingProgress(ILearning learning) {
        try {
            int stepCounter = learning.getStepCounter();
            if (stepCounter - lastSave >= Constants.MODEL_SAVE_FREQ) {
                dataManager.save(learning);
                lastSave = stepCounter;
            }

            dataManager.writeInfo(learning);
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }
}
