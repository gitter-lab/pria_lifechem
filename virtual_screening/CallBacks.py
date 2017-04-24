import keras
import sys
from evaluation import *


# define custom classes
# following class is used for keras to compute the AUC each epoch
# and do early stoppping based on that
class KeckCallBackOnROC(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_roc = 0
        self.best_roc = 0
        self.counter = 0
        self.patience = patience
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_roc = roc_auc_single(self.y_val, self.model.predict(self.X_val))
        self.best_roc = self.curr_roc
        self.model.save_weights(self.file_path)

    def on_epoch_end(self, epoch, logs={}):
        self.curr_roc = roc_auc_single(self.y_val, self.model.predict(self.X_val))
        if self.curr_roc < self.best_roc:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_roc = self.curr_roc
            self.model.save_weights(self.file_path)

        train_roc = roc_auc_single(self.y_train, self.model.predict(self.X_train))
        # train_bedroc = bedroc_auc_single(self.y_train, self.model.predict(self.X_train))
        train_pr = precision_auc_single(self.y_train, self.model.predict(self.X_train))
        # curr_bedroc = bedroc_auc_single(self.y_val, self.model.predict(self.X_val))
        curr_pr = precision_auc_single(self.y_val, self.model.predict(self.X_val))
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        # print 'Train\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (train_roc, train_bedroc, train_pr)
        # print 'Val\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (self.curr_roc, curr_bedroc, curr_pr)
        print 'Train\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (train_roc, train_pr)
        print 'Val\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (self.curr_roc, curr_pr)
        print

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_roc(self):
        return self.best_roc


# define custom classes
# following class is used for keras to compute the precision each epoch
# and do early stoppping based on that
class KeckCallBackOnPrecision(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_pr = 0
        self.best_pr = 0
        self.counter = 0
        self.patience = patience
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_pr = precision_auc_single(self.y_val, self.model.predict(self.X_val))
        self.best_pr = self.curr_pr
        self.model.save_weights(self.file_path)

    def on_epoch_end(self, epoch, logs={}):
        self.curr_pr = precision_auc_single(self.y_val, self.model.predict(self.X_val))
        if self.curr_pr < self.best_pr:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_pr = self.curr_pr
            self.model.save_weights(self.file_path)

        train_roc = roc_auc_single(self.y_train, self.model.predict(self.X_train))
        # train_bedroc = bedroc_auc_single(self.y_train, self.model.predict(self.X_train))
        train_pr = precision_auc_single(self.y_train, self.model.predict(self.X_train))
        curr_roc = roc_auc_single(self.y_val, self.model.predict(self.X_val))
        # curr_bedroc = bedroc_auc_single(self.y_val, self.model.predict(self.X_val))
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        # print 'Train\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (train_roc, train_bedroc, train_pr)
        # print 'Val\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (curr_roc, curr_bedroc, self.curr_pr)
        print 'Train\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (train_roc, train_pr)
        print 'Val\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (curr_roc, self.curr_pr)
        print

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_roc(self):
        return self.best_pr



class MultiCallBackOnROC(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 eval_indices, eval_mean_or_median, patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_roc = 0
        self.best_roc = 0
        self.counter = 0
        self.patience = patience
        self.best_model = None
        self.eval_indices = eval_indices
        self.eval_mean_or_median = eval_mean_or_median
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_roc = self.get_model_roc_auc(self.y_val, self.model.predict(self.X_val))
        self.best_roc = self.curr_roc
        self.model.save_weights(self.file_path)

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.X_val)
        y_pred_train = self.model.predict(self.X_train)
        self.curr_roc = self.get_model_roc_auc(self.y_val, y_pred_val)
        if self.curr_roc < self.best_roc:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_roc = self.curr_roc
            self.model.save_weights(self.file_path)

        curr_pr = self.get_model_precision_auc(self.y_val, y_pred_val)
        # curr_bedroc = self.get_model_bedroc_auc(self.y_val, y_pred_val)
        train_roc = self.get_model_roc_auc(self.y_train, y_pred_train)
        # train_bedroc = self.get_model_bedroc_auc(self.y_train, y_pred_train)
        train_pr = self.get_model_precision_auc(self.y_train, y_pred_train)
        print('Epoch {}/{}'.format(epoch + 1, self.nb_epoch))
        # print 'Train\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (train_roc, train_bedroc, train_pr)
        # print 'Val\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (self.curr_roc, curr_bedroc, curr_pr)
        print 'Train\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (train_roc, train_pr)
        print 'Val\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (self.curr_roc, curr_pr)

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_roc(self):
        return self.best_roc

    def get_model_roc_auc(self, true_label, predicted_label):
        return roc_auc_multi(true_label, predicted_label, self.eval_indices, self.eval_mean_or_median)

    def get_model_bedroc_auc(self, true_label, predicted_label):
        return bedroc_auc_multi(true_label, predicted_label, self.eval_indices, self.eval_mean_or_median)

    def get_model_precision_auc(self, true_label, predicted_label):
        return precision_auc_multi(true_label, predicted_label, self.eval_indices, self.eval_mean_or_median)


class MultiCallBackOnPR(keras.callbacks.Callback):
    # define custom classes
    # following class is used for keras to compute the precision each epoch
    # and do early stoppping based on that
    def __init__(self, X_train, y_train, X_val, y_val,
                 eval_indices, eval_mean_or_median, patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_pr = 0
        self.best_pr = 0
        self.counter = 0
        self.patience = patience
        self.best_model = None
        self.eval_indices = eval_indices
        self.eval_mean_or_median = eval_mean_or_median
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_pr = self.get_model_precision_auc(self.y_val, self.model.predict(self.X_val))
        self.best_pr = self.curr_pr
        self.model.save_weights(self.file_path)

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.X_val)
        y_pred_train = self.model.predict(self.X_train)
        self.curr_pr = self.get_model_precision_auc(self.y_val, y_pred_val)
        if self.curr_pr < self.best_pr:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_pr = self.curr_pr
            self.model.save_weights(self.file_path)

        curr_roc = self.get_model_roc_auc(self.y_val, y_pred_val)
        # curr_bedroc = self.get_model_bedroc_auc(self.y_val, y_pred_val)
        train_roc = self.get_model_roc_auc(self.y_train, y_pred_train)
        # train_bedroc = self.get_model_bedroc_auc(self.y_train, y_pred_train)
        train_pr = self.get_model_precision_auc(self.y_train, y_pred_train)

        print('Epoch {}/{}'.format(epoch + 1, self.nb_epoch))
        # print 'Train\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (train_roc, train_bedroc, train_pr)
        # print 'Val\tAUC[ROC]: %.6f\tAUC[BEDROC]: %.6f\tAUC[PR]: %.6f' % \
        #       (curr_roc, curr_bedroc, self.curr_pr)
        print 'Train\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (train_roc, train_pr)
        print 'Val\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
              (curr_roc, self.curr_pr)

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_pr(self):
        return self.best_pr

    def get_model_roc_auc(self, true_label, predicted_label):
        return roc_auc_multi(true_label, predicted_label, self.eval_indices, self.eval_mean_or_median)

    def get_model_bedroc_auc(self, true_label, predicted_label):
        return bedroc_auc_multi(true_label, predicted_label, self.eval_indices, self.eval_mean_or_median)

    def get_model_precision_auc(self, true_label, predicted_label):
        return precision_auc_multi(true_label, predicted_label, self.eval_indices, self.eval_mean_or_median)