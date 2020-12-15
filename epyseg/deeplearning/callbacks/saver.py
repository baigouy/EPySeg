# TODO do plots of the logs
from epyseg.img import Img
from epyseg.tools.logger import TA_logger
from timeit import default_timer as timer

logger = TA_logger()

import traceback
import tensorflow as tf
import os


class My_saver_callback(tf.keras.callbacks.Callback):
    '''a custom saver callback

    '''

    def __init__(self, save_output_name, deepTA, epochs=None, keep_n_best=5, output_folder_for_models=None,
                 monitor='loss', verbose=0, save_best_only=False, save_training_log=True, keep_best_validation=True,
                 save_weights_only=False, progress_callback=None):
        '''init saver callback

        Parameters
        ----------
        save_output_name : string
            generic name for model saving

        deepTA : deep learning tool


        epochs : int
            total nb of epochs (required to display progress)

        keep_n_best : int
            number of best models (lower loss) to keep during training

        output_folder_for_models : string
            folder where models should be saved

        monitor : str
            loss to monitor (could be loss or val_loss if validation set exists)

        verbose : int
            verbose level

        save_best_only : boolean
            if True only saves the best model (overrides keep_n_best)

        save_training_log :  boolean
            if True saves training log (containing losses and metrics outputs along with the epoch nb, useful for plots)

        keep_best_validation :  boolean
            keeps best model (lowest loss) on validation

        save_weights_only : boolean
            only save weights not model and optimizer

        progress_callback : anything to display progress
            dipslays progress

        '''

        self.save_weights_only = save_weights_only
        self.output_folder_for_models = output_folder_for_models
        self.save_output_name = save_output_name
        if self.output_folder_for_models is not None:
            if not os.path.exists(self.output_folder_for_models):
                os.makedirs(self.output_folder_for_models, exist_ok=True)
            self.save_output_name = os.path.join(self.output_folder_for_models, self.save_output_name)
        # print('joined path', self.save_output_name)
        self.deepTA = deepTA
        self.epochs = epochs
        self.keep_n_best = keep_n_best
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.save_training_log = save_training_log
        self.keep_best_validation = keep_best_validation
        self.best_val_loss = 10000
        self.progress_callback = progress_callback
        if self.save_best_only:
            self.keep_n_best = 1  # only keep the best
        self.stop_me = False
        self.best_model = None

    def on_train_begin(self, logs={}):
        self.best_loss = 1_000_000  # keeps the best/lowest loss value
        self.losses = []
        self.accs = []
        self.best_kept = []

        # self.rank_n_score={}
        # self.rank_n_filename = {}
        self.loss_n_filename = {}

        if self.save_training_log:
            # save raw log for plot purpose
            open(self.save_output_name + '.log', 'w+').close()

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = timer()
        try:
            if self.progress_callback is not None and self.epochs is not None:
                self.progress_callback.emit((epoch / self.epochs) * 100)
            else:
                logger.info(str((epoch / self.epochs) * 100) + '%')
        except:
            pass
        return

    # in fact need store rename first then run stuff
    def get_save_name(self, epoch, value):
        return self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, value)

    def on_epoch_end(self, epoch, logs={}):
        try:
            end_time = timer()
            epoch_time = end_time - self.start_time
            total_time_in_hours = (epoch_time * (self.epochs - epoch)) / 3600
            if total_time_in_hours >= 1.:
                logger.info('Estimated remaining run time: ' + str(total_time_in_hours) + ' hour(s)')
            else:
                total_time_in_hours *= 60
                if total_time_in_hours >= 1.:
                    logger.info('Estimated remaining run time: ' + str(total_time_in_hours) + ' minute(s)')
                else:
                    total_time_in_hours *= 60
                    logger.info('Estimated remaining run time: ' + str(total_time_in_hours) + ' second(s)')
        except:
            pass

        self.losses.append(logs.get(self.monitor))
        logger.debug('logs ' + str(logs))

        # TODO remove that ???
        if self.keep_best_validation and not self.stop_me:
            val_loss = logs.get('val_loss')
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # always overwrites it...
                logger.debug('Saving best val loss model')
                # TODO maybe change this and replace by accuracy ????
                if not self.save_weights_only:
                    logger.info('saving file: ' + self.save_output_name + '_best_val_loss.h5')
                    self.model.save(self.save_output_name + '_best_val_loss.h5')
                else:
                    logger.info('saving file: ' + self.save_output_name + '_best_val_loss.h5')
                    self.model.save_weights(self.save_output_name + '_best_val_loss.h5')

        save_file_name = self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))

        if self.keep_n_best is not None and not self.stop_me:
            # now seems ok but need do the same for all other steps
            # this is the part of the code I need to change to overwrite
            if self.best_kept:
                if len(self.best_kept) < self.keep_n_best:
                    self.best_kept.append(logs.get(self.monitor))
                    if logs.get(self.monitor) is not None:

                        self.best_kept.sort(reverse=True)
                        # print('sorted stuff', self.best_kept)
                        # print('idx=', self.best_kept.index(logs.get(self.monitor)))
                        # seems ok now need rename stuff

                        # self.best_kept.sort(reverse=True)
                        # if logs.get(self.monitor) is not None:
                        #     if logs.get(self.monitor) < self.best_kept[0]:
                        #         loss = self.best_kept[0]
                        #         self.best_kept[0] = logs.get(self.monitor)

                        # if not self.save_weights_only:
                        current_rank = self.best_kept.index(logs.get(self.monitor))
                        # print('need rename all images with rank <= ',current_rank)

                        real_rank = len(self.best_kept) - current_rank - 1
                        # print('new image real rank', real_rank)

                        save_file_name = self.save_output_name + '-' + str(
                            real_rank) + '.h5'  # + str('-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))
                        # print('filename ', save_file_name)

                        # en fait need rename tt les trucs de valeur superieure

                        # print('current files',self.loss_n_filename, logs.get(self.monitor))
                        # print('max rank', self.keep_n_best)
                        # counter = 0
                        updates = {}
                        for val in self.best_kept:
                            # if val>=logs.get(self.monitor):

                            if val == logs.get(self.monitor):
                                break

                            # need rename ? to rank + 1
                            # if rank is superior to max then delete file

                            if not val in self.loss_n_filename:
                                continue
                            # print('need rename ', self.loss_n_filename[val], ' to current rank + 1') # should be easy

                            new_index = self.best_kept.index(val)
                            real_new_index = len(self.best_kept) - new_index - 1
                            # print('expected_new_index_for_the image', real_new_index)
                            save_file_name2 = self.save_output_name + '-' + str(real_new_index) + '.h5'
                            # print(os.path.exists(save_file_name2), self.loss_n_filename[val])

                            try:
                                # bug fix to prevent filling google drive bin
                                open(save_file_name2,
                                     'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                            except:
                                pass
                            # os.rename seems to cause a crash if file already exists on windows os see https://stackoverflow.com/questions/45636341/is-it-wise-to-remove-files-using-os-rename-instead-of-os-remove-in-python
                            os.replace(self.loss_n_filename[val], save_file_name2)

                            # print('new_name_for_the_file', save_file_name2, 'before', self.loss_n_filename[val])
                            # self.loss_n_filename[val] = save_file_name2
                            updates[val] = save_file_name2

                            # if counter == 0:
                            #     print(remove)
                            # counter+=1

                        # else:
                        #     print(val,'<',logs.get(self.monitor))

                        # print("updates",updates)
                        # print('orig', self.loss_n_filename)
                        self.loss_n_filename.update(updates)
                        # print('orig corrected', self.loss_n_filename)
                        # then should save current as rank

                        # save rank and name
                        self.loss_n_filename[logs.get(
                            self.monitor)] = save_file_name  # self.save_output_name + str('-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                        # print('loss n filename',self.loss_n_filename)

                        # IN FACT SIMPLE ALWAYS NEED POP FIRST IN THE LIST THE WAY IT'S INVERTED

                        # need also store top 5 files

                        # up until here should be ok and no action is required it's only after that action is require

                        # logger.info('saving file: ' + self.save_output_name + str(                                '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))))
                        logger.info('saving file: ' + save_file_name)
                        # self.model.save(                                self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                        if not self.save_weights_only:
                            self.model.save(save_file_name)
                        else:
                            self.model.save_weights(save_file_name)
                        # every image of rank equal or > need be renamed
                        # save as rank_ --> so that one can easily cut name and replace it
                        # else:
                        #     logger.info('saving file: ' + self.save_output_name + str(
                        #         '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))))
                        #     self.model.save_weights(
                        #         self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                else:
                    self.best_kept.sort(reverse=True)
                    if logs.get(self.monitor) is not None:
                        if logs.get(self.monitor) < self.best_kept[0]:
                            # remove image that already exists
                            self.loss_n_filename.pop(self.best_kept[0], None)
                            # loss = self.best_kept[0]
                            self.best_kept[0] = logs.get(self.monitor)

                            self.best_kept.sort(reverse=True)
                            current_rank = self.best_kept.index(logs.get(self.monitor))
                            real_rank = len(self.best_kept) - current_rank - 1
                            save_file_name = self.save_output_name + '-' + str(
                                real_rank) + '.h5'  # + str('-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))

                            updates = {}
                            for val in self.best_kept:
                                if val == logs.get(self.monitor):
                                    break

                                if not val in self.loss_n_filename:
                                    continue

                                new_index = self.best_kept.index(val)
                                real_new_index = len(self.best_kept) - new_index - 1
                                save_file_name2 = self.save_output_name + '-' + str(real_new_index) + '.h5'
                                try:
                                    # bug fix to prevent filling google drive bin
                                    open(save_file_name2,
                                         'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                                except:
                                    pass
                                # os.rename seems to cause a crash if file already exists on windows os see https://stackoverflow.com/questions/45636341/is-it-wise-to-remove-files-using-os-rename-instead-of-os-remove-in-python
                                os.replace(self.loss_n_filename[val], save_file_name2)

                                # print('renaming 2', self.loss_n_filename[val], 'to', save_file_name2)

                                updates[val] = save_file_name2

                            # print("updates", updates)
                            # print('orig', self.loss_n_filename)
                            self.loss_n_filename.update(updates)
                            # print('orig corrected', self.loss_n_filename)

                            self.loss_n_filename[logs.get(
                                self.monitor)] = save_file_name  # self.save_output_name + str('-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                            # print('loss n filename', self.loss_n_filename)

                            logger.info('saving file: ' + save_file_name)
                            if not self.save_weights_only:
                                self.model.save(save_file_name)
                            else:
                                self.model.save_weights(save_file_name)

                            # logger.info('saving file: ' + self.save_output_name + str(
                            #     '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))))
                            # if not self.save_weights_only:
                            #
                            #     self.model.save(
                            #         self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                            # else:
                            #     self.model.save_weights(
                            #         self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                            # no need to remove any longer...
                            # logger.debug('deleting file: ' + str(self.save_output_name + '-ep*-l%0.6f.h5' % (loss)))
                            # fileList = glob.glob(self.save_output_name + '-ep*-l%0.6f.h5' % (loss))
                            # for f in fileList:
                            #     os.remove(f)
            else:
                self.best_kept.append(logs.get(self.monitor))
                if logs.get(self.monitor) is not None and not self.stop_me:
                    # self.loss_n_filename[logs.get(self.monitor)] = self.save_output_name + str(                        '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))) # tada
                    save_file_name = self.save_output_name + '-' + str(0) + '.h5'
                    self.loss_n_filename[logs.get(self.monitor)] = save_file_name
                    logger.info('saving file: ' + save_file_name)
                    if not self.save_weights_only:
                        # logger.info('saving file: ' + self.save_output_name + str(                            '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))))
                        # self.model.save(                            self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                        self.model.save(save_file_name)
                    else:
                        # logger.info('saving file: ' + self.save_output_name + str(                            '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor))))
                        # self.model.save_weights(                            self.save_output_name + '-ep%04d-l%0.6f.h5' % (epoch + 1, logs.get(self.monitor)))
                        self.model.save_weights(save_file_name)

            f = open(self.save_output_name + '_losses_n_corresponding_files' + '.log', "w")
            f.write(str(self.loss_n_filename))
            f.close()



        else:
            # save all files with their loss
            if logs.get(self.monitor) is not None and not self.stop_me:
                logger.info('saving file: ' + save_file_name)
                if not self.save_weights_only:
                    self.model.save(save_file_name)
                else:
                    self.model.save_weights(save_file_name)

        # best loss is always #1 in this new ranking system --> modify this too
        if logs.get(self.monitor) is not None:
            curloss = logs.get(self.monitor)
            if curloss <= self.best_loss and not self.stop_me:
                # store path to the best model
                self.best_loss = curloss
                self.best_model = save_file_name
                logger.debug('best model at epoch ' + str(epoch) + ' ' + self.best_model)

            if self.save_training_log:
                f2 = open(self.save_output_name + '.log', 'a+')
                f2.write('epoch %03d - loss %s\n' % (epoch + 1, str(logs)))
                f2.close()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def get_best_model(self):
        # return the past to the best model (lowest loss ever)
        return self.best_model
