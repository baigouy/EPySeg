import numpy as np
import random
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()


# do a real generator for this to avoid code dupes btw --> and add more flexibility --> less global variables
class MetaGenerator:

    def __init__(self, augmenters, shuffle, batch_size, gen_type):
        self.remains_of_previous_batch = None
        self.batch_size = batch_size
        self.augmenters = augmenters
        self.shuffle = shuffle
        self.gen_type = gen_type

    def generator(self, skip_augment, first_run):
        self.remains_of_previous_batch = None
        generators = []
        generators_length = []
        for gen in self.augmenters:
            if self.gen_type == 'train':
                if gen.has_train_set():
                    generators.append(gen.train_generator(skip_augment=skip_augment, first_run=first_run))
                    generators_length.append(gen.get_train_set_length())
            elif self.gen_type == 'test':
                if gen.has_test_set():
                    generators.append(gen.test_generator(skip_augment=skip_augment, first_run=first_run))
                    generators_length.append(gen.get_test_set_length())
            elif self.gen_type == 'valid':
                if gen.has_validation_set():
                    generators.append(gen.validation_generator(skip_augment=skip_augment, first_run=first_run))
                    generators_length.append(gen.get_validation_set_length())
            else:
                logger.error('unsupported generator ' + self.gen_type)

        if self.shuffle:
            # call random based on some data
            logger.debug('The datagenerator is using shuffle mode')
            random_generator_indices = []
            for iii, gener_ in enumerate(generators):
                random_generator_indices += [iii] * generators_length[iii]

            # print(len(random_generator_indices))
            # print(random_generator_indices)
            random_generator_indices = random.sample(random_generator_indices, len(random_generator_indices))
            # print("-->", len(random_generator_indices))
            # print(random_generator_indices)
            for idcs in random_generator_indices:
                choice = generators[idcs]
                for c in choice:  # TODO remove that line so that I always switch between generators
                    out = self.is_batch_ready(c, False)
                    while out is not None:
                        yield out
                        out = self.is_batch_ready(None, False)
                    break  # does that work ??

            # old random code --> not great because trains too much on one sample before moving to the next --> now changing
            # indices = random.sample(range(len(generators)), len(generators))
            # for idcs in indices:
            #     choice = generators[idcs]
            #     for c in choice: # TODO remove that line so that I always switch between generators
            #         out = self.is_batch_ready(c, False)
            #         while out is not None:
            #             yield out
            #             out = self.is_batch_ready(None, False)
        else:
            for gen in generators:
                for c in gen:
                    out = self.is_batch_ready(c, False)
                    while out is not None:
                        yield out
                        out = self.is_batch_ready(None, False)
        out = self.is_batch_ready(None, True)
        while out is not None:
            yield out
            out = self.is_batch_ready(None, True)

    def multiconcat(self, old_batch, new_batch):
        for idcs, input_output in enumerate(new_batch):
            for j, data in enumerate(input_output):
                try:
                    old_batch[idcs][j] = np.concatenate((old_batch[idcs][j], data), axis=0)
                except:
                    # if images don't have the same nb of Z add empty frames to the smallest one so that both fit --> bug fix for 3D generators with images having different depth
                    # TODO should I use padding in the Z axis with reflect here too --> in most cases that should work especially for

                    if len(old_batch[idcs][j].shape) == len(data.shape) == 5:
                        if old_batch[idcs][j].shape[1] != data.shape[1]:
                            # need add frames to the smallest..., try add black frames...
                            if old_batch[idcs][j].shape[1] < data.shape[1]:
                                smallest_z = old_batch[idcs][j]
                                biggest_z = data
                            else:
                                smallest_z = data
                                biggest_z = old_batch[idcs][j]

                            smallest_shape = list(smallest_z.shape)
                            z_dim_difference = biggest_z.shape[1] - smallest_z.shape[1]
                            smallest_shape[1] = z_dim_difference

                            missing_frames = np.zeros((smallest_shape), dtype=smallest_z.dtype)
                            # use min per channel --> it is a much better idea
                            # should test that changes are still ok but should be

                            # print("shp", missing_frames.shape)

                            for c in range(missing_frames.shape[-1]):
                                missing_frames[...,c].fill(smallest_z[...,c].min())

                            smallest_z = np.append(smallest_z, missing_frames, axis=1) # nb should do that per channel in fact... -->

                            old_batch[idcs][j] = np.concatenate((smallest_z, biggest_z), axis=0)

                            del smallest_z
                            del biggest_z

        return old_batch

    def multisplit(self, old_batch):
        out = [[], []]
        # print(len(old_batch))
        # print(type(old_batch)) # tuple
        # print(len(old_batch), len(old_batch[0][0]))

        for idcs, input_output in enumerate(old_batch):
            for j, data in enumerate(input_output):
                cur = data[:self.batch_size]
                # print(old_batch[idcs][j].shape, data[self.batch_size:].shape) # devrait etre le meme en fait ???
                old_batch[idcs][j] = data[self.batch_size:]
                out[idcs].append(cur)
        return old_batch, out

    def multi_add_images_to_batch(self, old_batch):
        missing_image_nb = self.batch_size - old_batch[0][0].shape[0]
        for idcs, input_output in enumerate(old_batch):
            for j, data in enumerate(input_output):
                missing_images = np.zeros((missing_image_nb, *data.shape[1:]), dtype=data.dtype)
                old_batch[idcs][j] = np.concatenate((data, missing_images), axis=0)
        return old_batch

    def is_batch_ready(self, current_batch, last_image):

        # print('cb', current_batch, last_image)
        # pb c'est que en 3D je ne dois pas faire pareil du tout car j'ai pas le droit de fusionner des stacks --> en fait si mais faut avoir le meme nombre de trucs pr les deux --> je dois ajouter ou enlever des images dans l'un ou l'autre

        if current_batch is not None and self.remains_of_previous_batch is not None:
            # this stuff should have the size and structure of cur batch
            self.remains_of_previous_batch = self.multiconcat(self.remains_of_previous_batch, current_batch)
        elif self.remains_of_previous_batch is None and current_batch is not None:
            self.remains_of_previous_batch = current_batch

        # NB duplicated code but ok could put it as a separate function though !!!!
        if self.remains_of_previous_batch is not None:
            # print('bs', self.batch_size, self.remains_of_previous_batch[0][0].shape[0])
            if self.remains_of_previous_batch[0][0].shape[0] == self.batch_size:
                out = self.remains_of_previous_batch
                self.remains_of_previous_batch = None
                return out
            elif self.remains_of_previous_batch[0][0].shape[0] > self.batch_size:
                self.remains_of_previous_batch, out = self.multisplit(self.remains_of_previous_batch)
                return out

        if last_image and not self.remains_of_previous_batch is None:
            out = self.multi_add_images_to_batch(self.remains_of_previous_batch)
            self.remains_of_previous_batch = None
            return out
        return None

    def close(self):
        # hack to avoid errors when tf stops the generator
        pass

    # this is to handle GeneratorExit that is called at the end
    def __exit__(self,  exc_type, exc_value, traceback):
        if exc_type:
            logger.error("Aborted %s", self,
                          exc_info=(exc_type, exc_value, traceback))

if __name__ == '__main__':
    test_input_img_1 = np.zeros((10, 1, 1, 1))
    test_output_img_1 = np.zeros((10, 1, 1, 1))
    combined_test = [[test_input_img_1], [test_output_img_1]]

    test_add_image = np.zeros((16, 1, 1, 1))

    test = MetaGenerator()

    if True:
        import sys

        out = test.is_batch_ready(combined_test, False)
        print(out[0][0].shape)
        add_to_bacth_test = [[test_add_image], [test_add_image]]
        out = test.is_batch_ready(add_to_bacth_test, False)
        print(out[0][0].shape)
        while out is not None:
            print(out[0][0].shape)
            out = test.is_batch_ready(None, False)

        print("done")

        sys.exit(0)

    # gives the desired output

    result = test.is_batch_ready(combined_test, False)
    print('out', result[0][0].shape)
    print(test.remains_of_previous_batch[0][0].shape)
    while test.remains_of_previous_batch is not None:
        if test.remains_of_previous_batch[0][0].shape[0] < test.batch_size:
            break
        result = test.is_batch_ready(None, True)
        if result is not None:
            print('out1b', result[0][0].shape)
            if test.remains_of_previous_batch is not None:
                print(test.remains_of_previous_batch[0][0].shape)
            else:
                print(None)

    add_to_bacth_test = [[test_add_image], [test_add_image]]
    result = test.is_batch_ready(add_to_bacth_test, False)
    print('out1', result[0][0].shape)
    print(test.remains_of_previous_batch[0][0].shape)

    while result is not None:
        result = test.is_batch_ready(None, True)
        if test.remains_of_previous_batch is not None:
            print(test.remains_of_previous_batch[0][0].shape)
        if result is not None:
            print('out2', result[0][0].shape)

        if test.remains_of_previous_batch is not None:
            print(test.remains_of_previous_batch[0][0].shape)
        if result is None:
            break
