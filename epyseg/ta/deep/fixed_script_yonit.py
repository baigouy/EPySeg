from epyseg.deeplearning.deepl import EZDeepLearning
import os


# TODO --> rename this class

def run_seg(deepTA=None, IS_TA_OUTPUT_MODE = True,    input_channel_of_interest = None, TILE_WIDTH = 256, TILE_HEIGHT = 256, TILE_OVERLAP = 32, INPUT_FOLDER = None, EPYSEG_PRETRAINING = 'Linknet-vgg16-sigmoid-v2', SIZE_FILTER = None, progress_callback=None):
    # predict parameters
    # IS_TA_OUTPUT_MODE = True  # stores as handCorrection.tif in the folder with the same name as the parent file without ext
    # input_channel_of_interest = None  # assumes image is single channel or multichannel nut channel of interest is ch0, needs be changed otherwise, e.g. 1 for channel 1
    # TILE_WIDTH = 256  # 128 # 64
    # TILE_HEIGHT = 256  # 128 # 64
    # TILE_OVERLAP = 32
    # INPUT_FOLDER = '/path/to/files_to_segment/'
    # EPYSEG_PRETRAINING = 'Linknet-vgg16-sigmoid-v2'  # or 'Linknet-vgg16-sigmoid' for v1
    # SIZE_FILTER = None  # 100 # set to 100 to get rid of cells having pixel area < 100 pixels

    # raw code for predict
    if deepTA is None:
        deepTA = EZDeepLearning()
    deepTA.load_or_build(architecture='Linknet', backbone='vgg16', activation='sigmoid', classes=7,
                         pretraining=EPYSEG_PRETRAINING)

    deepTA.get_loaded_model_params()
    deepTA.summary()

    input_shape = deepTA.get_inputs_shape()
    output_shape = deepTA.get_outputs_shape()

    input_normalization = {'method': 'Rescaling (min-max normalization)', 'range': [0, 1], 'individual_channels': True}

    predict_parameters = {}

    predict_parameters["input_channel_of_interest"] = input_channel_of_interest
    predict_parameters["default_input_tile_width"] = TILE_WIDTH
    predict_parameters["default_input_tile_height"] = TILE_HEIGHT
    predict_parameters["default_output_tile_width"] = TILE_WIDTH
    predict_parameters["default_output_tile_height"] = TILE_HEIGHT
    predict_parameters["tile_width_overlap"] = TILE_OVERLAP
    predict_parameters["tile_height_overlap"] = TILE_OVERLAP
    predict_parameters["hq_pred_options"] = "Use all augs (pixel preserving + deteriorating) (Recommended for segmentation)"  # not yet supported on current versions of epyseg
    predict_parameters["post_process_algorithm"] = "default (slow/robust) (epyseg pre-trained model only!)"  # None
    predict_parameters["input_normalization"] = input_normalization
    predict_parameters["filter"] = SIZE_FILTER

    predict_generator = deepTA.get_predict_generator(
        inputs=[INPUT_FOLDER], input_shape=input_shape,
        output_shape=output_shape,
        clip_by_frequency=None, **predict_parameters)  # before was 0.05 which is bad especially with restored images

    if not IS_TA_OUTPUT_MODE:
        predict_output_folder = os.path.join(INPUT_FOLDER, 'predict')
    else:
        predict_output_folder = 'TA_mode'

    deepTA.predict(predict_generator, output_shape, predict_output_folder=predict_output_folder, batch_size=1,progress_callback=progress_callback,
                   **predict_parameters)

    # del deepTA

if __name__ == '__main__':
    INPUT_FOLDER = '/path/to/files_to_segment/'
    IS_TA_OUTPUT_MODE = True  # stores as handCorrection.tif in the folder with the same name as the parent file without ext
    input_channel_of_interest = None  # assumes image is single channel or multichannel nut channel of interest is ch0, needs be changed otherwise, e.g. 1 for channel 1
    TILE_WIDTH = 256  # 128 # 64
    TILE_HEIGHT = 256  # 128 # 64
    TILE_OVERLAP = 32
    EPYSEG_PRETRAINING = 'Linknet-vgg16-sigmoid-v2'  # or 'Linknet-vgg16-sigmoid' for v1
    SIZE_FILTER = None  # 100 # set to 100 to get rid of cells having pixel area < 100 pixels

    run_seg(INPUT_FOLDER = INPUT_FOLDER, IS_TA_OUTPUT_MODE=IS_TA_OUTPUT_MODE, input_channel_of_interest=input_channel_of_interest, TILE_WIDTH=TILE_WIDTH, TILE_HEIGHT=TILE_HEIGHT, TILE_OVERLAP=TILE_OVERLAP,EPYSEG_PRETRAINING=EPYSEG_PRETRAINING, SIZE_FILTER=SIZE_FILTER )