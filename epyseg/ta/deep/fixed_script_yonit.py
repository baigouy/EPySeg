from epyseg.deeplearning.deepl import EZDeepLearning
import os

def run_segmentation(deepTA=None, IS_TA_OUTPUT_MODE=True, input_channel_of_interest=None, TILE_WIDTH=256,
                     TILE_HEIGHT=256, TILE_OVERLAP=32, INPUT_FOLDER=None, EPYSEG_PRETRAINING='Linknet-vgg16-sigmoid-v2',
                     SIZE_FILTER=None, progress_callback=None):
    """
    Run the segmentation process.

    Args:
        deepTA (EZDeepLearning): An instance of EZDeepLearning class. If None, a new instance will be created.
        IS_TA_OUTPUT_MODE (bool): Indicates if the output should be in TA mode.
        input_channel_of_interest (int): The channel of interest in case of multichannel input image.
        TILE_WIDTH (int): Width of the input tiles for prediction.
        TILE_HEIGHT (int): Height of the input tiles for prediction.
        TILE_OVERLAP (int): Overlap between tiles for prediction.
        INPUT_FOLDER (str): Path to the folder containing the files to segment.
        EPYSEG_PRETRAINING (str): Pretraining model to use for segmentation.
        SIZE_FILTER (int): Size filter to remove cells below a certain pixel area.
        progress_callback (callable): Callback function for reporting progress.

    """
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
    predict_parameters["hq_pred_options"] = "Use all augs (pixel preserving + deteriorating) (Recommended for segmentation)"
    predict_parameters["post_process_algorithm"] = "default (slow/robust) (epyseg pre-trained model only!)"
    predict_parameters["input_normalization"] = input_normalization
    predict_parameters["filter"] = SIZE_FILTER

    predict_generator = deepTA.get_predict_generator(
        inputs=[INPUT_FOLDER], input_shape=input_shape, output_shape=output_shape, clip_by_frequency=None,
        **predict_parameters)

    if not IS_TA_OUTPUT_MODE:
        predict_output_folder = os.path.join(INPUT_FOLDER, 'predict')
    else:
        predict_output_folder = 'TA_mode'

    deepTA.predict(predict_generator, output_shape, predict_output_folder=predict_output_folder, batch_size=1,
                   progress_callback=progress_callback, **predict_parameters)

if __name__ == '__main__':
    INPUT_FOLDER = '/path/to/files_to_segment/'
    IS_TA_OUTPUT_MODE = True
    input_channel_of_interest = None
    TILE_WIDTH = 256
    TILE_HEIGHT = 256
    TILE_OVERLAP = 32
    EPYSEG_PRETRAINING = 'Linknet-vgg16-sigmoid-v2'
    SIZE_FILTER = None

    run_segmentation(INPUT_FOLDER=INPUT_FOLDER, IS_TA_OUTPUT_MODE=IS_TA_OUTPUT_MODE,
                     input_channel_of_interest=input_channel_of_interest, TILE_WIDTH=TILE_WIDTH,
                     TILE_HEIGHT=TILE_HEIGHT, TILE_OVERLAP=TILE_OVERLAP, EPYSEG_PRETRAINING=EPYSEG_PRETRAINING,
                     SIZE_FILTER=SIZE_FILTER)
