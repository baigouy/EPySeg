import random
import matplotlib.pyplot as plt
from epyseg.img import apply_2D_gradient, create_random_intensity_graded_perturbation, Img

def GuidedDataRecoverer(input_data, indices_to_recover, auto_unpack_if_single=True):
    """
    Recovers specific elements from input_data based on indices_to_recover.

    Args:
        input_data (list or tuple): The input data.
        indices_to_recover (int or list): The indices to recover.
        auto_unpack_if_single (bool, optional): Whether to automatically unpack the result if only one element is recovered.
            Defaults to True.

    Returns:
        Recovered data based on the provided indices.

    Examples:
        >>> input_data = ['crap', 'important', 'crap2']
        >>> GuidedDataRecoverer(input_data, 1)
        'important'

        >>> GuidedDataRecoverer(input_data, -2)
        'important'

        >>> GuidedDataRecoverer(input_data, -2, auto_unpack_if_single=False)
        ['important']

        >>> GuidedDataRecoverer(input_data, [0, 1])
        ['crap', 'important']
    """
    if input_data is None:
        return None

    assert isinstance(input_data, (list, tuple)), "input_data should be a list or a tuple"

    # create a list/tuple with all the values
    if isinstance(indices_to_recover, int):
        indices_to_recover = [indices_to_recover]

    for val in indices_to_recover:
        assert isinstance(val, int), 'indices must be integers'

    data_to_be_recovered = [input_data[idx] for idx in indices_to_recover]

    if auto_unpack_if_single and len(indices_to_recover) == 1:
        return data_to_be_recovered[0]
    else:
        return data_to_be_recovered


def graded_intensity_modification(orig, parameters, is_mask):
    """
    Applies graded intensity modification to an image.

    Args:
        orig (ndarray): The input image.
        parameters: Additional parameters (not used in the function).
        is_mask (bool): Indicates whether the input is a mask.

    Returns:
        The modified image.

    # Examples:
    #     >>> orig = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[...,1]
    #     >>> out = graded_intensity_modification(orig, None, False)
    #     >>> plt.imshow(out)
    #     >>> plt.show()
    """
    out = orig
    if not is_mask:
        # Apply 2D gradient and create random intensity graded perturbation
        out = apply_2D_gradient(orig, create_random_intensity_graded_perturbation(orig, off_centered=True))
    return [], out


if __name__ == '__main__':
    # this is a test

    input_data = ['crap', 'important', 'crap2']
    assert GuidedDataRecoverer(input_data, 1) == 'important'
    assert GuidedDataRecoverer(input_data, -2) == 'important'
    assert GuidedDataRecoverer(input_data, -2, auto_unpack_if_single=False) == ['important']
    assert GuidedDataRecoverer(input_data, [0, 1]) == ['crap', 'important']

    # try pipes to get the stuff done
    # so easy with random to apply the same the only difficulty is that the nb of random calls should be the same --> inportant to keep in mind when doing this

    from datetime import datetime
    seed = datetime.now()
    random.seed(seed)

    orig = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[...,1]
    out = GuidedDataRecoverer(graded_intensity_modification(orig, None,False),-1)
    plt.imshow(out)
    plt.show()

    random.seed(seed)
    random.randint(0,12)# any extra random call messes it up but that makes sense

    # whatever happens the nb of random calls should be the same --> could add a fake option that calls random the exact same nb of times --> TODO
    # if I have that parameters become useless --> ok most likely

    orig = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[..., 1]
    out = GuidedDataRecoverer(graded_intensity_modification(orig, None, False), -1)
    plt.imshow(out)
    plt.show()



