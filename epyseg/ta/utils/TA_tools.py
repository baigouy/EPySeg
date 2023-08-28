def get_TA_clone_file_name(clone_nb=0):
    """
    Get the TA clone file name based on the clone number.

    Args:
        clone_nb (int): Clone number (default: 0).

    Returns:
        TA clone file name.

    Examples:
        >>> file_name = get_TA_clone_file_name(clone_nb=0)
        >>> print('TA Clone File Name:', file_name)
        TA Clone File Name: tracked_clone.tif
        >>> file_name = get_TA_clone_file_name(clone_nb=10)
        >>> print('TA Clone File Name:', file_name)
        TA Clone File Name: tracked_clone_010.tif
    """
    return get_TA_db_name(clone_nb=clone_nb) + '.tif'


def get_TA_db_name(clone_nb=0):
    """
    Get the TA database name based on the clone number.

    Args:
        clone_nb (int): Clone number (default: 0).

    Returns:
        TA database name.

    Examples:
        >>> db_name = get_TA_db_name(clone_nb=0)
        >>> print('TA Database Name:', db_name)
        TA Database Name: tracked_clone
        >>> db_name = get_TA_db_name(clone_nb=10)
        >>> print('TA Database Name:', db_name)
        TA Database Name: tracked_clone_010

    """
    if clone_nb == 0:
        return 'tracked_clone'
    return 'tracked_clone_' + f'{clone_nb:03}'
