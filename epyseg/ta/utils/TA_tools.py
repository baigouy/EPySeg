


# pad a number the TA way --> TODO maybe make it a generic fonction --> TODO
def get_TA_clone_file_name(clone_nb=0):
    # if clone_nb == 0:
    #     return 'tracked_clone.tif'
    # return 'tracked_clone_'+ f'{clone_nb:03}' + '.tif'
    return  get_TA_db_name(clone_nb=clone_nb)+ '.tif'

def get_TA_db_name(clone_nb=0):
    if clone_nb == 0:
        return 'tracked_clone'
    return 'tracked_clone_'+ f'{clone_nb:03}'