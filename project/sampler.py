import os
import shutil
import numpy as np
import logging
logger = logging.getLogger('Sampler')
logging.basicConfig(filename='log.log', filemode='w', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    try:
        files = os.listdir("project/test_inputs/")
        logger.info("Successfully extracted the contents of test_inputs")
    except Exception as e:
        logger.exception(f"Error in retrieving the contents of test_inputs : {e}")
    randind = np.random.randint(len(files))
    
    lucky_file = files[randind]
    complete_file_path = os.path.join("project/test_inputs/",lucky_file)
    print(f"The selected word is {lucky_file[:-4]}")
    try:
        shutil.copy(complete_file_path, "project/text_image.json")
        logger.info(f"Successfully copied {lucky_file} as test file")
    except Exception as e:
        logger.exception(f"{e}: \n Choose a different file")
    
    