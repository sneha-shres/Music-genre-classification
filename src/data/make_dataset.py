import click
import kagglehub
import logging
from dotenv import load_dotenv
import shutil
import os


load_dotenv()


RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
KAGGLE_PATH = os.getenv("KAGGLE_DATASET")
RAW_WAVEFORM_PATH = os.getenv("RAW_WAVEFORM_PATH")


def downloadAndSave() -> str:
    """Download and extract GTZAN dataset from KaggleHub."""
    path = kagglehub.dataset_download(KAGGLE_PATH)+"/Data"
    return path


def moveData(source, destination) -> None:
    """Move files or directories from source to destination."""
    if os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok=True) # to move whole directory
    else:
        shutil.copy2(source, destination) # to move a file
    

@click.command() #handling command-line arguments
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # confguring logging, showing in cmd INFO level and higher with format date-type-message
    logger = logging.getLogger(__name__) # logger instance for the current module
    logger.info("Prepraing raw data set")


    # if data exists, process
    absFilePath = os.path.join(os.getcwd(), RAW_WAVEFORM_PATH)
    
    if os.path.exists(absFilePath):
        logger.info("Data already exists")

    else:
        logger.info("No data, start downloading...")
        target_dir = os.path.join(os.getcwd(), RAW_DATA_DIR)
        os.makedirs(target_dir, exist_ok=True) # create dir if doesn't exist


        try:
            # kaggle download path
            downloaded_path = downloadAndSave() 

            # move each item in dowanloded to where we want it
            for item in os.listdir(downloaded_path):
                source_item = os.path.join(downloaded_path, item)
                target_item = os.path.join(target_dir, item)
                moveData(source_item, target_item)
                
                
            # remove dir where originally donwloaded
            cache_dir = os.path.expanduser("~/.cache/kagglehub/datasets")
            pathToDelete = os.path.join(cache_dir, *KAGGLE_PATH.split("/"))
            if os.path.isdir(pathToDelete):
                shutil.rmtree(pathToDelete)
            logger.info("Data downloaded")

        except Exception as e:
            logger.error(f"An error occurred during the dataset downloading: {e}")
            raise 




if __name__=="__main__":
    main()