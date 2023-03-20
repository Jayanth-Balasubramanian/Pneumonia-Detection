DATA_PATH = "./data/chest_xray"

TRAIN_PATH = f'{DATA_PATH}/train'
TEST_PATH = f'{DATA_PATH}/test'
VAL_PATH = f'{DATA_PATH}/val'
MODEL_SAVE_PATH = "./model-store"
# data config
MODEL_INPUT_SHAPE = (512, 512)
CENTRE_CROP = 400
BATCH_SIZE = 32
