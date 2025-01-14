"""

TLDR : THESE ARE THE TRAIN AND TEST DATASETS THAT ARE REFFERED TO AS DATASETS['ffhq_aging'] in the training arguments by the name of 'train_source_root' ....
			 THE ACTUAL PATH OF THE ROOTS ARE MENTIONED IN dataset_paths IN path_config

"""
from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_aging': {
		'transforms': transforms_config.AgingTransforms,
        
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],

		# 'train_source_root': dataset_paths['train_source'],
		# 'train_target_root': dataset_paths['train_target'],
		# 'test_source_root': dataset_paths['test_source'],
		# 'test_target_root': dataset_paths['test_target'],
	}
}
