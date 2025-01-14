dataset_paths = {
    'celeba_test': '/home/adity/SAM/datasets/demo',         # CelebA id both the test source and target, as mentioned in the data_config
    'ffhq': '/home/adity/SAM/datasets/demo',                # FFHQ is both the train source and target here, look at the data_config

    'train_source' : '/home/adity/SAM/datasets/demo_train_source',     # these are different source and target files for both train and test, to use them change the address in data_congifs
    'train_target' : '/home/adity/SAM/datasets/demo_train_target',
    'test_source' : '/home/adity/SAM/datasets/demo_test_source',
    'test_target' : '/home/adity/SAM/datasets/demo_test_target',
}

model_paths = {
    'pretrained_psp': '/home/adity/SAM_Aux_Models/psp_ffhq_encode.pt',
    'ir_se50': '/home/adity/SAM_Aux_Models/model_ir_se50.pth',
    'stylegan_ffhq': '/home/adity/SAM_Aux_Models/stylegan2-ffhq-config-f.pt',
    'shape_predictor': '/home/adity/SAM_Aux_Models/shape_predictor_68_face_landmarks.dat',
    'age_predictor': '/home/adity/SAM_Aux_Models/dex_age_classifier.pth'
}
