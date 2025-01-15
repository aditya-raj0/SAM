"""

TLDR : SINCE SAM USES A PRETRAINED MODEL TO COMAPRE THE ID VLAUES OF THE IMPUT AND GENERATED IMAGE AND PASS IT AS A LOSS WITH A LAMBDA TO ENSORE MINIMAL LOSS OF IDENTITY 
             WHILE REAGING. THIS IS THE CODE FOR CALCULATING THE LOSS USING A PT MODLE.

"""
import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from PIL import Image
from torchvision.transforms import transforms


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        id_image = Image.open('/home/adity/SAM/datasets/00284.jpg')
        id_image = id_image.convert('RGB')
        basic_transform = transforms.Compose([
            transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        tr_id_image = basic_transform(id_image)
        tr_id_image = tr_id_image.unsqueeze(0).to('cuda')
        self.tr_id_image = tr_id_image

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x, label=None, weights=None):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        total_loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        id_image_feats = self.extract_feats(self.tr_id_image)
        for i in range(n_samples):

            diff_target = y_hat_feats[i].dot(y_feats[i])
            new_diff_target = y_hat_feats[i].dot(id_image_feats[0])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])

            if label is None:
                id_logs.append({'diff_target': float(new_diff_target),
                                'diff_input': float(diff_input),
                                'diff_views': float(diff_views)})
            else:
                id_logs.append({f'diff_target_{label}': float(new_diff_target),
                                f'diff_input_{label}': float(diff_input),
                                f'diff_views_{label}': float(diff_views)})

            loss = 1 - new_diff_target
            if weights is not None:
                loss = weights[i] * loss

            total_loss += loss
            id_diff = float(new_diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return total_loss / count, sim_improvement / count, id_logs


        # n_samples = x.shape[0]
        # x_feats = self.extract_feats(x)
        # y_feats = self.extract_feats(y)
        # y_hat_feats = self.extract_feats(y_hat)
        # y_feats = y_feats.detach()
        # total_loss = 0
        # sim_improvement = 0
        # id_logs = []
        # count = 0
        # for i in range(n_samples):
        #     diff_target = y_hat_feats[i].dot(y_feats[i])
        #     diff_input = y_hat_feats[i].dot(x_feats[i])
        #     diff_views = y_feats[i].dot(x_feats[i])

        #     if label is None:
        #         id_logs.append({'diff_target': float(diff_target),
        #                         'diff_input': float(diff_input),
        #                         'diff_views': float(diff_views)})
        #     else:
        #         id_logs.append({f'diff_target_{label}': float(diff_target),
        #                         f'diff_input_{label}': float(diff_input),
        #                         f'diff_views_{label}': float(diff_views)})

        #     loss = 1 - diff_target
        #     if weights is not None:
        #         loss = weights[i] * loss

        #     total_loss += loss
        #     id_diff = float(diff_target) - float(diff_views)
        #     sim_improvement += id_diff
        #     count += 1

        # return total_loss / count, sim_improvement / count, id_logs
