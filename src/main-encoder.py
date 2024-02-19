import graph_tool as gt
import os
import pathlib
import warnings
import argparse
import datetime
import json
import numpy as np
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from models.transformer_model import GraphTransformer
import torch.nn.functional as F
#import timm

#assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_rdm import train_one_epoch
from omegaconf import OmegaConf
from rdm.util import instantiate_from_config
from rdm.models.diffusion.ddim import DDIMSampler
from pathlib import Path

warnings.filterwarnings("ignore", category=PossibleUserWarning)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def get_args_parser():
    parser = argparse.ArgumentParser('RDM training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # config
    parser.add_argument('--config', default='./rdm/rdm.yaml', type=str, help='config file')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--cosine_lr', action='store_true',
                        help='Use cosine lr scheduling.')
    parser.add_argument('--warmup_epochs', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_output_features):
        super(GCN, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolution
        x = self.conv2(x, edge_index)

        # You can include global pooling here if you need a fixed size output vector for the whole graph
        x = global_mean_pool(x, data.batch)

        return x

def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

    print(resume)
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def contrastive_loss(z1, z2, temperature=0.5):
    """
    Compute contrastive loss between two sets of vectors, using the InfoNCE loss.

    Parameters:
    - z1: embeddings of the first set of augmented graphs
    - z2: embeddings of the second set of augmented graphs
    - temperature: a temperature hyperparameter, controlling the separation of the distributions

    Returns:
    - loss: computed contrastive loss
    """
    # Normalize the representations along the features
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Compute similarity scores
    scores = torch.matmul(z1, z2.T) / temperature

    # For each element in z1, the corresponding element in z2 is the positive; others are negatives
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size).to(z1.device)
    loss = F.cross_entropy(scores, labels)

    return loss


import random


def my_transform(batch, feature_drop_rate=0.5, edge_drop_rate=0.):
    """
    Perform random feature dropping or edge dropping to augment graph data.

    Parameters:
    - batch: batch of graph data
    - feature_drop_rate: probability of dropping features
    - edge_drop_rate: probability of dropping edges

    Returns:
    - batch: augmented graph data
    """

    # Assuming 'batch' is a batch of graphs from a library like PyTorch Geometric or DGL

    # Feature dropping
    if feature_drop_rate > 0:
        drop_mask = torch.bernoulli((1 - feature_drop_rate) * torch.ones(batch.x.shape)).to(batch.x.device)
        batch.x = batch.x * drop_mask

    # Edge dropping

    if edge_drop_rate > 0:
        edge_keep_mask = [random.random() > edge_drop_rate for _ in range(batch.edge_index.size(1))]

        # Apply the mask to the edge index
        new_edge_index = batch.edge_index[:, edge_keep_mask]

        print(new_edge_index.shape)

        # If edge attributes exist, apply the same mask to them
        new_edge_attr = batch.edge_attr[edge_keep_mask, :] if batch.edge_attr is not None else None

        print(new_edge_attr.shape)

        # Update batch with the new edges and edge attributes
        batch.edge_index = new_edge_index
        batch.edge_attr = new_edge_attr

    return batch


# Note: The above transformation functions are placeholders and need to be adapted
# to fit the actual structure and framework you are using for your graph data.

def define_compute_extra_data(extra_features, domain_features):
    """ At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input. """
    def compute_extra_data(noisy_data):
        extra_features_extracted = extra_features(noisy_data)
        extra_molecular_features = domain_features(noisy_data)

        extra_X = torch.cat((extra_features_extracted.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features_extracted.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features_extracted.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    return compute_extra_data


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):

    #cfg.general.test_only='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-21/14-57-55-planar/checkpoints/planar/checkpoint_planar.ckpt'

    cfg['train'].batch_size//=2


    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'ego']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos,EgoDataModule
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization


        if dataset_config['name'] == 'sbm':
            datamodule = SpectreGraphDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            datamodule = SpectreGraphDataModule(cfg)
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "ego":
            #datamodule = EgoDataModule(cfg)
            datamodule = SpectreGraphDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule)
        else:
            datamodule = SpectreGraphDataModule(cfg)
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)

        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))





    model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=dataset_infos.input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                         output_dims={'X': 128, 'E': 128, 'y': 0},
                         act_fn_in=torch.nn.ReLU(),
                                      act_fn_out=torch.nn.ReLU(),
                                                     if_cond=False).cuda()

    #model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-25/22-17-02-planar/model_epoch_375.pt'
    #model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-29/12-15-59-sbm/model_epoch_100.pt' #sbm
    #model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-26/00-15-10-planar/model_epoch_3000.pt'
    #model.load_state_dict(torch.load(model_save_path))


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Prepare model for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 10000
    model.train()

    model.compute_extra_data= define_compute_extra_data(extra_features, domain_features)

    for epoch in range(num_epochs):
        embs=[]
        for batch in datamodule.train_dataloader():
            optimizer.zero_grad()

            batch=batch.cuda()
            embeddings = model.encode_batch(batch)

            # Apply any necessary transformations and send the batch to the device
            transformed_batch = my_transform(batch)  # Replace with your actual transformation function
            transformed_batch = transformed_batch.to(device)

            # Get embeddings from the GNN

            embeddings_transformed = model.encode_batch(transformed_batch)


            # Compute the contrastive loss
            # You will need to modify this call according to your loss function's API
            loss = contrastive_loss(embeddings,embeddings_transformed)

            # Backpropagation
            loss.backward()
            optimizer.step()

            embs.append(embeddings.cpu().detach().numpy())

        #np.save(f'embeddings_epoch_{epoch+1}.npy', np.concatenate(embs, axis=0))


        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if (epoch + 1) % 25 == 0:
            model_save_path = f'model_epoch_{epoch + 1}.pt'  # Define your model saving path
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at epoch {epoch + 1}')



if __name__ == '__main__':
    main()
