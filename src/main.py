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



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):

    #cfg.general.test_only='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-21/14-57-55-planar/checkpoints/planar/checkpoint_planar.ckpt'


    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'ego']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "ego":
            #datamodule = EgoDataModule(cfg)
            datamodule = SpectreGraphDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule)
        else:
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

    elif dataset_config["name"] == 'protein':
        from datasets import protein_dataset

        datamodule = protein_dataset.ProteinDataModule(cfg)
        dataset_infos = protein_dataset.ProteinInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders


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

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])



    load_rdm=False

    if load_rdm:
        rdm_config = OmegaConf.load(pretrained_rdm_cfg)
        self.rdm_fake_class_label = rdm_config.model.params.cond_stage_config.params.n_classes - 1
        rdm_model = load_model(rdm_config, pretrained_rdm_ckpt)
        self.rdm_sampler = DDIMSampler(rdm_model)

        # sampling RDM

        with self.rdm_sampler.model.ema_scope("Plotting"):
            shape = [self.rdm_sampler.model.model.diffusion_model.in_channels,
                     self.rdm_sampler.model.model.diffusion_model.image_size,
                     self.rdm_sampler.model.model.diffusion_model.image_size]
        if self.rdm_sampler.model.class_cond:
            cond = {"class_label": class_label}
        else:
            class_label = self.rdm_fake_class_label * torch.ones(bsz).cuda().long()
            cond = {"class_label": class_label}
        cond = self.rdm_sampler.model.get_learned_conditioning(cond)

        sampled_rep, _ = self.rdm_sampler.sample(rdm_steps, conditioning=cond, batch_size=bsz,
                                                 shape=shape,
                                                 eta=eta, verbose=False)
        sampled_rep = sampled_rep.squeeze(-1).squeeze(-1)


        # add uncond for cfg
        if cfg > 0:
            uncond_rep = self.fake_latent.repeat(bsz, 1)
            sampled_rep = torch.cat([sampled_rep, uncond_rep], dim=0)


    # load model config
    args_dict = {
        "batch_size": 128,
        "epochs": 200,
        "accum_iter": 1,
        "config": "../../../src/rdm/rdm.yaml",
        "weight_decay": 0.01,
        "lr": None,
        "blr": 1e-6*10,
        "min_lr": 0.0,
        "cosine_lr": False,
        "warmup_epochs": 0,
        "output_dir": "../../../src/output_dir",
        "log_dir": "../../../src/output_dir",
        "device": "cuda",
        "seed": 0,
        "resume": "",
        "start_epoch": 0,
        "num_workers": 10,
        "pin_mem": True,
        "world_size": 1,
        "local_rank": -1,
        "dist_on_itp": False,
        "dist_url": "env://"
    }

    args = DictConfig(args_dict)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir

    config = OmegaConf.load(args.config)
    rdm_model = instantiate_from_config(config.model)
    rdm_model.pretrained_encoder=  GraphTransformer(n_layers=cfg.model.n_layers if dataset_config["name"]!='ego' else 2,
                                                    input_dims=dataset_infos.input_dims,
                                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                                    hidden_dims=cfg.model.hidden_dims,
                                                    output_dims={'X': 128, 'E': 128, 'y': 0},
                                                    act_fn_in=torch.nn.ReLU(),
                                                    act_fn_out=torch.nn.ReLU(),
                                                    if_cond=False)

    #model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-23/21-35-14-planar/model_epoch_7000.pt'

    if dataset_config["name"] == 'planar':
        model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-26/00-15-10-planar/model_epoch_3000.pt'
    elif dataset_config["name"] == 'qm9':
        model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-27/17-39-38-qm9h/model_epoch_150.pt'
    elif dataset_config["name"] == 'sbm':
        model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-29/12-15-59-sbm/model_epoch_100.pt'
    elif dataset_config["name"] == 'ego':
        model_save_path = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-02-04/05-08-12-ego/model_epoch_200.pt'



    rdm_model.pretrained_encoder.load_state_dict(torch.load(model_save_path))

    rdm_model.pretrained_encoder.cuda()
    rdm_model.pretrained_encoder.eval()
    rdm_model.pretrained_encoder.train = disabled_train
    try:
        rdm_model.pretrained_enc_withproj = config.params.pretrained_enc_withproj
    except:
        rdm_model.pretrained_enc_withproj = False

    for param in rdm_model.pretrained_encoder.parameters():
        param.requires_grad = False

    # #GCN(num_node_features = 1, num_output_features = 256 )
    rdm_model.cuda()
    args.class_cond = config.model.params.get("class_cond", False)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size

    print("base lr: %.2e" % (args.lr / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Log parameters
    model_without_ddp=rdm_model
    params = list(model_without_ddp.model.parameters())
    params = params + list(model_without_ddp.cond_stage_model.parameters())
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))


    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()



    #cfg.general.samples_to_generate=200

    #cfg.general.test_only='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-25/04-02-02-planar/checkpoints/planar/epoch=3499.ckpt'
    #cfg.general.resume = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-25/15-42-04-planar/checkpoints/planar/last.ckpt'
    #cfg.general.resume='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-26/00-39-44-planar/checkpoints/planar/last.ckpt'


    #cfg.general.test_only='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-26/10-45-03-planar/checkpoints/planar/last.ckpt' # ori
    #cfg.general.test_only='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-27/13-22-53-planar/checkpoints/planar/last.ckpt' # normed
    #cfg.general.test_only='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-01-29/15-02-27-sbm/checkpoints/sbm/last.ckpt' # normed
    #cfg.general.test_only = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-02-02/18-13-59-sbm/checkpoints/sbm/last.ckpt'  # normed
    cfg.general.test_only = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-02-03/20-54-06-sbm/checkpoints/sbm/last.ckpt'  # normed
    #cfg.general.test_only = '/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-02-06/15-22-10-ego/checkpoints/ego/last.ckpt'  # normed

    #cfg.general.resume='/home/sw3wv/Desktop/SelfCon/DiGress/outputs/2024-02-02/18-13-59-sbm/checkpoints/sbm/last.ckpt'


    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, args=args, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, args=args, **model_kwargs)

    model.rdm_model=rdm_model
    model.rdm_optimizer=optimizer
    model.rdm_loss_scaler=loss_scaler
    model.rdm_sampler = DDIMSampler(rdm_model)



    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)




    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      #limit_val_batches=0.,  # remove this if running validation
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])



    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
