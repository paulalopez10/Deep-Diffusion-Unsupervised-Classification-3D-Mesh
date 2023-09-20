import os
import torch
import pandas as pd
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser


from src.models import  PointNetfeat, Net_DeepDiff
from src.datasets import DataModule, h5_dataset
from utils import retrieval




def save_predictions(model, output_fname):

    preds = torch.cat(model.predictions, dim=0)
    targets = torch.cat(model.targets, dim=0)

    mean_average_precision = retrieval(preds, targets)

    print('MAP test')
    print(mean_average_precision)


    df = pd.DataFrame(data=preds.cpu().numpy())
    df['target'] = targets.cpu().numpy()
    df.to_csv(output_fname, index=False)


def main(hparams):
    #Directory to 
    train_data_dir = hparams.train_data
    test_data_dir = hparams.test_data



    val_split = hparams.val_split #Percentage of training data used for validation

    sub_model = PointNetfeat


    epochs = hparams.epochs
    batch_size = hparams.batch_size




    num_workers = hparams.num_workers
    hidden_features = 1024
    latent_space_size = hparams.latent_space_size

    param_lambda = hparams.param_lambda
    n_knn = hparams.n_knn


  
    pl.seed_everything(42, workers=True)

    # Load the data
    data = DataModule(dataset=h5_dataset, path_train=train_data_dir, path_test=test_data_dir, val_split=val_split,
                          batch_size=batch_size, num_workers=num_workers)





    print()
    print('=============================================================')
    print(f'Number of samples: {len(data.dataset)}')
    print(f'Training: {len(data.train_set)}')
    print(f'Validation: {len(data.val_set)}')
    print(f'Testing: {len(data.test_set)}')
    print()
    print('=============================================================')
    sample = data.train_set[0]['x']
    n_points = sample['pos'].shape[0]
    print(f'Number of points per mesh:{n_points}')


    # Create output directory
    out_name = hparams.out_name
    out_dir = 'output/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    

    #Diffusion sources
    x_sources = {}
    for item in data.train_set.indices :
        one_hot = torch.zeros((len(data.train_set.indices)))
        one_hot[np.sort(data.train_set.indices)==item]=1
        x_sources[item] = one_hot

    for item in data.val_set.indices :
        one_hot = torch.zeros((len(data.train_set.indices)))
        one_hot[np.sort(data.train_set.indices)==item]=1
        x_sources[item] = one_hot

    for item in data.test_set.indices :
        one_hot = torch.zeros((len(data.train_set.indices)))
        one_hot[np.sort(data.train_set.indices)==item]=1
        x_sources[item] = one_hot


   
    model = Net_DeepDiff(num_sources=len(data.train_set), hidden_features=hidden_features, num_outputs=latent_space_size,  sub_model=sub_model,  n_knn=n_knn, param_lambda=param_lambda, batch_size=batch_size, x_sources=x_sources)





    # train
    checkpoint_callback = ModelCheckpoint(monitor="val_auc", mode='max')
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 1,
        max_epochs=epochs,
        logger=TensorBoardLogger(out_dir, name=out_name),
        strategy='ddp_find_unused_parameters_true',
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)


    model = Net_DeepDiff.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_sources=len(data.train_set), hidden_features=hidden_features, num_outputs=latent_space_size,  sub_model=sub_model,  n_knn=n_knn, param_lambda=param_lambda, batch_size=batch_size, x_sources=x_sources)
    
    print('=============================================================')
    print('Testing model')

    trainer.test(model=model, datamodule=data)

    save_predictions(model=model, output_fname=os.path.join(out_dir, 'predictions_test.csv'))
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_data', default='./data/trainingset.h5')
    parser.add_argument('--test_data', default='./data/testingset.h5')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--latent_space_size', default=256)
    parser.add_argument('--param_lambda', default=1)
    parser.add_argument('--n_knn', default=20)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--val_split', default=0.1)
    parser.add_argument('--out_name', default='PointNet-DD')
    args = parser.parse_args()

    main(args)
