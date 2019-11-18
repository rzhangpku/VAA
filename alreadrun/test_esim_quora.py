"""
Train the ESIM model on the preprocessed SNLI dataset.
"""
# Aurelien Coet, 2018.

from utils.runned.utils_test_esim_quora import validate
from mfae.model import ESIM
from mfae.data import NLIDataset
from torch.utils.data import DataLoader
import os
import argparse
import pickle
import json
import torch

import matplotlib
matplotlib.use('Agg')


def main(train_file,
         valid_file,
         test_file,
         embeddings_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=3,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None):
    """
    Train the ESIM model on the Quora dataset.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the model.
        valid_file: A path to some preprocessed data that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    # print("\t* Loading training data...")
    # with open(train_file, "rb") as pkl:
    #     train_data = NLIDataset(pickle.load(pkl))
    #
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    # print("\t* Loading test data...")
    # with open(test_file, "rb") as pkl:
    #     test_data = NLIDataset(pickle.load(pkl))
    #
    # test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
            .to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_accuracy = validate(model,
                                             valid_loader)
    print("\t* Validation accuracy: {:.4f}%".format(valid_accuracy*100))




if __name__ == "__main__":
    default_config = "../../config/training/quora_training.json"

    parser = argparse.ArgumentParser(
        description="Train the ESIM model on quora")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = script_dir + '/scripts/training'

    parser.add_argument("--checkpoint",
                        default=os.path.dirname(os.path.realpath(__file__)) + '/data/checkpoints/quora/' +"best.pth.tar",
                        help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
         os.path.normpath(os.path.join(script_dir, config["valid_data"])),
         os.path.normpath(os.path.join(script_dir, config["test_data"])),
         os.path.normpath(os.path.join(script_dir, config["embeddings"])),
         os.path.normpath(os.path.join(script_dir, config["target_dir"])),
         config["hidden_size"],
         0,
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         args.checkpoint)
