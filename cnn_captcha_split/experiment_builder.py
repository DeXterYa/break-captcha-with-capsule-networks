import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from torch.optim.adam import Adam
from storage_utils import save_statistics


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        self.experiment_name = experiment_name
        self.model = network_model
        self.device = torch.cuda.current_device()

        # Use multiple GPUs if possible
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = Adam(self.parameters(), amsgrad=False,
                              weight_decay=weight_decay_coefficient)

        total_num_parameters = sum(p.numel() for p in self.model.parameters())
        print('Total number of parameters', total_num_parameters)

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.spread_loss = SpreadLoss().to(self.device)

        # Check if to load model or start fresh
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def run_train_iter(self, x, y, batch_idx, epoch_idx):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A tensor of shape (batch_size, channels, height, width)
        :param y: The targets for the model. A tensor of shape (batch_size,)
        :return: the loss and accuracy for this batch
        """
        self.train() # sets model to training mode

        x = x.to(self.device)
        y = y.to(self.device)

        outputs = self.model(x)
        # r = (1. * batch_idx + epoch_idx * len(self.train_data)) / (self.num_epochs * len(self.train_data))

        _, preds = outputs.max(dim=-1)
        # print("prediction: ", outputs[1])
        # print("target: ", y[1])
        loss = self.criterion(outputs.view(-1, 10), y.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        accuracy = np.mean(list(preds.eq(y.data).view(-1).cpu()))
        real_acc = np.mean(list(torch.all((preds.eq(y.data)), dim=1).cpu()))
        acc_1 = np.mean(list(preds.eq(y.data)[:, 0].view(-1).cpu()))
        acc_2 = np.mean(list(preds.eq(y.data)[:, 1].view(-1).cpu()))
        acc_3 = np.mean(list(preds.eq(y.data)[:, 2].view(-1).cpu()))

        return loss.data.detach().cpu().numpy(), accuracy, real_acc, acc_1, acc_2, acc_3

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A tensor of shape (batch_size, channels, height, width)
        :param y: The targets for the model. A tensor of shape (batch_size,)
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        x = x.to(self.device)
        y = y.to(self.device)

        outputs = self.model(x)
        _, preds = outputs.max(dim=-1)
        loss = self.criterion(outputs.view(-1, 10), y.view(-1))

        accuracy = np.mean(list((preds.eq(y.data)).view(-1).cpu()))
        real_acc = np.mean(list(torch.all((preds.eq(y.data)), dim=1).cpu()))
        acc_1 = np.mean(list(preds.eq(y.data)[:, 0].view(-1).cpu()))
        acc_2 = np.mean(list(preds.eq(y.data)[:, 1].view(-1).cpu()))
        acc_3 = np.mean(list(preds.eq(y.data)[:, 2].view(-1).cpu()))

        return loss.data.detach().cpu().numpy(), accuracy, real_acc, acc_1, acc_2, acc_3

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def run_training_epoch(self, current_epoch_losses, epoch_idx):
        with tqdm.tqdm(total=len(self.train_data), file=sys.stdout) as pbar_train:  # create a progress bar for training
            for idx, (x, y) in enumerate(self.train_data):  # get data batches
                loss, accuracy, real_acc, acc_1, acc_2, acc_3 = self.run_train_iter(x=x, y=y, batch_idx=idx, epoch_idx=epoch_idx)  # take a training iter step
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                current_epoch_losses["train_real_acc"].append(real_acc)
                current_epoch_losses['train_acc_1'].append(acc_1)
                current_epoch_losses['train_acc_2'].append(acc_2)
                current_epoch_losses['train_acc_3'].append(acc_3)
                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}, real_acc: {:.4f}".format(loss, accuracy, real_acc))

        return current_epoch_losses

    def run_validation_epoch(self, current_epoch_losses):

        with tqdm.tqdm(total=len(self.val_data), file=sys.stdout) as pbar_val:  # create a progress bar for validation
            for x, y in self.val_data:  # get data batches
                loss, accuracy, real_acc, acc_1, acc_2, acc_3 = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                current_epoch_losses["val_real_acc"].append(real_acc)
                current_epoch_losses['val_acc_1'].append(acc_1)
                current_epoch_losses['val_acc_2'].append(acc_2)
                current_epoch_losses['val_acc_3'].append(acc_3)
                pbar_val.update(1)  # add 1 step to the progress bar
                pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}, real_acc: {:.4f}".format(loss, accuracy, real_acc))

        return current_epoch_losses

    def run_testing_epoch(self, current_epoch_losses):

        with tqdm.tqdm(total=len(self.test_data), file=sys.stdout) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                loss, accuracy, real_acc, acc_1, acc_2, acc_3 = self.run_evaluation_iter(x=x,
                                                          y=y)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
                current_epoch_losses["test_real_acc"].append(real_acc)
                current_epoch_losses['test_acc_1'].append(acc_1)
                current_epoch_losses['test_acc_2'].append(acc_2)
                current_epoch_losses['test_acc_3'].append(acc_3)
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}, real_acc: {:.4f}".format(loss, accuracy, real_acc))  # update progress bar string output
        return current_epoch_losses

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_real_acc": [],"train_loss": [], "val_acc": [], "val_real_acc": [],
                        "val_loss": [], "curr_epoch": [], "train_acc_1": [], "train_acc_2": [], "train_acc_3": [],
                        "val_acc_1": [], "val_acc_2": [], "val_acc_3": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_real_acc": [], "train_loss": [], "val_acc": [], "val_real_acc": [], "val_loss": [],
                                    "train_acc_1": [], "train_acc_2": [], "train_acc_3": [],
                                    "val_acc_1": [], "val_acc_2": [], "val_acc_3": []
                                    }

            current_epoch_losses = self.run_training_epoch(current_epoch_losses, epoch_idx)
            current_epoch_losses = self.run_validation_epoch(current_epoch_losses)

            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))
                # get mean of all metrics of current epoch metrics dict,
                # to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (
                                        self.starting_epoch != 0 or i > 0) else False)  # save statistics to stats file.

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_acc'] = self.best_val_model_acc
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_acc": [], "test_real_acc": [], "test_loss": [],
                                "test_acc_1": [], "test_acc_2": [], "test_acc_3": []}  # initialize a statistics dict

        current_epoch_losses = self.run_testing_epoch(current_epoch_losses=current_epoch_losses)

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format

        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses