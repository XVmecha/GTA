#from anomaly_detection import find_optimal_threshold
from data.data_loader_dad import (
    NASA_Anomaly,
    WADI,
    SWaT
)
import json
from exp.exp_basic import Exp_Basic
from models.gta import GTA
from sklearn.metrics import f1_score, recall_score, precision_score

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from sklearn.metrics import classification_report

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')


def find_optimal_threshold(anomaly_scores, true_labels, setting, n_thresholds=100,
                           experiment_name='experiment'):
    """
    Perform grid search to find optimal threshold for anomaly detection.

    Parameters:
    -----------
    anomaly_scores : numpy.ndarray
        Anomaly scores for each time point, shape (N,)
    true_labels : numpy.ndarray
        Ground truth labels (0 for normal, 1 for anomaly), shape (N,)
    n_thresholds : int
        Number of threshold values to try
    save_path : str
        Directory path to save results file
    experiment_name : str
        Name for the experiment (used in filename)

    Returns:
    --------
    dict
        Results containing best threshold, F1 score, recall, and precision
    """
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    save_path = './results/' + setting + '/'
    thresholds = np.linspace(min_score, max_score, n_thresholds)

    best_f1 = 0
    best_recall = 0
    best_f1_threshold = None
    best_recall_threshold = None
    best_f1_precision = 0
    best_recall_precision = 0

    results = {
        'thresholds': [],
        'f1_scores': [],
        'recall_scores': [],
        'precision_scores': []
    }

    for threshold in thresholds:
        # Apply threshold to get binary predictions
        predicted_labels = (anomaly_scores > threshold).astype(int)

        # Calculate metrics
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        # Store results
        results['thresholds'].append(float(threshold))
        results['f1_scores'].append(float(f1))
        results['recall_scores'].append(float(recall))
        results['precision_scores'].append(float(precision))

        # Track best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_f1_recall = re
            best_f1_threshold = threshold
            best_f1_precision = precision

        # Track best recall
        if recall > best_recall:
            best_recall = recall
            best_recall_threshold = threshold
            best_recall_precision = precision

    # Create a dictionary with summary results
    summary = {
        'best_f1': float(best_f1),
        'best_f1_threshold': float(best_f1_threshold),
        'best_f1_precision': float(best_f1_precision),
        'best_f1_recall': float(best_f1_recall),
        'best_recall': float(best_recall),
        'best_recall_threshold': float(best_recall_threshold),
        'best_recall_precision': float(best_recall_precision),
        'results': {
            'thresholds': results['thresholds'],
            'f1_scores': results['f1_scores'],
            'recall_scores': results['recall_scores'],
            'precision_scores': results['precision_scores']
        }
    }

    # Save results to file if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(save_path, f"{experiment_name}_threshold_results.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Save a simple text summary
        txt_path = os.path.join(save_path, f"{experiment_name}_threshold_summary.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Best F1 Score: {best_f1:.4f}\n")
            f.write(f"Best F1 Threshold: {best_f1_threshold:.4f}\n")
            f.write(f"Precision at Best F1: {best_f1_precision:.4f}\n")
            f.write(f"Recall at Best F1: {best_f1_recall:.4f}\n\n")
            f.write(f"Best Recall: {best_recall:.4f}\n")
            f.write(f"Best Recall Threshold: {best_recall_threshold:.4f}\n")
            f.write(f"Precision at Best Recall: {best_recall_precision:.4f}\n")

        print(f"Results saved to {json_path} and {txt_path}")

    return summary

class Exp_GTA_DAD(Exp_Basic):
    def __init__(self, args):
        super(Exp_GTA_DAD, self).__init__(args)

    def extract_and_save_embeddings(self, setting):
        """
        Extract and save node embeddings and graph structure after training

        Parameters
        ----------
        setting : str
            Name of the experiment setting to use as filename prefix

        Returns
        ----------
        dict
            Dictionary containing node embeddings and graph structure
        """
        # Load the best model if not already loaded
        best_model_path = f'./checkpoints/{setting}/checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

        # Set model to evaluation mode
        self.model.eval()

        # Get a batch of data to feed through the model
        # Use validation data to generate representative embeddings
        val_data, val_loader = self._get_data(flag='val')

        # Get first batch
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label in val_loader:
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)
            break

        # Extract graph structure
        with torch.no_grad():
            # Get graph logits - representing edge weights
            graph_logits = self.model.gt_embedding.gc_module.logits.detach().cpu().numpy()

            # Reshape logits to adjacency matrix (num_nodes × num_nodes)
            num_nodes = self.args.num_nodes
            adjacency_matrix = graph_logits.reshape(num_nodes, num_nodes, 2)[:, :, 0]

            # Get edge index for graph topology
            edge_index = self.model.gt_embedding.edge_index.detach().cpu().numpy()

            # Generate node embeddings by running the graph temporal embedding
            node_embeddings = self.model.gt_embedding(batch_x).detach().cpu().numpy()

            # Average embeddings across time and batch dimensions for a compact representation
            # Shape: [batch_size, seq_len, num_nodes] -> [num_nodes, embedding_dim]
            avg_node_embeddings = np.mean(node_embeddings, axis=(0, 1))

        # Save embeddings and graph structure
        embeddings_path = f'./results/{setting}/embeddings/'
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)

        # Save to numpy files
        np.save(f'{embeddings_path}node_embeddings.npy', avg_node_embeddings)
        np.save(f'{embeddings_path}adjacency_matrix.npy', adjacency_matrix)
        np.save(f'{embeddings_path}edge_index.npy', edge_index)

        # Create a dictionary with all the information
        embeddings_dict = {
            'node_embeddings': avg_node_embeddings,
            'adjacency_matrix': adjacency_matrix,
            'edge_index': edge_index,
            'num_nodes': num_nodes
        }

        print(f"Embeddings and graph structure saved to {embeddings_path}")

        return embeddings_dict

    def _build_model(self):
        model_dict = {
            'gta':GTA,
        }
        if self.args.model=='gta':
            model = model_dict[self.args.model](
                self.args.num_nodes,
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.num_levels,
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.data,
                self.args.activation,
                self.device
            )
        
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'SMAP':NASA_Anomaly,
            'MSL':NASA_Anomaly,
            'WADI':WADI,
            'SWaT':SWaT,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_label) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)

            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # decoder input
            # dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
            batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

                loss = criterion(outputs, batch_y) + \
                        torch.sum(torch.abs(self.model.gt_embedding.gc_module.logits[:, 0]))
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        labels = []
        
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_label) in enumerate(test_loader):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

                pred = outputs.detach().cpu().numpy()#.squeeze()
                true = batch_y.detach().cpu().numpy()#.squeeze()
                batch_label = batch_label.long().detach().numpy()
                
                preds.append(pred)
                trues.append(true)
                labels.append(batch_label)

        preds = np.array(preds)
        trues = np.array(trues)
        labels = np.array(labels)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        np.save(folder_path+'label.npy', labels)

        preds = preds[:,0,:]
        trues = trues[:,0,:]
        labels = labels[:,0]

        squared_diff = np.square(trues - preds)
        anomaly_scores = np.sum(squared_diff, axis=-1)

        find_optimal_threshold(anomaly_scores,labels,setting)
        return


