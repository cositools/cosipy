import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt
from histpy import Histogram

from cosipy.background_estimation.ContinuumEstimationInterp import ContinuumEstimationInterp
from cosipy.response import DetectorResponse
from mhealpy import HealpixMap, HealpixBase
import time
import logging
from tqdm.auto import tqdm
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphUNet as PyGGraphUNet

class GCN(nn.Module):
     
    def __init__(self, in_channels=1, hidden_channels=32):
       

        """
        Initialize a 4-layer Graph Convolutional Network (GCN) for node-level regression tasks.

        This model consists of three hidden graph convolutional layers with ReLU activation functions,
        followed by a final output layer without activation. Each convolutional layer aggregates
        information from neighboring nodes using the graph structure.

        Parameters
        ----------
        in_channels : int, optional
            Number of input features per node. Default is 1.
        hidden_channels : int, optional
            Number of hidden features (output channels) in each intermediate convolutional layer.
            Default is 32.

        Architecture
        ------------
        Layer 1: GCNConv(in_channels → hidden_channels) + ReLU
        Layer 2: GCNConv(hidden_channels → hidden_channels) + ReLU
        Layer 3: GCNConv(hidden_channels → hidden_channels) + ReLU
        Layer 4: GCNConv(hidden_channels → 1)  # Output layer (no activation)

        Returns
        -------
        None
        """

        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, 1)

        return

    def forward(self, x, edge_index):

        """
        Perform a forward pass through the Graph Convolutional Network (GCN).

        Applies a sequence of graph convolutional layers with ReLU activations 
        to the input node features. The final layer produces the output feature 
        representations without an activation function.

        Parameters
        ----------
        x : torch.Tensor: 
            Node feature matrix of shape (N, F_in), where 
            N is the number of nodes and F_in is the number of 
            input features per node.
        edge_index : torch.LongTensor
            Graph connectivity in COO format 
            with shape (2, E), where E is the number of edges.

        Returns
        -------
        torch.Tensor: 
            Output node feature matrix of shape (N, F_out) 
            after the final convolution layer.
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        
        return self.conv4(x, edge_index)

class ContinuumEstimationNN(ContinuumEstimationInterp):
    
    def __init__(self):
        """
        COSI Background Inpainting using PyTorch Geometric (Hybrid / Self-Supervised)

        Modes:
        - supervised: uses background model as ground truth
        - self: random masking (no model)
        - hybrid: combination of both (falls back to self if no model provided)

        Includes accuracy evaluation comparing inpainted results to true background.
        """

    def select_device(self):

        """Device selection with GPU compatibility check."""
    
        if torch.cuda.is_available():
            
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPUs available: {gpu_count}")
            capability = torch.cuda.get_device_capability(0)
            major, minor = capability
            
            # check capatability (may need to be generalized):
            if major < 3 or (major == 3 and minor < 7):
                logger.info(f"[WARNING] GPU {torch.cuda.get_device_name(0)} "
                  f"(capability {major}.{minor}) is not supported by this PyTorch build.")
                logger.info("Falling back to CPU.")
                return torch.device('cpu')
           
            else:
                logger.info(f"Using GPU 0: {torch.cuda.get_device_name(0)} (capability {major}.{minor})")
                if gpu_count > 1:
                    logger.info("Multi-GPU training not enabled; using a single GPU.")
                return torch.device('cuda')
        
        # For Mac M chips:
        elif torch.backends.mps.is_available():
            return torch.device('mps')

        else:
            logger.info("No GPU detected. Using CPU.")
            return torch.device('cpu')

    def build_healpix_graph(self, nside):
        
        """Build HEALPix graph.
        
        Parameters
        ----------
        nside : int
            nside of healpix map.

        edge_index : tensor
            Healpix graph (nodes and edges) given as tensor object. 
        """

        edges = []
        npix = hp.nside2npix(nside)
        for i in range(npix):
            neighbors = hp.get_all_neighbours(nside, i)
            for n in neighbors:
                if n >= 0:
                    edges.append([i, n])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index

    def train_inpaint(self, input_data_map, mask_map, nside,
                  mode='self', model_map=None,
                  epochs=200, lr=1e-3,
                  self_mask_fraction=0.1,
                  lambda_sup=0.5, lambda_self=0.5,
                  model_type="gcn",
                  nn_model="new",
                  nn_model_file=None,
                  nn_model_savename="inpainting_nn_model",
                  verbose=True):

        """Training function for inpainting.
        
        Parameters
        ----------
        input_data_map : array
            Input file with total data. Returned from 
            load_projected_data method. 
        mask_map : array
            Boolean array specifying mask. Returned from
            mask_from_cumdist_vectorized method. 
        nside : int
            Nside of background map.
        mode : str, optional
            Training mode to use (default is self). Choices are:
            hybrid: uses both supervised and self.
            supervised: trains from input model map.
            self: trains from random masking. 
        model_map : array, optional
            Input model map array. Returned from 
            load_projected_model method.  
        epochs : int, optional
            Number of training epochs. Default is 800. 
        lr : float, optional
            Learning rate. Default is 1e-3. 
        self_mask_fraction : float, optional
            Fraction of pixels to randomly mask in self training mode. 
        lambda_sup : float, optional
            Weight for supervised training loss function used in 
            hybride mode. Default is 0.5. 
        lambda_self : float, optional
            Weight for self-supervised training loss function used in
            hybride mode. Default is 0.5.
        model_type : str, optional 
            Which model to use. Options are gcn (default) or unet.
        nn_model : str, optional
            Train new model or load existing model. Options are 
            new (default), load (loads model weights), and 
            load_full (loads model weights, optimizer state, and epochs). 
        nn_model_file : str, optional
            Name of NN model to load. Default is None.
        nn_model_savename : str, optional
            Prefix of saved NN model. Default is inpainting_nn_model.  
        verbose : bool, optional
            Gives logger info for validation loss every 50 epochs.
            Default is False.

        Returns
        -------
        inpainted : array
            Inpainted background map. 
        """
      
        # Initiate device, either CPU or GPU if available. 
        self.device = self.select_device()

        npix = hp.nside2npix(nside)
        edge_index = self.build_healpix_graph(nside).to(self.device)
   
        # Either start new NN model or load existing model:
        if nn_model == "new":
            if model_type == "unet":
                model = PyGGraphUNet(in_channels=1, hidden_channels=32, out_channels=1, depth=3).to(self.device)
                logger.info("Using GraphUNet model.")
            else:
                model = GCN(in_channels=1, hidden_channels=32).to(self.device)
                logger.info("Using GCN model.")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if nn_model != "new" and nn_model_file == None:
            raise ValueError("Must specify nn_model_file if loading NN model.")

        if nn_model == "load":
            model = self.load_NN_Model(nn_model_file, full_state=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            logger.info(f"Loaded NN model (weights only) from {nn_model_file}")

        if nn_model == "load_full":
            model, optimizer, start_epoch, final_loss = self.load_NN_Model(nn_model_file, full_state=True)
            logger.info(f"Loaded full NN model from {nn_model_file}")

        loss_fn = nn.MSELoss()

        inpainted = np.zeros_like(input_data_map)
        n_energy, n_phi, _ = input_data_map.shape

        # training loss:
        training_loss = np.zeros([n_energy,n_phi,epochs])

        # Loss weights should only be applied in hybrid mode:
        if mode != 'hybrid':
            lambda_sup = 1
            lambda_self = 1

        # Progress bar:
        total_steps = n_energy * n_phi
        with tqdm(total=total_steps, desc="Inpainting (Em, Phi)", unit="map") as pbar:
        
            for e in range(n_energy):
                for p in range(n_phi):     

                    # advance progress bar:
                    pbar.update(1)

                    y = input_data_map[e, p].reshape(-1)
                    mask = mask_map[e, p].reshape(-1)

                    # Masked input
                    x = (y * mask).reshape(-1, 1)

                    # Convert to tensors
                    data = Data(x=torch.tensor(x, dtype=torch.float32).to(self.device),
                        edge_index=edge_index)
                    y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(self.device)

                    target_model = None
                    if model_map is not None:
                        target_model = torch.tensor(model_map[e, p], dtype=torch.float32).to(self.device).reshape(-1)

                    # Training loop
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        pred = model(data.x, data.edge_index).squeeze()

                        loss_total = 0.0

                        # Supervised component
                        if mode in ['supervised', 'hybrid'] and target_model is not None:
                            sup_loss = loss_fn(pred * (1 - mask_tensor),
                                       target_model * (1 - mask_tensor))
                            loss_total += lambda_sup * sup_loss

                        # Self-supervised component
                        if mode in ['self', 'hybrid']:
                            unmasked_idx = np.where(mask == 1)[0]
                            num_rand = max(1, int(len(unmasked_idx) * self_mask_fraction))
                            rand_mask_idx = np.random.choice(unmasked_idx, num_rand, replace=False)

                            mask_self = mask.copy()
                            mask_self[rand_mask_idx] = 0
                            mask_self_tensor = torch.tensor(mask_self, dtype=torch.float32).to(self.device)

                            self_loss = loss_fn(pred * (1 - mask_self_tensor),
                                        y_tensor * (1 - mask_self_tensor))
                            loss_total += lambda_self * self_loss

                        # Save training loss:
                        training_loss[e,p,epoch] = loss_total

                        loss_total.backward()
                        optimizer.step()

                        if verbose == True: 
                            if epoch % 50 == 0:
                                logger.info(f"[E{e} Phi{p}] Epoch {epoch}/{epochs}: Loss {loss_total.item():.6f}")

                    # Combine prediction with unmasked data
                    pred_np = pred.detach().cpu().numpy().reshape(npix)
                    inpainted[e, p] = y * mask + pred_np * (1 - mask)
    
        # write training loss array to file:
        np.save(f"{nn_model_savename}_training_loss",training_loss)

        # Save NN model:
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss_total}, f"{nn_model_savename}.pth")

        return inpainted

    def plot_training_loss(self, input_file, energy_bin, save_prefix, show_plot=True, vmax=70000):

        """Plot training loss as a function of Phi and number of epochs
        for a given energy bin.

        Parameters
        ----------
        input_file : str
            File name for input training loss array.
        energy_bin : int
            Energy bin to plot.
        save_prefix : str
            Prefix of saved image file.
        show_plot : bool, optional
            Whether to show plot (default is True).
        vmax : float, optional
            Max plot value. Default is 70000.
        """

        # Load loss data
        loss_data = np.load(input_file)  # shape: (Energy, Phi, Epochs)

        loss_slice = loss_data[energy_bin]  # shape: (Phi, Epochs)

        # Transpose to shape (Epochs, Phi) for plotting
        loss_slice = loss_slice.T

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.imshow(loss_slice, aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, loss_slice.shape[1], 0, loss_slice.shape[0]], vmax=vmax)

        plt.colorbar(label='Loss')
        plt.xlabel('Phi bin')
        plt.ylabel('Epoch')
        plt.title(f"Training Loss Map (Energy bin {energy_bin})")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png", dpi=150)
        if show_plot:
            plt.show()

        return

    def load_NN_Model(self, nn_model_file, full_state=True):

        """Loads NN model saved from torch.save.
        
        Parameters
        ----------
        nn_model_file : str
            NN model file (.pth) from torch.save.
        full_state : bool, optional
            Option to load full state, which includes model weights, 
            optimizer state, and epochs. Default is False, in which 
            case only the model weights are loaded. 
        
        Returns
        -------
        model : dict
            State dictionary with NN model weights
        optimizer : torch.optim.Adam
            Optimizer and state. Only for full_state=True.
        start_epoch : int
            Epoch number. Only for full_state=True.
        final_loss : float
            Training loss. Only for full_state=True.
        """

        logger.info("loading NN model...")

        if not self.device:
            # Initiate device, either CPU or GPU if available.
            self.device = self.select_device()

        checkpoint = torch.load(nn_model_file, map_location=torch.device(self.device))

        # Need to recreate the model architecture first
        # (same layer definitions, etc.) before loading weights.
        # For now we just use the GCN:
        model = GCN(in_channels=1, hidden_channels=32).to(self.device)

        # Load the model weights:
        model.load_state_dict(checkpoint['model_state_dict'])

        # Option to load the optimizer state and epochs:
        if full_state == True:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            final_loss = checkpoint['loss']

            return model, optimizer, start_epoch, final_loss

        else:
            return model

    def estimate_bg(self, input_data, psr_file, background_model=None, 
            training_mode="self", containment=0.6, epochs=200, model_type="gcn",
            nn_model="new", nn_model_file=None, nn_model_savename="inpainting_nn_model",
            lr=1e-3, self_mask_fraction=0.1, lambda_sup=0.5, lambda_self=0.5,  
            prefix="inpainted", visualize=False, em_bin=2, phi_bin=4, 
            evaluate_only=False, inpainted_file=None,
            evaluate=False, show_plots=False, verbose=True):

        """Convenience function for estimating the background. 
        
        Parameters
        ----------
        input_data : str
            Path to HDF5 file with the input data that will be used 
            to estimate the background.
        psr_file : str
            Path to point source response HDF5 file. 
        background_model : hist, optional
            Optional background model HDF5 file. 
        training_mode : str, optional
            Training mode to use. There are three options:  
            hybrid (combination of supervised and self-supervised), 
            supervised (training based on input model), 
            and self-supervised (training based on randomly masking a fraction of the sky). 
            Default is self-supervised.
        containment : float, optional
            Containment fraction for mask. Default is 0.6. 
        epochs : int, optional 
            Number of training epochs. Default is 800.
        model_type : str, optional
            Type of NN to use. Options are gcn or unet. Default is gcn.
        nn_model : str, optional
            Train new model or load existing model. Options are 
            new (default), load (loads model weights), and 
            load_full (loads model weights, optimizer state, and epochs). 
        nn_model_file : str, optional
            Name of NN model to load. Default is None.
        nn_model_savename : str, optional
            Prefix of saved NN model. Default is inpainting_nn_model. 
        lr : float, optional
            Learning rate. Default is 1e-3. 
        self_mask_fraction : float, optional
            Fraction of pixels to mask in self-supervised mode. Default is 0.1. 
        lambda_sup : float, optional
            Weight of loss function of supervised mode for hybrid 
            training. Default is 0.5.
        lambda_self : float, optional 
            Weight of loss function of self-supervised mode for hybrid 
            training. Default is 0.5.
        prefix : str, optional
            Prefix for output files. Default is 'inpainted'.
        visualize : boolean, optional
            Visualize Mollweide plots.
        em_bin : int, optional
            Energy bin index for visualization. Default is 2. 
        phi_bin : int, optional
            Phi bin index for visualization. Default is 4. 
        evaluate_only : boolean, optional
            Skip training and evaluate two histograms. Requires 
            background_model file and inpainted_file. 
        inpainted_file : hist, optional
            Inpainted histogram file (for evaluate-only). Default is None.
        evaluate : boolean
            Evaluate after training (inline). Default is False. 
        show_plots : boolean
            Display plots to screen. Default is False. 
        verbose : bool, optional
            Gives logger info for validation loss every 50 epochs.
            Default is False.
        """
        
        # Record run time:
        start_time = time.time()

        # --- Evaluation-only mode ---
        if evaluate_only == True:
            if not background_model or not inpainted_file:
                raise ValueError("--evaluate-only requires --background_model and --inpainted-file")
            true_hist = Histogram.open(background_model)
            inpainted_hist = Histogram.open(inpainted_file)
            self.evaluate_inpainting_accuracy(true_hist, 
                    inpainted_hist, prefix=prefix, show_plots=show_plots)
        
            return

        # Check that training mode has needed input model:
        if training_mode in ["hybrid","supervised"] and background_model is None:
            raise ValueError("Hybrid and supervised modes require --background_model")

        # For in-line evaluation, check that model is passed:
        if evaluate and background_model is None:
            raise ValueError("--evaluate requires --background_model")

        # Load data
        input_data_proj, input_data_map = self.load_projected_data(input_data)
        _, psr_map = self.load_projected_psr(psr_file)

        # Optional background model
        model_map = None
        if background_model:
            model_proj, model_map = self.load_projected_model(background_model)

        # Derive nside
        npix = input_data_map.shape[-1]
        nside = hp.npix2nside(npix)

        # Mask from PSR
        mask_map = self.mask_from_cumdist_vectorized(psr_map, containment=containment)

        # Train & inpaint
        self.inpainted_maps = self.train_inpaint(input_data_map, mask_map, nside, 
            mode=training_mode, model_map=model_map,
            nn_model=nn_model, nn_model_file=nn_model_file, nn_model_savename=nn_model_savename,
            epochs=epochs, model_type=model_type,
            lr=lr, self_mask_fraction=self_mask_fraction, 
            lambda_sup=lambda_sup, lambda_self=lambda_self, verbose=verbose)

        # Save result
        self.save_inpainted_histpy(input_data_proj, self.inpainted_maps, output_file=f"{prefix}_estimated_bg.h5")

        # Visualization
        if visualize:
            self.visualize_and_save(input_data_map, mask_map, self.inpainted_maps,
                           em_bin=em_bin, phi_bin=phi_bin, prefix=prefix, show_plots=show_plots)

        # Inline evaluation
        if evaluate:
            inpainted_hist = Histogram(input_data_proj.axes, contents=self.inpainted_maps, sparse=True)
            self.evaluate_inpainting_accuracy(model_proj, inpainted_hist, prefix=prefix, show_plots=show_plots)

        end_time = time.time()
        logger.info(f"Total time elapsed: {end_time - start_time:.2f} seconds")

        return
