# Copyright : (c) KCL, Nicolas Skatchkovsky
# Licence : GPLv3

import numpy as np
import matplotlib.pyplot as plt

"""Assistant utility for automatically load network from network
description."""

import torch
from lava.lib.dl.slayer.utils.assistant import Assistant


class HybridAssistant(Assistant):
    """Assistant that bundles training, validation and testing workflow
        for DECOLLE models
    Parameters
    ----------
    net : torch.nn.Module
        network to train.
    error : object or lambda
        an error object or a lambda function that evaluates error.
        It is expected to take ``(output, target)`` | ``(output, label)``
        as it's argument and return a scalar value.
    optimizer : torch optimizer
        the learning optimizer.
    stats : slayer.utils.stats
        learning stats logger. If None, stats will not be logged.
        Defaults to None.
    classifier : slayer.classifier or lambda
        classifier object or lambda function that takes output and
        returns the network prediction. None means regression mode.
        Classification steps are bypassed.
        Defaults to None.
    count_log : bool
        flag to enable count log. Defaults to False.
    lam : float
        lagrangian to merge network layer based loss.
        None means no such additional loss.
        If not None, net is expected to return the accumulated loss as second
        argument. It is intended to be used with layer wise sparsity loss.
        Defaults to None.
   training_mode : str, one of "online" or "batch"
        perform gradient descent at every time-step ("online") as in the
        original paper, or after presentation of a batch of examples.
        Empirically, "online" is expected to perform better, but is slower.
    Attributes
    ----------
    net
    error
    optimizer
    stats
    classifier
    count_log
    lam
    device : torch.device or None
        the main device memory where network is placed. It is None at start and
        gets initialized on the first call.
    """
    def __init__(
            self,
            net, error, optimizer,
            stats=None, classifier=None, count_log=False,
            lam=None, training_mode='online'
    ):

        super(HybridAssistant, self).__init__(net, error, optimizer,
                                               stats, classifier,
                                               count_log, lam)
        if training_mode not in ['online', 'batch']:
            print("training_mode should be one of 'online' or 'batch'")
            raise ValueError
        self.training_mode = training_mode

    def stdp_weights(self, time_delays, a_plus, tau_plus, a_minus, tau_minus):
        ''' Calculates STDP weights given input hyperparameters and an input
            vector of time step delays. This is quite wasteful in terms of 
            memory but has the advantage of using vectorized operations,
            thus speeding up training.
            
            Parameters
            ---------
            time_delays : torch tensor
                input 1D tensor of time delays between neuron firing
            a_plus : float
                scaling factor for potentiation
            tau_plus : float
                decay constant for potentiation
            a_minus : float
                scaling factor for depression
            tau_minus : float
                decay constant for depression
                
            Returns
            ----------
            weights : torch tensor
                the weight updates corresponding to the input time delays'''
        
        # Finding indices where ltp should be performed 
        ltp_indices = torch.argwhere(time_delays >= 0)

        # Performing both ltp and ltd in parallel on all timepoints
        ltp = a_plus * torch.exp(time_delays/tau_plus)
        weights = a_minus * torch.exp(-time_delays/tau_minus)

        # Special case where there's only one weight to update and it's LTD
        # (needed because otherwise indexing fails)
        if ltp_indices.nelement() == 0:
            return weights
        # All other cases
        else:
            # Sorting out the ltp and ltd values we actually want
            weights[ltp_indices] = ltp[ltp_indices]

            return weights
    
    def plot_stdp_curve(self):
        ''' Plots the curve corresponding to the currently-used STDP rule, 
            subject to current hyperparameters'''
        
        delays = torch.arange(-25.0,25.0,1.0)
        weight_updates = self.stdp_weights(delays, 
                                        a_plus=self.net.stdp_learning_rate, 
                                        tau_plus=2*self.net.stdp_tau_combined, 
                                        a_minus=-self.net.stdp_learning_rate, 
                                        tau_minus=5*self.net.stdp_tau_combined)
        plt.plot(delays, weight_updates, lw=2.0)
        plt.axhline(0, color='k', alpha=0.6)
        plt.axvline(0, color='k', alpha=0.6)
        plt.xlim([-25, 25])
        plt.xlabel('$\Delta t$ (discrete timepoints)')
        plt.ylabel('$\Delta w$')

    def stdp(self, spikes, t, n_inputs, ignore_zeros):

        # Finding coordinates of spikes in tensors of size (batch_size, n_neurons).
        pre_spike_coords = torch.argwhere(
            spikes[-2].data.squeeze() == 1)
        post_spike_coords = torch.argwhere(
            spikes[-1].data.squeeze() == 1)

        # Updating when neurons last spiked, using coordinates above
        self.last_fired_presynaptic[pre_spike_coords[:,0], pre_spike_coords[:,1]] = t
        self.last_fired_postsynaptic[post_spike_coords[:,0], post_spike_coords[:,1]] = t

        # Updating the update cycle statuses.
        self.update_cycles[pre_spike_coords[:,0],pre_spike_coords[:,1],:] += 1 
        self.update_cycles[post_spike_coords[:,0],:,post_spike_coords[:,1]] += 1 

        # Checking if both pre- and post-synaptic neurons have fired. In the 
        # coordinates where the sum is 2, a weight update must be performed. 
        # Dimensions represent (weight_update, coord) where one coordinate represents
        # (sample, pre_neuron, post_neuron)
        update_coords = torch.argwhere(self.update_cycles == 2)

        # Time delays for each weight update
        time_delays = \
            self.last_fired_postsynaptic[
                update_coords[:,0], 
                update_coords[:,2]] - \
            self.last_fired_presynaptic[
                update_coords[:,0], 
                update_coords[:,1]]
        
        # Resetting update cycles for weights which we will now update
        self.update_cycles[
            update_coords[:,0],
            update_coords[:,1],
            update_coords[:,2]] = 0
        
        # If delays of zero are ignored, get rid of them and their corresponding
        # coordinates
        if ignore_zeros:
            update_coords = torch.squeeze(update_coords[torch.argwhere(time_delays != 0)])
            time_delays = torch.squeeze(time_delays[torch.argwhere(time_delays != 0)])

        # If there are (still) weights to update, perform them now
        if time_delays.nelement() > 0:
            # Calculating weight updates
            weight_updates = self.stdp_weights(
                time_delays, 
                self.net.stdp_learning_rate, 
                2*self.net.stdp_tau_combined, 
                -self.net.stdp_learning_rate, 
                5*self.net.stdp_tau_combined)

            if time_delays.nelement() > 1:
                # Putting all weight updates into a matrix of size (batch_size, n_post, n_pre)  
                # using coordinates for each weight
                weight_update_matrix = \
                    torch.zeros(
                        (n_inputs, 
                        self.last_fired_postsynaptic.shape[-1], 
                        self.last_fired_presynaptic.shape[-1])).to(self.device)
                weight_update_matrix[
                    update_coords[:,0], 
                    update_coords[:,2], 
                    update_coords[:,1]] = weight_updates.float()
                
                # Averaging over batch_size to get average weight updates for the batch
                weight_update_matrix = torch.mean(weight_update_matrix, axis=0)
                # Expanding dimensions to make this compatible with the weight matrix which
                # the block expects to receive
                weight_update_matrix = \
                    weight_update_matrix.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
                # Updating weights
                self.net.blocks[-1].synapse.weight.data += weight_update_matrix
    
            # Exception when there's only one weight to update (must be done
            # separately anyway because indexing above doesn't work in this case)
            elif time_delays.nelement() == 1:
                self.net.blocks[-1].synapse.weight.data[\
                    update_coords[2], update_coords[1]] += \
                    weight_updates/n_inputs
                    
    def train(self, input, target):
        """Training assistant.
        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.
        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled
        """
        self.net.train()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        input = input.to(device)
        target = target.to(device)

        if self.count_log:
            count = [0. for _ in range(len(self.net.blocks))]
        else:
            count = None

        # reset net + burnin
        input = self.net.init_state(input)

        if self.training_mode == 'online':
            readout = torch.Tensor().to(device)
            for t in range(input.shape[-1]):
                x = input[..., t].unsqueeze(-1)
                spikes, readouts_t, voltages, count_t = self.net(x)
                loss = self.error(readouts_t, voltages, target)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                if self.count_log:
                    count = [count[i] + count_t[i] / input.shape[-1]
                             for i in range(len(count_t))]
                if self.stats is not None:
                    self.stats.training.loss_sum \
                        += loss.cpu().data.item() * readouts_t[-1].cpu().shape[0]
                        
                # Absolute value of ORN-PN, PN-KC weights 
                for i in range(1, len(self.net.blocks)):
                    self.net.blocks[i].synapse.weight.data = \
                        torch.abs(self.net.blocks[i].synapse.weight.data)
        else:
            spikes, readouts, voltages, count = self.net(input)
            loss = self.error(readouts, voltages, target)
            loss.backward()

            readout = readouts[-1]
            if self.stats is not None:
                self.stats.training.loss_sum\
                    += loss.cpu().data.item() * readout.cpu().shape[0]
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.stats is not None:
            if len(target.shape) > 1:
                target_idx = target.argmax(-1)
            self.stats.training.num_samples += input.shape[0]
            if self.classifier is not None:   # classification
                self.stats.training.correct_samples += torch.sum(
                    self.classifier(readout) == target_idx
                ).cpu().data.item()

        if count is None:
            return readout

        return readout, count

    def train_hybrid(self, input, target, ignore_zeros):
        """Training assistant.
        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.
        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled
        """
        self.net.train()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        input = input.to(device)
        target = target.to(device)

        if self.count_log:
            count = [0. for _ in range(len(self.net.blocks))]
        else:
            count = None

        # reset net + burnin
        input = self.net.init_state(input)
        assert self. training_mode == 'online', \
            'Hybrid training only available in online mode'
        
        if self.training_mode == 'online':

            '''  ----------------- START CHECK ---------------------------'''
            # Timepoints when each of the neurons last spiked
            self.last_fired_presynaptic = torch.zeros(
                (input.shape[0], 
                self.net.layer_sizes[-2])).to(device)
            
            self.last_fired_postsynaptic = torch.zeros(
                (input.shape[0], 
                self.net.layer_sizes[-1])).to(device)

            # Matrix used to keep track of if both pre- and postsynaptic neurons 
            # of a particular weight have fired within the current update cycle, 
            # thus indicating that a weight update should be performed. First 
            # batch_size slices along 0th dimension are for presynaptic neurons,
            # the rest for postsynaptic ones. 
            self.update_cycles = torch.zeros(
                (input.shape[0], 
                self.last_fired_presynaptic.shape[-1], 
                self.last_fired_postsynaptic.shape[-1])).to(device)
            ''' ---------------------- END CHECK ---------------------------'''


            readout = torch.Tensor().to(device)
            for t in range(input.shape[-1]):
                x = input[..., t].unsqueeze(-1)
                spikes, readouts_t, voltages, count_t = self.net(x)

                loss = self.error(readouts_t, voltages, target)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                if self.count_log:
                    count = [count[i] + count_t[i] / input.shape[-1]
                             for i in range(len(count_t))]
                if self.stats is not None:
                    self.stats.training.loss_sum \
                        += loss.cpu().data.item() * readouts_t[-1].cpu().shape[0]
                

                ''' ------------------- START CHECK --------------------------'''
                # STDP updates 
                self.stdp(spikes, t, input.shape[0], ignore_zeros)
                # Absolute value of ORN-PN weights 
                self.net.blocks[-2].synapse.weight.data = \
                        torch.abs(self.net.blocks[-2].synapse.weight.data)
                ''' ---------------------- END CHECK ------------------------'''

        else:
            spikes, readouts, voltages, count = self.net(input)
            loss = self.error(readouts, voltages, target)
            loss.backward()

            readout = readouts[-1]
            if self.stats is not None:
                self.stats.training.loss_sum\
                    += loss.cpu().data.item() * readout.cpu().shape[0]
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.stats is not None:
            if len(target.shape) > 1:
                target_idx = target.argmax(-1)
            self.stats.training.num_samples += input.shape[0]
            if self.classifier is not None:   # classification
                self.stats.training.correct_samples += torch.sum(
                    self.classifier(readout) == target_idx
                ).cpu().data.item()

        if count is None:
            return readout

        return readout, count
    
    def test(self, input, target):
        """Testing assistant.
        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.
        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled
        """
        self.net.eval()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)

            if self.count_log:
                count = [0. for _ in range(len(self.net.blocks))]
            else:
                count = None

            # reset net + burnin
            input = self.net.init_state(input)

            if self.training_mode == 'online':
                readout = torch.Tensor().to(device)
                for t in range(input.shape[-1]):
                    x = input[..., t].unsqueeze(-1)
                    spikes, readouts_t, voltages, count_t = self.net(x)
                    loss = self.error(readouts_t, voltages, target)

                    readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                    if self.count_log:
                        count = [count[i] + count_t[i] / input.shape[-1]
                                 for i in range(len(count_t))]
                    if self.stats is not None:
                        self.stats.testing.loss_sum \
                            += loss.cpu().data.item() * readouts_t[-1].shape[0]
            else:
                spikes, readouts, voltages, count = self.net(input)
                loss = self.error(readouts, voltages, target)

                readout = readouts[-1]
                if self.stats is not None:
                    self.stats.testing.loss_sum \
                        += loss.cpu().data.item() * readout.cpu().shape[0]

            if self.stats is not None:
                if len(target.shape) > 1:
                    target_idx = target.argmax(-1)
                self.stats.testing.num_samples += input.shape[0]
                self.stats.testing.loss_sum \
                    += loss.cpu().data.item() * readout.cpu().shape[0]
                if self.classifier is not None:   # classification
                    self.stats.testing.correct_samples += torch.sum(
                        self.classifier(readout) == target_idx
                    ).cpu().data.item()

            if count is None:
                return readout

            return readout, count

    def valid(self, input, target):
        """Validation assistant.
        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.
        Returns
        -------
        output
            network's last readout layer output.
        count : optional
            spike count if ``count_log`` is enabled
        """
        self.net.eval()

        with torch.no_grad():
            device = self.net.device
            input = input.to(device)
            target = target.to(device)

            if self.count_log:
                count = [0. for _ in range(len(self.net.blocks))]
            else:
                count = None

            # reset net + burnin
            input = self.net.init_state(input)

            if self.training_mode == 'online':
                readout = torch.Tensor()
                for t in range(input.shape[-1]):
                    x = input[..., t].unsqueeze(-1)
                    spikes, readouts_t, voltages, count_t = self.net(x)
                    loss = self.error(readouts_t, voltages, target)

                    readout = torch.cat((readout, readouts_t[-1]), dim=-1)
                    if self.count_log:
                        count = [count[i] + count_t[i] / input.shape[-1]
                                 for i in range(len(count_t))]
                    if self.stats is not None:
                        self.stats.validation.loss_sum \
                            += loss.cpu().data.item() * readouts_t[-1].shape[0]
            else:
                spikes, readouts, voltages, count = self.net(input)
                loss = self.error(readouts, voltages, target)

                readout = readouts[-1]
                if self.stats is not None:
                    self.stats.validation.loss_sum \
                        += loss.cpu().data.item() * readout[-1].shape[0]

            if self.stats is not None:
                self.stats.validation.num_samples += input.shape[0]
                if self.lam is None:
                    self.stats.validation.loss_sum \
                        += loss.cpu().data.item() * readout.shape[0]
                else:
                    self.stats.validation.loss_sum \
                        += loss.cpu().data.item() * readout.shape[0]
                if self.classifier is not None:   # classification
                    if len(target.shape) > 1:
                        target_idx = target.argmax(-1)
                    self.stats.validation.correct_samples += torch.sum(
                        self.classifier(readout) == target_idx
                    ).cpu().data.item()

            if count is None:
                return readout

            return readout, count