import logging

import numpy as np

import torch
import torch.optim as optim

from timeit import default_timer as timer

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.apis.fl_context import FLContext
from torch.utils.tensorboard import SummaryWriter
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.signal import Signal

from nvflare.fuel.utils import fobs
from nvflare.fuel.f3.stats_pool import StatsPoolManager

from src.vfl.split_nn_dataset import SplitNNDataset


logging.getLogger().setLevel(logging.DEBUG)


class SplitNNDataKind:
    GRADIENT = "_splitnn_gradient_"
    ACTIVATIONS = "_splitnn_activations_"


class SplitNNConstants:
    BATCH_INDICES = "_splitnn_batch_indices_"
    DATA = "_splitnn_data_"
    BATCH_SIZE = "_splitnn_batch_size_"
    TARGET_NAMES = "_splitnn_target_names_"
    
    TASK_INIT_MODEL = "_splitnn_task_init_model_"
    TASK_TRAIN_SCB_STEP = "_splitnn_task_train_scb_step_"
    TASK_TRAIN_CARDX_STEP = "_splitnn_task_train_cardx_step_"
    TASK_TRAIN = "_splitnn_task_train_"
    TASK_VALID_LABEL_STEP = "_splitnn_task_val_scb_step_"
    
    TIMEOUT = 30.0


class SplitNNLearner(Learner):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        intersection_file: str = None,
        lr: float = 1e-2,
        model: dict = None,
        analytic_sender_id: str = "analytic_sender",
        fp16: bool = True,
        val_freq: int = 20,
    ):
        """Simple CIFAR-10 Trainer for split learning.

        Args:
            dataset_root: directory with CIFAR-10 data.
            intersection_file: Optional. intersection file specifying overlapping indices between both clients.
                Defaults to `None`, i.e. the whole training dataset is used.
            lr: learning rate.
            model: Split learning model.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            fp16: If `True`, convert activations and gradients send between clients to `torch.float16`.
                Reduces bandwidth needed for communication but might impact model accuracy.
            val_freq: how often to perform validation in rounds. Defaults to 1000. No validation if <= 0.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.intersection_file = intersection_file
        self.lr = lr
        self.model = model
        self.analytic_sender_id = analytic_sender_id
        self.fp16 = fp16
        self.val_freq = val_freq
        
        self.writer = None
        
        self.val_loss = []
        self.val_labels = []
        self.val_pred_labels = []
        
        fobs.register(TensorDecomposer)
        
    def _get_model(self, fl_ctx: FLContext):
        """Get model from client config. Modelled after `PTFileModelPersistor`."""
        if isinstance(self.model, str):
            # treat it as model component ID
            if not self.model:
                self.log_error(fl_ctx, f"cannot find model component '{model_component_id}'")
                return
            
            model_component_id = self.model
            engine = fl_ctx.get_engine()
            self.model = engine.get_component(model_component_id)
            
        if self.model and isinstance(self.model, dict):
            # try building the model
            try:
                engine = fl_ctx.get_engine()
                # use provided or default optimizer arguments and add the model parameters
                if "args" not in self.model:
                    self.model["args"] = {}
                self.model = engine.build_component(self.model)
            except Exception as e:
                self.system_panic(
                    f"Exception while parsing `model`: " f"{self.model} with Exception {e}",
                    fl_ctx,
                )
                return
        if self.model and not isinstance(self.model, torch.nn.Module):
            self.system_panic(f"expect model to be torch.nn.Module but got {type(self.model)}: {self.model}", fl_ctx)
            return
        if self.model is None:
            self.system_panic(f"Model wasn't built correctly! It is {self.model}", fl_ctx)
            return
        self.log_info(fl_ctx, f"Running model {self.model}")
        
    def initialize(self, parts: dict, fl_ctx: FLContext):
        t_start = timer()
        self._get_model(fl_ctx=fl_ctx)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.BCELoss()
        
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.client_name = fl_ctx.get_identity_name()
        self.split_id = self.model.get_split_id()
        self.log_info(fl_ctx, f"Running `split_id` {self.split_id} on site `{self.client_name}`")
        
        # if self.intersection_file is not None:
        #     _intersect_indices = np.loadtxt(self.intersection_file)
        # else:
        #     _intersect_indices = None
        
        self.train_dataset = SplitNNDataset(
            root=self.dataset_root,
            intersect_idx=list(range(20_000)),  # TODO: 
            split_id=self.split_id
        )
        
        self.valid_dataset = SplitNNDataset(
            root=self.dataset_root,
            intersect_idx=list(range(20_000, 30_000, 1)),  # TODO:
            split_id=self.split_id
        )

        self.train_size = len(self.train_dataset)
        if self.train_size <= 0:
            raise ValueError(f"Expected train dataset size to be larger zero but got {self.train_size}")
        self.log_info(fl_ctx, f"Training with {self.train_size}")
        
        if self.split_id == 0:  # metrics can only be computed for client with labels
            self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
            if not self.writer:  # use local TensorBoard writer only
                self.writer = SummaryWriter(self.app_root)
        
        # register aux message handlers
        engine = fl_ctx.get_engine()
        
        if self.split_id == 0:
            engine.register_aux_message_handler(
                topic=SplitNNConstants.TASK_TRAIN_SCB_STEP, message_handle_func=self._aux_train_scb
            )
            engine.register_aux_message_handler(
                topic=SplitNNConstants.TASK_VALID_LABEL_STEP, message_handle_func=self._aux_val_scb
            )
            self.log_info(fl_ctx, f"Registered aux message handlers for split_id {self.split_id}")
        
        self.compute_stats_pool = StatsPoolManager.add_time_hist_pool(
            "Compute_Time", "Compute time in secs", scope=self.client_name
        )

        self.compute_stats_pool.record_value(category="initialize", value=timer() - t_start)
        
    def _forward_cardx(self, batch_indices):
        t_start = timer()
        self.model.train()
        
        inputs: torch.Tensor = self.train_dataset.get_batch(batch_indices)
        inputs = inputs.to(self.device)
        
        self.cardx_activations: torch.Tensor = self.model.forward(inputs)
        
        self.compute_stats_pool.record_value(category="_forward_cardx", value=timer() - t_start)
        
        return self.cardx_activations.detach().requires_grad_().flatten(start_dim=1, end_dim=-1)
    
    def _forward_backward_scb(self, batch_indices, activations, fl_ctx: FLContext):
        t_start = timer()
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs, targets = self.train_dataset.get_batch(batch_indices)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        activations = activations.to(self.device)
        activations.requires_grad_(True)
        
        pred = self.model.forward(inputs, activations)
        loss = self.criterion(pred, targets)
        loss.backward()
        
        pred_labels = (pred >= 0.5).to(torch.float32)
        acc = (pred_labels == targets).sum() / len(targets)
        
        self.optimizer.step()
        
        self.compute_stats_pool.record_value(category="_forward_backward_scb", value=timer() - t_start)
        
        # self.log_debug(fl_ctx, f"pred: {pred}, grad: {activations.grad}, target: {targets}")
        self.log_debug(fl_ctx, f"Training loss: {loss}")
        
        if self.writer:
            self.writer.add_scalar("training_loss", loss.item(), self.current_round)
            self.writer.add_scalar("training_accuracy", acc.item(), self.current_round)
        
        return activations.grad
    
    def _backward_cardx(self, gradient, fl_ctx: FLContext):
        t_start = timer()
        self.model.train()
        self.optimizer.zero_grad()
        
        gradient = gradient.to(self.device)
        self.cardx_activations.backward(gradient=gradient.reshape(self.cardx_activations.shape))
        self.optimizer.step()
        
        self.log_debug(
            fl_ctx, f"{self.client_name} runs model with `split_id` {self.split_id} for backward step on data side."
        )
        self.compute_stats_pool.record_value(category="_backward_step_data_side", value=timer() - t_start)
        
        return make_reply(ReturnCode.OK)
    
    def _aux_train_scb(self, topic: str, request: Shareable, fl_ctx: FLContext):
        t_start = timer()
        
        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Train label in round {self.current_round} of {self.num_rounds} rounds.")
        
        dxo = from_shareable(request)
        if dxo.data_kind != SplitNNDataKind.ACTIVATIONS:
            raise ValueError(f"Expected data kind {SplitNNDataKind.ACTIVATIONS} but received {dxo.data_kind}")

        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")

        activations = dxo.data.get(SplitNNConstants.DATA)
        if activations is None:
            raise ValueError("No activations in DXO!")

        gradient = self._forward_backward_scb(
            batch_indices=batch_indices, activations=fobs.loads(activations), fl_ctx=fl_ctx
        )

        self.log_debug(fl_ctx, "_aux_train_scb finished.")
        return_shareable = DXO(
            data={SplitNNConstants.DATA: fobs.dumps(gradient)}, data_kind=SplitNNDataKind.GRADIENT
        ).to_shareable()

        self.compute_stats_pool.record_value(category="_aux_train_scb", value=timer() - t_start)

        self.log_debug(fl_ctx, f"Sending train label return_shareable: {type(return_shareable)}")
        return return_shareable
    
    def _val_forward_cardx(self, batch_indices):
        t_start = timer()
        self.model.eval()
        
        inputs: torch.Tensor = self.valid_dataset.get_batch(batch_indices)
        inputs = inputs.to(self.device)
        
        _val_activations: torch.Tensor = self.model.forward(inputs)
        
        self.compute_stats_pool.record_value(category="_val_forward_cardx", value=timer() - t_start)
        
        return _val_activations.detach().flatten(start_dim=1, end_dim=-1)
    
    def _val_forward_scb(self, batch_indices, activations, fl_ctx: FLContext):
        t_start = timer()
        self.model.eval()
        
        inputs, targets = self.valid_dataset.get_batch(batch_indices)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        activations = activations.to(self.device)
        
        pred = self.model.forward(inputs, activations)
        loss = self.criterion(pred, targets)
        
        self.compute_stats_pool.record_value(category="_val_forward_scb", value=timer() - t_start)
        
        self.val_loss.append(loss.unsqueeze(0))  # unsqueeze needed for later concatenation

        pred_labels = (pred >= 0.5).to(torch.float32)

        self.val_pred_labels.extend(pred_labels.unsqueeze(0))
        self.val_labels.extend(targets.unsqueeze(0))
    
    def _aux_val_scb(self, topic: str, request: Shareable, fl_ctx: FLContext):
        t_start = timer()
        
        val_round = request.get_header(AppConstants.CURRENT_ROUND)
        val_num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Validate label in round {self.current_round} of {self.num_rounds} rounds.")
        
        dxo = from_shareable(request)
        if dxo.data_kind != SplitNNDataKind.ACTIVATIONS:
            raise ValueError(f"Expected data kind {SplitNNDataKind.ACTIVATIONS} but received {dxo.data_kind}")

        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")
        
        activations = dxo.data.get(SplitNNConstants.DATA)
        if activations is None:
            raise ValueError("No activations in DXO!")

        self._val_forward_scb(
            batch_indices=batch_indices, activations=fobs.loads(activations), fl_ctx=fl_ctx
        )
        
        if val_round == val_num_rounds - 1:
            self._log_validation(fl_ctx)
        
        self.compute_stats_pool.record_value(category="_aux_val_scb", value=timer() - t_start)
        
        return make_reply(ReturnCode.OK)
    
    # Model initialization task (one time only in beginning)
    def init_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        t_start = timer()
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed.") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError("No global weights loaded!")

        self.compute_stats_pool.record_value(category="init_model", value=timer() - t_start)

        self.log_info(fl_ctx, "init_model finished.")

        return make_reply(ReturnCode.OK)
    
    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        t_start = timer()
        """main training logic"""
        engine = fl_ctx.get_engine()

        self.num_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        if not self.num_rounds:
            raise ValueError("No number of rounds available.")
        self.batch_size = shareable.get_header(SplitNNConstants.BATCH_SIZE)
        self.target_names = np.asarray(
            shareable.get_header(SplitNNConstants.TARGET_NAMES)
        )  # convert to array for string matching below
        self.other_client = self.target_names[self.target_names != self.client_name][0]
        self.log_info(fl_ctx, f"Starting training of {self.num_rounds} rounds with batch size {self.batch_size}")

        gradients = None  # initial gradients
        for _curr_round in range(self.num_rounds):
            self.current_round = _curr_round
            if self.split_id == 0:
                continue
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.log_debug(fl_ctx, f"Starting current round={self.current_round} of {self.num_rounds}.")
            self.train_batch_indices = np.random.randint(0, self.train_size - 1, self.batch_size)

            # Site-1 image forward & backward (from 2nd round)
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)
            if gradients is not None:
                self._backward_cardx(fobs.loads(gradients), fl_ctx)
            
            activations = self._forward_cardx(self.train_batch_indices)

            # Site-2 label loss & backward
            dxo = DXO(data={SplitNNConstants.DATA: fobs.dumps(activations)}, data_kind=SplitNNDataKind.ACTIVATIONS)
            dxo.set_meta_prop(SplitNNConstants.BATCH_INDICES, self.train_batch_indices)

            data_shareable = dxo.to_shareable()
            data_shareable.set_header(AppConstants.CURRENT_ROUND, self.current_round)
            data_shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
            data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

            # send to other side
            result = engine.send_aux_request(
                targets=self.other_client,
                topic=SplitNNConstants.TASK_TRAIN_SCB_STEP,
                request=data_shareable,
                timeout=SplitNNConstants.TIMEOUT,
                fl_ctx=fl_ctx,
            )
            shareable = result.get(self.other_client)
            if shareable is not None:
                dxo = from_shareable(shareable)
                if dxo.data_kind != SplitNNDataKind.GRADIENT:
                    raise ValueError(f"Expected data kind {SplitNNDataKind.GRADIENT} but received {dxo.data_kind}")
                gradients = dxo.data.get(SplitNNConstants.DATA)
            else:
                raise ValueError(f"No message returned from {self.other_client}!")

            self.log_debug(fl_ctx, f"Ending current round={self.current_round}.")
            
            if self.val_freq > 0:
                if _curr_round % self.val_freq == 0:
                    self._validate(fl_ctx)

        self.compute_stats_pool.record_value(category="train", value=timer() - t_start)

        return make_reply(ReturnCode.OK)
    
    def _log_validation(self, fl_ctx: FLContext):
        if len(self.val_loss) > 0:
            loss = torch.mean(torch.cat(self.val_loss))

            _val_pred_labels = torch.cat(self.val_pred_labels)
            _val_labels = torch.cat(self.val_labels)
            acc = (_val_pred_labels == _val_labels).sum() / len(_val_labels)

            self.log_info(
                fl_ctx,
                f"Round {self.current_round}/{self.num_rounds} val_loss: {loss.item():.4f}, val_accuracy: {acc.item():.4f}",
            )
            if self.writer:
                self.writer.add_scalar("val_loss", loss.item(), self.current_round)
                self.writer.add_scalar("val_accuracy", acc.item(), self.current_round)

            self.val_loss = []
            self.val_labels = []
            self.val_pred_labels = []
    
    def _validate(self, fl_ctx: FLContext):
        t_start = timer()
        engine = fl_ctx.get_engine()

        idx = np.arange(len(self.valid_dataset))
        n_batches = int(np.ceil(len(self.valid_dataset) / self.batch_size))
        for _val_round, _val_batch_indices in enumerate(np.array_split(idx, n_batches)):
            activations = self._val_forward_cardx(batch_indices=_val_batch_indices)

            # Site-2 label loss & accuracy
            dxo = DXO(data={SplitNNConstants.DATA: fobs.dumps(activations)}, data_kind=SplitNNDataKind.ACTIVATIONS)
            dxo.set_meta_prop(SplitNNConstants.BATCH_INDICES, _val_batch_indices)

            data_shareable = dxo.to_shareable()
            data_shareable.set_header(AppConstants.CURRENT_ROUND, _val_round)
            data_shareable.set_header(AppConstants.NUM_ROUNDS, n_batches)
            data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, _val_round)

            # send to other side to validate
            engine.send_aux_request(
                targets=self.other_client,
                topic=SplitNNConstants.TASK_VALID_LABEL_STEP,
                request=data_shareable,
                timeout=SplitNNConstants.TIMEOUT,
                fl_ctx=fl_ctx,
            )

        self.compute_stats_pool.record_value(category="_validate", value=timer() - t_start)

        self.log_debug(fl_ctx, "finished validation.")
