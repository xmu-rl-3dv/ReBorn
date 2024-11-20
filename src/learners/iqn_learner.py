import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmix import DMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.optim import Adam
import numpy
from modules.layers.act_layer import ActivateLayer
from torch.nn import Linear

class IQNLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmix":
                self.mixer = DMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())

            self.target_mixer = copy.deepcopy(self.mixer)

        self.neu_data = {}
        self.neu_avg = {}
        self.mask = {}

        for (name, module) in self.mac.agent.named_modules():  # time need to be added
            if isinstance(module, ActivateLayer):
                self.mask[module.name] = th.ones_like(module.weight, device=args.device,dtype=th.int)
                self.neu_data[module.name] = th.zeros_like(module.weight, device=args.device)
                def hook(module, fea_in, fea_out):
                    fea_out = fea_out.reshape(-1, args.n_agents, fea_out.shape[1]) # t(batch*agent, dim)  avg of each time
                    self.neu_data[module.name] = self.neu_data[module.name] + th.mean(fea_out, dim=0)
                    return None
                module.register_forward_hook(hook=hook)

        for (name, module) in self.mixer.named_modules():
            if isinstance(module, ActivateLayer):
                self.mask[module.name] = th.ones_like(module.weight, device=args.device, dtype=th.int)
                self.neu_data[module.name] = th.zeros_like(module.weight, device=args.device)
                def hook(module, fea_in, fea_out):
                    self.neu_data[module.name] = self.neu_data[module.name] + th.mean(fea_out, dim=0) # 1(batch, dim)
                    return None
                module.register_forward_hook(hook=hook)

        if args.optimizer == "RMSProp":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            raise ValueError("Unknown Optimizer")

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, check=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        episode_length = rewards.shape[1]
        assert rewards.shape == (batch.batch_size, episode_length, 1)
        actions = batch["actions"][:, :-1]
        assert actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        assert mask.shape == (batch.batch_size, episode_length, 1)
        avail_actions = batch["avail_actions"]
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)

        for key in self.neu_data.keys():
            self.neu_data[key] = th.zeros_like(self.neu_data[key], device=self.args.device)

        # Mix
        if self.mixer is not None:
            # Same quantile for quantile mixture
            n_quantile_groups = 1
        else:
            n_quantile_groups = self.args.n_agents

        # Calculate estimated Q-Values
        mac_out = []
        rnd_quantiles = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_rnd_quantiles = self.mac.forward(batch, t=t, forward_type="policy")
            assert agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_quantiles)
            assert agent_rnd_quantiles.shape == (batch.batch_size * n_quantile_groups, self.n_quantiles)
            agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
            rnd_quantiles.append(agent_rnd_quantiles)
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)
        del agent_outs
        del agent_rnd_quantiles
        mac_out = th.stack(mac_out, dim=1) # Concat over time
        rnd_quantiles = th.stack(rnd_quantiles, dim=1) # Concat over time
        assert mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length+1, n_quantile_groups, self.n_quantiles)
        rnd_quantiles = rnd_quantiles[:,:-1]
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)

        # Pick the Q-Values for the actions taken by each agent
        actions_for_quantiles = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
        del actions
        chosen_action_qvals = th.gather(mac_out[:,:-1], dim=3, index=actions_for_quantiles).squeeze(3)  # Remove the action dim
        del actions_for_quantiles
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_quantiles)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t, forward_type="target")
            assert target_agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_agent_outs = target_agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            assert target_agent_outs.shape == (batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_mac_out.append(target_agent_outs)
        del target_agent_outs
        del _

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        assert target_mac_out.shape == (batch.batch_size, episode_length, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)

        # Mask out unavailable actions
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)
        target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_mac_out[target_avail_actions[:,1:] == 0] = -9999999
        avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:,1:].mean(dim=4).max(dim=3, keepdim=True)[1]
            del mac_out_detach
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            del cur_max_actions
        else:
            # [0] is for max value; [1] is for argmax
            cur_max_actions = target_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1]
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            del cur_max_actions
        del target_mac_out
        assert target_max_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_target_quantiles)

        # Mix
        if self.mixer is not None:
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target=True)
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)
            assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
            assert target_max_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        # Calculate 1-step Q-Learning targets
        target_samples = rewards.unsqueeze(3) + \
            (self.args.gamma * (1 - terminated)).unsqueeze(3) * \
            target_max_qvals
        del target_max_qvals
        del rewards
        del terminated
        assert target_samples.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        # Quantile Huber loss
        target_samples = target_samples.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1)
        assert target_samples.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        chosen_action_qvals = chosen_action_qvals.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # u is the signed distance matrix
        u = target_samples.detach() - chosen_action_qvals
        del target_samples
        del chosen_action_qvals
        assert u.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert tau.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # The abs term in quantile huber loss
        abs_weight = th.abs(tau - u.le(0.).float())
        del tau
        assert abs_weight.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Huber loss
        loss = F.smooth_l1_loss(u, th.zeros(u.shape).cuda(self.args.device), reduction='none')
        del u
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Quantile Huber loss
        loss = (abs_weight * loss).mean(dim=4).sum(dim=3)
        del abs_weight
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups)

        assert mask.shape == (batch.batch_size, episode_length, 1)
        mask = mask.expand_as(loss)

        # 0-out the targets that came from padded data
        loss = loss * mask

        loss = loss.sum() / mask.sum()
        assert loss.shape == ()

        if check:
            self.find(t_env)
            self.optimiser.zero_grad()
            return

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def find(self, t_env):
        tau = 0.1
        beta = 3
        if hasattr(self.args, 'tau'):
            tau = self.args.tau
        if hasattr(self.args, 'beta'):
            beta = self.args.beta
        for name, item in self.neu_data.items():
            if len(item.shape) == 2:
                item = item.sum(dim=0)  # agent,dim / dim
            avg = item.mean()
            self.neu_avg[name] = avg
            self.mask[name][item <= tau * avg] = 0
            self.mask[name][item > tau * avg] = 1
            self.mask[name][item > beta * avg] = 2
            # self.spilt[name][self.mask[name]==0] = 1      # dead is not overload

            value = item.tolist()
            value = [round(num, 2) for num in value]

            count_01 = th.count_nonzero(item <= tau * avg).item()

            print("dead_neural_%s" % (name), count_01, item.shape[0], t_env)
            print("neural_value_%s" % (name), value)
            # self.logger.log_stat("dead_neural_%s%d" % (name, number), (count / sum), t_env)

    def recycle(self):
        layers = list(self.mixer.named_modules()) + list(self.mac.agent.named_modules())
        exc = 0
        for i in range(len(layers) - 2):
            act_layer = layers[i + 2][1]

            if isinstance(act_layer, ActivateLayer):
                input_name, input_layer = layers[i]
                output_name, output_layer = layers[i + 3]
                layer_mask = self.mask[act_layer.name]
                weight = input_layer.weight.data.T.clone()
                bias = input_layer.bias.data.clone()

                input_layer.reset_parameters()
                # avg_weight = (th.matmul(weight, layer_mask)/th.count_nonzero(layer_mask)).reshape(-1, 1)
                # avg_bias = (th.matmul(bias, layer_mask) / th.count_nonzero(layer_mask))
                # # avg_weight = th.mean(weight, dim=1).reshape(-1, 1)
                # # avg_bias = th.mean(bias)

                input_layer.weight.data = th.where(layer_mask != exc, weight, input_layer.weight.data.T).T
                input_layer.bias.data = th.where(layer_mask != exc, bias, input_layer.bias.data)

                if isinstance(output_layer, Linear):
                    output_weight = output_layer.weight.data.T
                    output_weight[layer_mask == exc] = 0
                    # output_weight[layer_mask == 2] = 0
                    output_layer.weight.data = output_weight.T

                layers[i] = (input_name, input_layer)
                layers[i + 3] = (output_name, output_layer)

    def reborn(self):
        layers = list(self.mixer.named_modules())

        for i in range(len(layers) - 2):
            act_layer = layers[i + 2][1]

            if isinstance(act_layer, ActivateLayer):

                input_name, input_layer = layers[i]
                output_name, output_layer = layers[i + 3]

                layer_mask = self.mask[act_layer.name]
                weight = input_layer.weight.data.T.clone()
                bias = input_layer.bias.data.clone()

                input_layer.reset_parameters()
                input_layer.weight.data = th.where(layer_mask != 0, weight, input_layer.weight.data.T).T
                input_layer.bias.data = th.where(layer_mask != 0, bias, input_layer.bias.data)

                weight = input_layer.weight.data.T.clone()
                bias = input_layer.bias.data.clone()

                dorm = th.where((layer_mask == 0))[0]
                dorm = dorm[th.randperm(dorm.size(0))]
                dec = th.rand(dorm.size(0), device=self.args.device) * 0.5 + 0.3
                over = th.where((layer_mask == 2))[0]
                over = over[th.randperm(over.shape[0])]
                #
                over_set = [[] for i in range(over.shape[0])]
                for dorm_i in range(dorm.shape[0]):
                    over_can = [over_i for over_i in range(over.shape[0]) if len(over_set[over_i]) < 5]
                    if len(over_can) == 0:
                        continue
                    over_select = th.randperm(len(over_can))[0].item()
                    # if over_select == len(over_can):
                    #     continue
                    over_set[over_can[over_select]].append(dorm_i)
                #
                for over_i in range(over.shape[0]):
                    for dorm_i in over_set[over_i]:
                        weight[:, dorm[dorm_i]] = weight[:, over[over_i]] * dec[dorm_i]
                        bias[dorm[dorm_i]] = bias[over[over_i]] * dec[dorm_i]

                input_layer.weight.data = weight.T
                input_layer.bias.data = bias

                # input_layer.reset_parameters()
                if isinstance(output_layer, Linear):
                    output_weight = output_layer.weight.data.T
                    output_weight[layer_mask == 0] = 0
                    for over_i in range(over.shape[0]):
                        emb = output_weight.shape[1]
                        div = th.softmax(th.randn(len(over_set[over_i]) + 1, emb, device=self.args.device), dim=0)
                        for k, dorm_i in enumerate(over_set[over_i]):
                            output_weight[dorm[dorm_i]] = output_weight[over[over_i]] * div[k] / dec[dorm_i]

                        output_weight[over[over_i]] = output_weight[over[over_i]] * div[len(over_set[over_i])]

                    output_layer.weight.data = output_weight.T

                layers[i] = (input_name, input_layer)
                layers[i + 3] = (output_name, output_layer)

    def reset(self):
        layers = list(self.mixer.named_modules()) + list(self.mac.agent.named_modules())
        for i in range(len(layers)):
            mlp_name, mlp_layer = layers[i]
            if isinstance(mlp_layer, Linear):
                if i == len(layers)-1:
                    mlp_layer.reset_parameters()
                    continue

                next_layer = layers[i+1][1]
                if not isinstance(next_layer, ActivateLayer) and not isinstance(next_layer, GRU):
                    mlp_layer.reset_parameters()


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda(self.args.device)
            self.target_mixer.cuda(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))