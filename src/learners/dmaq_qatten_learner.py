# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np
import numpy
from utils.rl_utils import build_td_lambda_targets
from modules.layers.act_layer import ActivateLayer
from torch.nn import Linear,GRU

class DMAQ_qattenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
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

        para = sum([np.prod(list(p.size())) for p in self.mixer.parameters()])
        print(para * 4 / 1000 / 1000)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None, check=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        for key in self.neu_data.keys():
            self.neu_data[key] = th.zeros_like(self.neu_data[key], device=self.args.device)
        # Calculate estimated Q-Values
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda(self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            raise "Use Double Q"

        # Mix
        if mixer is not None:
            ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
            ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                            max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                target_chosen = self.target_mixer(target_chosen_qvals, batch["state"], is_v=True)
                target_adv = self.target_mixer(target_chosen_qvals, batch["state"],
                                                actions=cur_max_actions_onehot,
                                                max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
            else:
                raise "Use Double Q"

        # Calculate 1-step Q-Learning targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)


        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        if check:
            self.find(t_env)
            self.optimiser.zero_grad()
            return

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, check=False):
        self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                       show_demo=show_demo, save_data=save_data, check=check)
        if check:
            return
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

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
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


