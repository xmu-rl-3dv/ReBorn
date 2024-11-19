from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from utils.risk import get_risk_q, get_risk_q_mode,get_current_risk_param

# This multi-agent controller shares parameters between agents
class RiskMAC:
    def __init__(self, scheme, groups, args):
        # self.cumulative_density = th.tensor(
            # (2 * np.arange(self.args.n_target_quantiles) + 1) / (2.0 * self.args.n_target_quantiles), dtype=th.float32).view(1, -1).to(
            # self.args.device)
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.rnd01 = th.ones(1, self.args.n_target_quantiles).cuda() * (1/self.args.n_target_quantiles)

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.r_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        rnd01 = None
        if self.args.agent in ["iqn_rnn", "risk_rnn", "risk_rnn_v2", "risk_rnn_v3", "risk_rnn_v4", "risk_rnn_v5",
                               "spl_rnn", "splv2_rnn", "ncqrdqn_rnn", "qrdqn_rnn"]:
            agent_outputs, rnd01 = self.forward(ep_batch, t_ep, forward_type="approx")
        else:
            agent_outputs = self.forward(ep_batch, t_ep, forward_type=test_mode)
        if self.args.agent == "iqn_rnn":  
            agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3)
        if self.args.agent in ["risk_rnn", "risk_rnn_v2", "risk_rnn_v3", "risk_rnn_v4", "risk_rnn_v5",
                               "spl_rnn", "splv2_rnn", "ncqrdqn_rnn", "qrdqn_rnn"]:
            risk_type = self.args.risk_type
            risk_param = self.args.risk_param
            if rnd01 is None:
                rnd01 = self.rnd01
             #这个时候rnd01的shape是[1, n_quantile], agent_output的shape是[n_agents, n_actions, n_quantiles]
            ain = agent_outputs[bs]
            if ain.shape[0] == 1:
                ain = ain.squeeze(0)
            # if hasattr(self.args, "no_rigm_mode"):
            #     riskq_value, tau_hat = get_risk_q_mode(risk_type, risk_param, rnd01, ain, self.args.no_rigm_mode)
            # else:
            #     riskq_value, tau_hat = get_risk_q(risk_type, risk_param, rnd01, ain) #返回的是(batch_size, n_agents, n_actions, 1)
            if isinstance(risk_type, list):
                if hasattr(self.args, "num_notrigm_agents"):
                    agent_num = self.args.num_notrigm_agents
                    riskq_value_1, tau_hat_1 = get_risk_q(risk_type[0], risk_param[0], rnd01,
                                                          ain[:-agent_num])
                    riskq_value_2, tau_hat_2 = get_risk_q(risk_type[1], risk_param[1], rnd01,
                                                          ain[-agent_num:])
                else:
                    riskq_value_1, tau_hat_1 = get_risk_q(risk_type[0], risk_param[0], rnd01, ain[:-1]) # 将第一个智能体到倒数第二个个智能体采取risk_1以及risk_param_1的操作
                    riskq_value_2, tau_hat_2 = get_risk_q(risk_type[1], risk_param[1], rnd01, ain[-1].unsqueeze(0)) # 计算最后一个智能体的采取risk_2以及risk_param_2的操作

                riskq_value, tau_hat = th.cat((riskq_value_1, riskq_value_2) , dim = 0), th.cat((tau_hat_1, tau_hat_2) , dim = 0)
            else:
                rebuttal = getattr(self.args, "rebuttal", "nothing")
                if rebuttal == "nothing":
                    riskq_value, tau_hat = get_risk_q(risk_type, risk_param, rnd01, ain)
                elif rebuttal == "simple":
                    # "这里要不就直接在joint function那边用risk-sensitive，而在每个小智能体的动作选择的时候只是用 expectation的方式？
                    # 具体来说就是在mac_controller那边的智能体执行用expectation的方式（就是强制输入cvar 1），而在learner那边保持不变。"
                    riskq_value, tau_hat = get_risk_q("cvar", 1, rnd01, ain)
                elif rebuttal == "LQNdynamicrisk":
                    stop_t = getattr(self.args, "stop_riskparam_t", 1000000)
                    start_risk_param = getattr(self.args, "start_riskparam", -0.75)
                    target_risk_param = getattr(self.args, "target_riskparam", 0.75)
                    dynamic_risk_param = get_current_risk_param(t_env, stop_t, start_risk_param, target_risk_param)
                    riskq_value, tau_hat = get_risk_q(risk_type, dynamic_risk_param, rnd01, ain)



            riskq_value = riskq_value.squeeze(-1).unsqueeze(0)
            # print("riskq_value.shape", riskq_value.shape, "avail_actions.shape", avail_actions[bs].shape)
            chosen_actions = self.action_selector.select_action(riskq_value, avail_actions[bs], t_env,
                                                                test_mode=test_mode)
            del tau_hat
            del riskq_value
            del rnd01
        else:
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        del agent_outputs
        return chosen_actions

    def forward(self, ep_batch, t, forward_type=None, rnd_value01=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.agent in ["iqn_rnn", "risk_rnn", "risk_rnn_v2", "risk_rnn_v3", "risk_rnn_v5",
                               "spl_rnn", "splv2_rnn", "ncqrdqn_rnn"]:
            agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type, rnd_value01=rnd_value01)
        elif self.args.agent in ["risk_rnn_v4"]:
            risk_agent_inputs = self._build_inputs(ep_batch, 0 if t == 0 else t - 1)
            agent_outs, self.hidden_states, self.r_hidden_states, rnd_quantiles = self.agent(agent_inputs, risk_agent_inputs, self.hidden_states, self.r_hidden_states, forward_type=forward_type, rnd_value01=rnd_value01)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        del agent_inputs
        if self.args.agent in ["iqn_rnn", "risk_rnn", "risk_rnn_v2", "risk_rnn_v3", "risk_rnn_v4",
                               "risk_rnn_v5", "spl_rnn", "splv2_rnn", "ncqrdqn_rnn"]:
            return agent_outs, rnd_quantiles
        elif self.args.agent == "qrdqn_rnn":
            return agent_outs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, self.args.n_target_quantiles), None
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)  # [batch_size, n_agent, n_action]

    def init_hidden(self, batch_size):
        if self.args.agent in ["risk_rnn_v4"]:
            hidden_states, r_hidden_states = self.agent.init_hidden()
            hidden_states = hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)
            if r_hidden_states is not None:
                r_hidden_states = r_hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)

            self.hidden_states, self.r_hidden_states = hidden_states, r_hidden_states  # bav
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
