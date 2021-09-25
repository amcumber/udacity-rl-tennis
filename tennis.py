import matplotlib.pyplot as plt
import seaborn as sns

from reacher_agents.gym_environments import GymContinuousEnvMgr
from reacher_agents.unity_environments import UnityEnvMgr
from reacher_agents.trainers import SingleAgentTrainer, MultiAgentTrainer
from reacher_agents.config import *
from reacher_agents.ddpg_agent import DDPGAgent


def main():
    if ENV_TYPE.lower() == 'gym':
        envh = GymContinuousEnvMgr('Pendulum-v0')
        root_name = 'gym'
        Trainer = SingleAgentTrainer
        upper_bound = 2.0
        solved = -250
        env = envh.start()
        state_size = envh.state_size
        action_size = envh.action_size
    else:
        root_name = 'unity'
        if N_WORKERS == 1:
            file_name = 'envs/Reacher_Windows_x86_64-one-agent/Reacher.exe'
        else:
            file_name = 'envs/Reacher_Windows_x86_64-twenty-agents/Reacher.exe'
        envh = UnityEnvMgr(file_name)
        Trainer = MultiAgentTrainer
        upper_bound = 1.0
        solved = 30.0
        state_size = 33
        action_size = 4

    if CLOUD:
        if N_WORKERS==1:
            file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64'
        else:
            file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64'
        envh = UnityEnvMgr(file_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    agent = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        learn_f=LEARN_F,
        weight_decay=WEIGHT_DECAY,
        device=device,
        random_seed=42,
        upper_bound=upper_bound,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
        add_noise=ADD_NOISE
    )
    trainer = Trainer(
        agent=agent,
        env=envh,
        n_episodes=N_EPISODES,
        max_t=MAX_T,
        window_len=WINDOW_LEN,
        solved=solved,
        n_workers=N_WORKERS,
        max_workers=MAX_WORKERS,  # note can be lower than n
        save_root=root_name,
    )
    return envh, agent, trainer

def plot_scores(scores, i_map=0):
    sns.set_style('darkgrid')
    sns.set_context('talk')
    sns.set_palette('Paired')
    cmap = sns.color_palette('Paired')

    scores = scores.squeeze()
    score_df = pd.DataFrame({'scores': scores})
    score_df = score_df.assign(mean=lambda df: df.rolling(10).mean()['scores'])

    fig ,ax = plt.subplots(1,1, figsize=(10,8))

    ax = score_df.plot(ax=ax, color=cmap[2*(i_map%4):])
    ax.set_title(f'DDPG Scores vs Time (LR=({LR_ACTOR:.1e}, {LR_CRITIC:.1e}), Lf={LEARN_F})')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score')
    plt.show()

    
if __name__ == '__main__':
    _, _, trainer = main()
    scores = trainer.train(save_all=True, is_cloud=CLOUD)
    plot_scores(scores)