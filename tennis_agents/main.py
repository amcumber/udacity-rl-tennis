import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict
import toml
from pathlib import Path
import numpy as np

from tennis_agents.unity_environments import UnityEnvMgr
from tennis_agents.trainers import TennisTrainer
from tennis_agents.ddpg_agent import DDPGAgent


def read_toml(file: str) -> Dict[str, Any]:
    assert Path(file).exists()
    with Path(file).open('r') as fh:
        data = toml.load(fh)
    return data


def configure(data: dict[str, Any]):
    # Unpack config
    # # Environment
    ROOT_NAME     = data['environment']['ROOT_NAME']
    ENV_FILE      = data['environment']['ENV_FILE']
    UPPER_BOUND   = data['environment']['UPPER_BOUND']
    STATE_SIZE    = data['environment']['STATE_SIZE']
    ACTION_SIZE   = data['environment']['ACTION_SIZE']

    # # Trainer
    Trainer       = TennisTrainer
    BATCH_SIZE    = data['trainer']['BATCH_SIZE']
    N_EPISODES    = data['trainer']['N_EPISODES']
    MAX_T         = data['trainer']['MAX_T']
    WINDOW_LEN    = data['trainer']['WINDOW_LEN']
    SAVE_ALL      = data['trainer']['SAVE_ALL']

    # # Agent
    N_AGENTS      = data['agent']['N_AGENTS']
    BUFFER_SIZE   = data['agent']['BUFFER_SIZE']
    LEARN_F       = data['agent']['LEARN_F']
    GAMMA         = data['agent']['GAMMA']
    TAU           = data['agent']['TAU']
    LR_ACTOR      = data['agent']['LR_ACTOR']
    LR_CRITIC     = data['agent']['LR_CRITIC']
    WEIGHT_DECAY  = data['agent']['WEIGHT_DECAY']
    ACTOR_HIDDEN  = data['agent']['ACTOR_HIDDEN']
    CRITIC_HIDDEN = data['agent']['CRITIC_HIDDEN']
    ADD_NOISE     = data['agent']['ADD_NOISE']

    envh = UnityEnvMgr(ENV_FILE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.device("cpu")
    agents = [DDPGAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
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
        upper_bound=UPPER_BOUND,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
        add_noise=ADD_NOISE[idx],
    ) for idx in range(N_AGENTS)]
    trainer = Trainer(
        agents=agents,
        env=envh,
        n_episodes=N_EPISODES,
        max_t=MAX_T,
        window_len=WINDOW_LEN,
        solved=solved,
        save_root=ROOT_NAME,
    )
    return envh, trainer

def plot_scores(scores, i_map=0):
    sns.set_style('darkgrid')
    sns.set_context('talk')
    sns.set_palette('Paired')
    cmap = sns.color_palette('Paired')

    scores = np.array(scores).squeeze()
    score_df = pd.DataFrame({'scores': scores})
    score_df = score_df.assign(mean=lambda df: df.rolling(10).mean()['scores'])

    _ ,ax = plt.subplots(1,1, figsize=(10,8))

    ax = score_df.plot(ax=ax, color=cmap[2*(i_map%4):])
    ax.set_title(f'DDPG Scores vs Time (LR=({LR_ACTOR:.1e}, ' +
                 f'{LR_CRITIC:.1e}), Lf={LEARN_F})')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score')
    plt.show()


def main(file:str) -> None:
    """
    Main Function that will load configuration from file, train, and plot the
    resulting scores
    """
    data = read_toml(file)
    _, trainer = configure(data)
    scores = trainer.train()
    plot_scores(scores)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Run Tennis from terminal')
    p.add_argument('config', help='Configuration File - toml')
    args = p.parse_args()

    main(args.config)

