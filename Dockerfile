# # Modified from https://github.com/salvioli/deep-rl-tennis
# # Build failing - error with TF + Torch Mismatch
# FROM pytorch/pytorch
# LABEL author="aaron.mcumber@gmail.com"

# WORKDIR /deep-rl-tennis/

# RUN apt-get update && \
#     apt-get install -y unzip && \
#     python -m pip install unityagents

# RUN apt-get install wget && \
#     mkdir -p runs && \
#     mkdir -p data && \
#     wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip -P ./resources/ && \
#     wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip -P ./resources/ && \
#     cd data && \
#     unzip Tennis_Linux.zip && \
#     unzip Tennis_Linux_NoVis.zip && \
#     rm Tennis_Linux.zip && \
#     rm Tennis_Linux_NoVis.zip

# # download repo
# RUN git clone https://github.com/amcumber/udacity-rl-tennis.git

# # TODO - determine if I can zip tennis_agents or something
# # COPY run_app.sh \
# #     main.py \
# #     tennis_agents/__init__.py \
# #     tennis_agents/agents.py \
# #     tennis_agents/config.toml \
# #     tennis_agents/ddpg_agent.py \
# #     tennis_agents/ddpg_model.py \
# #     tennis_agents/environments.py \
# #     tennis_agents/factory.py \
# #     tennis_agents/gym_environments.py \
# #     tennis_agents/noise_models.py \
# #     tennis_agents/replay_buffers.py \
# #     tennis_agents/trainers.py \
# #     tennis_agents/unity_environments.py \
# #     tennis_agents/workspace_utils.py \
# #     ./

# # # CMD ["python", "tennis.py", "train", "-f", "config.yml"]
