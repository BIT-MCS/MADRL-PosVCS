import torch
from MAPPO.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy 

def _load_actor_critic(dir):
    model = Policy()
    actor = torch.load(dir + '/actor.pt')
    critic= torch.load(dir + '/critic.pt')
    print(actor.eval())

if __name__ == "__main__":
    _load_actor_critic('./MAPPO/results/MyEnv/MyEnv/mappo/check/run86/models')
