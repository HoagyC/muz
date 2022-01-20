import random

ACTION_SIZE = 4


def choose_action(policy):  # Dummy policy chooses a random number
    return random.randrange(ACTION_SIZE)


def representation_net(frame):  # Dummy representation function returns a list of zeros
    return [0] * 5


def dynamics_net(
    latent, action
):  # Dummy dynamics function returns a list of random floats
    return [random.random() for _ in range(5)]


def prediction_net(latent):  #  Dummy prediction function
    value = random.random() * 10
    policy = [random.random() for _ in range(ACTION_SIZE)]
    return value, policy


class MCTS:
    def __init__(self, action_size):
        self.action_size = action_size

    def traverse_tree(self, root_node):
        current_node = root_node
        new_node = False

        while not new_node:
            value_pred, policy_pred, latent = (
                current_node.val_pred,
                current_node.pol_pred,
                current_node.latent,
            )
            action = choose_action(policy_pred)
            if current_node.children[action] == None:
                new_latent = dynamics_net(latent, action)
                new_val, new_policy = prediction_net(new_latent)
                current_node.insert(
                    action_n=action,
                    latent=new_latent,
                    val_pred=new_val,
                    pol_pred=new_policy,
                )
                new_node = True
            else:
                current_node = current_node.children[action]

    def search(self, n_simulations, current_frame):

        init_latent = representation_net(current_frame)
        root_node = TreeNode(init_latent, self.action_size)

        for i in range(n_simulations):
            self.traverse_tree(root_node)

        return root_node


class TreeNode:
    def __init__(self, latent, action_size, val_pred=None, pol_pred=None, parent=None):
        self.action_size = action_size
        self.children = [None] * action_size
        self.latent = latent
        self.val_pred = val_pred
        self.pol_pred = pol_pred
        self.n_descendents = 0
        self.parent = parent

    def insert(self, action_n, latent, val_pred, pol_pred):
        if self.children[action_n] is None:
            self.children[action_n] = TreeNode(
                latent=latent,
                val_pred=val_pred,
                pol_pred=pol_pred,
                action_size=self.action_size,
                parent=self,
            )
            self.increment()

        else:
            raise ValueError("This node has already been traversed")

    def increment(self):
        self.n_descendents += 1
        if self.parent:
            self.parent.increment()


if __name__ == "__main__":
    mcts = MCTS(ACTION_SIZE)
