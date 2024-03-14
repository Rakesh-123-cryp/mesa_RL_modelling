from tf_agents.networks.q_network import QNetwork
from keras.activations import relu
from tf_agents.agents import DqnAgent
from keras.optimizers import Adam
import mesa
from Tf_agent_WolfSheep.random_walk import RandomWalker
from wrappers.Wrapper import Agentwrapper


# Tensorflow Agents for Wolf and Sheep
wolf_network = QNetwork(
    fc_layer_params=(50,50),
    dropout_layer_params=(0.01,0.01),
    activation_fn=relu
)
wolf_agent = DqnAgent(q_network=wolf_network,optimizer=Adam(),epsilon_greedy=0.3)

sheep_network = QNetwork(
    fc_layer_params=(50,50),
    dropout_layer_params=(0.01,0.01),
    activation_fn=relu
)
sheep_agent = DqnAgent(q_network=wolf_network,optimizer=Adam(),epsilon_greedy=0.3)


# Similar to a Tf_agent wrapper which can be built similarly

'''
def AgentWrapper(TF_Agent,**kwargs):
    class TfWrapper(mesa.Agent,TF_Agent):
        ........methods
    return TfWrapper.__new__(**kwargs)
'''

# The functions can be wrapped just by using a decorator @AgentWrapper(**kwargs)

class Wolf(RandomWalker,wolf_agent):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def step(self) -> None:
        self.random_move()
        self.energy -= 1

        # If there are sheep present, eat one
        x, y = self.pos
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        sheep = [obj for obj in this_cell if isinstance(obj, Sheep)]
        if len(sheep) > 0:
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.model.wolf_gain_from_food

            # Kill the sheep
            self.model.grid.remove_agent(sheep_to_eat)
            self.model.schedule.remove(sheep_to_eat)

        # Death or reproduction
        if self.energy < 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
        else:
            if self.random.random() < self.model.wolf_reproduce:
                # Create a new wolf cub
                self.energy /= 2
                cub = Wolf(
                    self.model.next_id(), self.pos, self.model, self.moore, self.energy
                )
                self.model.grid.place_agent(cub, cub.pos)
                self.model.schedule.add(cub)
       
       
                
class Sheep(RandomWalker,sheep_agent):
    """
    A sheep that walks around, reproduces (asexually) and gets eaten.

    The init is the same as the RandomWalker.
    """

    energy = None

    def __init__(self, unique_id, pos, model, moore, energy=None):
        super().__init__(unique_id, pos, model, moore=moore)
        self.energy = energy

    def step(self):
        """
        A model step. Move, then eat grass and reproduce.
        """
        self.random_move()
        living = True

        if self.model.grass:
            # Reduce energy
            self.energy -= 1

            # If there is grass available, eat it
            this_cell = self.model.grid.get_cell_list_contents([self.pos])
            grass_patch = next(obj for obj in this_cell if isinstance(obj, GrassPatch))
            if grass_patch.fully_grown:
                self.energy += self.model.sheep_gain_from_food
                grass_patch.fully_grown = False

            # Death
            if self.energy < 0:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                living = False

        if living and self.random.random() < self.model.sheep_reproduce:
            # Create a new sheep:
            if self.model.grass:
                self.energy /= 2
            lamb = Sheep(
                self.model.next_id(), self.pos, self.model, self.moore, self.energy
            )
            self.model.grid.place_agent(lamb, self.pos)
            self.model.schedule.add(lamb)
            
            
            
class GrassPatch(mesa.Agent):

    def __init__(self, unique_id, pos, model, fully_grown, countdown):

        super().__init__(unique_id, model)
        self.fully_grown = fully_grown
        self.countdown = countdown
        self.pos = pos

    def step(self):
        if not self.fully_grown:
            if self.countdown <= 0:
                # Set as fully grown
                self.fully_grown = True
                self.countdown = self.model.grass_regrowth_time
            else:
                self.countdown -= 1