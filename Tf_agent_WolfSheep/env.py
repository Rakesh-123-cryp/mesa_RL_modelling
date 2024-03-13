from typing import Any

from tf_agents.trajectories.time_step import TimeStep
from mesa.model import Model
from collections.abc import Iterable,Mapping
from numpy import ndarray
from tf_agents.environments.py_environment import PyEnvironment
from mesa.time import RandomActivationByTypeFiltered
from Tf_agent_WolfSheep.tfagent import wolf_agent, sheep_agent, GrassPatch
from tf_agents.specs import array_spec
import numpy as np
from tf_agents.typing.types import NestedArray,NestedArraySpec
import mesa


# This can we assumed as a Model Wrapper that encompases of all the tensorflow functionality but under the
# Mesa Model so integration with the Web interface is easier

'''For the Model we can either add it as a base class similar to the agents or we can retrieve the functions\
    from the data collector for action_spec and observation_spec in PyEnvironments (tf_agents library)'''

class WolfSheep(Model, PyEnvironment):
    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=50,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=False,
        grass_regrowth_time=30,
        sheep_gain_from_food=4,
        **kwargs) -> None:
        
        super(Model, self).__init__(**kwargs)
        super(PyEnvironment, self).__init__(**kwargs)
        self.width = width
        self.height = height
        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.wolf_gain_from_food = wolf_gain_from_food
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time
        self.sheep_gain_from_food = sheep_gain_from_food

        self.schedule = RandomActivationByTypeFiltered(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self.datacollector = mesa.DataCollector(
            {
                "Wolves": lambda m: m.schedule.get_type_count(wolf_agent),
                "Sheep": lambda m: m.schedule.get_type_count(sheep_agent),
                "Grass": lambda m: m.schedule.get_type_count(
                    GrassPatch, lambda x: x.fully_grown
                ),
            }
        )
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='states')

        # Create sheep:
        for i in range(self.initial_sheep):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.sheep_gain_from_food)
            sheep = sheep_agent(self.next_id(), (x, y), self, True, energy)
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Create wolves
        for i in range(self.initial_wolves):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.wolf_gain_from_food)
            wolf = wolf_agent(self.next_id(), (x, y), self, True, energy)
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

        # Create grass patches
        if self.grass:
            for agent, (x, y) in self.grid.coord_iter():
                fully_grown = self.random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = self.random.randrange(self.grass_regrowth_time)

                patch = GrassPatch(self.next_id(), (x, y), self, fully_grown, countdown)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        self.running = True
        self.datacollector.collect(self)

    def action_spec(self) -> Any | Iterable[NestedArraySpec] | Mapping[str, NestedArraySpec]:
        """Either tf agents like function as such or can create return facility using the datacollector"""
        
        # Write code to create action spec like the one in TF-Agents
        return self._action_spec
    
    def observation_spec(self) -> Any | Iterable[NestedArraySpec] | Mapping[str, NestedArraySpec]:
        """Similar to the action_spec function but for states"""
        
        # Write code to create observation spec returning the
        return self._observation_spec
    
    def step(self) -> None:
        return super().step()
    