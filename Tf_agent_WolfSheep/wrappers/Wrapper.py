import mesa
import inspect
#from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.typing.types import TimeStep, ArraySpec

def Agentwrapper(agent_class, name):
    class TFAgentWrapper(mesa.Agent):
        
        def __init__(self, unique_id: int, model: mesa.Model, agent_class) -> None:
            super(Agentwrapper,self).__init__(unique_id, model)
            self.fgn_class = agent_class
            self.__name__ = name
            self.functions = inspect.getmembers(agent_class,predicate=inspect.ismethod)
            
        def tf_functions(self, func_name=None, atrribute=None, *args, **kwargs):
            if func_name in self.functions:
                self.functions[func_name](*args,**kwargs)
            else:
                raise AttributeError(f"Function {func_name} not found in {self.__name__}")
        
        def step(self):
            return super().step()
        
        def advance(self) -> None:
            return super().advance()
        
        def remove(self) -> None:
            return super().remove()

def EnvWrapper(env_class, name):
    class EnvironmentWrapper(mesa.Model):

        def __init__(self, *args: inspect.Any, **kwargs: inspect.Any) -> None:
            super().__init__(*args, **kwargs)
            self.__name__ = name
            
        def get_action_spec(self):
            pass