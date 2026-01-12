from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization import SolaraViz, make_space_component
import random

# Define an agent that moves randomly on a grid
class WanderingAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        # Random color for each agent
        self.color = f"#{random.randint(0, 0xFFFFFF):06x}"
    
    def step(self):
        # Move to a random neighboring cell
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

# Define a model with a grid
class GridModel(Model):
    def __init__(self, n_agents=10, width=20, height=20):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        
        # Create agents and place them randomly on the grid
        for i in range(n_agents):
            agent = WanderingAgent(self)
            self.schedule.add(agent)
            
            # Place agent at random position
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
    
    def step(self):
        self.schedule.step()

# Create visualization
def agent_portrayal(agent):
    return {
        "color": agent.color,
        "size": 50,
    }

if __name__ == "__main__":
    model = GridModel(n_agents=10, width=20, height=20)
    page = SolaraViz(
        model,
        components=[make_space_component(agent_portrayal)],
        name="Wandering Agents"
    )
    # Display makes the visualization available
    page