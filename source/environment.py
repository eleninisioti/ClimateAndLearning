""" Describes the abstract interface to an environment
"""

class Env:

  def __init__(self, mean):
    pass

  def climate_func(self, gen):
    pass


  def step(self, gen):
    self.mean = self.climate_func(gen)