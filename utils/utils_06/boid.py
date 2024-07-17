# Pygame
import pygame as pg
from pygame import Vector2, Surface
from typing_extensions import Self

# Random number generator
import random

# Rules class
from .rules import Rules


class Boid(Rules):
    def __init__(self, screen_width: int, screen_height: int):
        super().__init__(screen_width, screen_height)
        self.position = Vector2(
            random.uniform(0, screen_width), random.uniform(0, screen_height)
        )
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.radius = 100

    # Draw the boid
    def draw(self, screen: Surface):
        pg.draw.circle(screen, "red", (int(self.position.x), int(self.position.y)), 5)

    ### Update the position of the boid
    def update(
        self, boids: list[Self], ALIGNMENT: float, COHESION: float, SEPARATION: float
    ):
        ### Weights of the rules

        ### Neighbors in range
        n = Rules.neighbors(self, boids)  # type: ignore

        ### Rules to follow
        alignment = ALIGNMENT * Rules.match_velocity(self, n)
        cohesion = COHESION * Rules.fly_towards_center(self, n)
        separation_from_boids = SEPARATION * Rules.keep_distance_away(self, n, 9)

        # Update velocity
        self.velocity += alignment + cohesion + separation_from_boids

        # Limit the speed of the boids
        self.velocity.scale_to_length(5)

        # Update position
        self.position += self.velocity

        # Wrap the position of the boids
        Rules.bound_position(self)
