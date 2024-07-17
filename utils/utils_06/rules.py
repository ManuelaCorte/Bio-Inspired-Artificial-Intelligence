# Pygame
from pygame import Vector2
from .boid import Boid

# Random
import random


class Rules:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

    ###############
    ### Boid rules
    ###############

    # Find the neighbors of a boid in range
    def neighbors(self, boids: list[Boid]) -> list[Boid]:
        neighbors: list[Boid] = []
        for boid in boids:
            if boid.position != self.position:
                if self.position.distance_to(boid.position) < self.radius:  # type: ignore
                    neighbors.append(boid)
        return neighbors

    # Cohesion, boids try to fly towards the center of mass of neighboring boids
    def fly_towards_center(self, boids: list[Boid]):
        center = Vector2(0, 0)
        for boid in boids:
            if boid.position != self.position:
                center += boid.position
            return (center - self.position) / 100
        return Vector2(0, 0)

    # Separation, boids try to keep a small distance away from other objects (boids, obstacles)
    def keep_distance_away(self, boids: list[Boid], range: int = 9) -> Vector2:
        distance = Vector2()
        for boid in boids:
            if boid.position != self.position:
                if (boid.position - self.position).length() < range:
                    distance = distance - (boid.position - self.position)
        return distance

    # Alignment, boids try to match velocity with nearby boids
    def match_velocity(self, boids: list[Boid]) -> Vector2:
        velocity = Vector2(0, 0)
        for boid in boids:
            if boid.position != self.position:
                velocity += boid.velocity
        if len(boids) > 1:
            velocity = velocity / (len(boids) - 1)
            return (velocity - self.velocity) / 8  # type: ignore
        return Vector2(0, 0)

    # Avoid the hoiks
    def tend_to_place(self, hoiks: list[Boid]):
        run = Vector2(0, 0)
        for hoik in hoiks:
            if (hoik.position - self.position).length() < self.radius:  # type: ignore
                run = -(hoik.position - self.position).normalize()
            if (hoik.position - self.position).length() < hoik.size:  # type: ignore
                # Respawn object if colided with other object
                self.position = Vector2(
                    random.uniform(0, self.screen_width),
                    random.uniform(0, self.screen_height),
                )
        return run

    ###############
    ### Hoik rules
    ###############

    # Separation, hoiks try to keep distance away from eachother
    def my_food(self, hoiks: list[Boid]):
        distance = Vector2()
        for hoik in hoiks:
            if hoik.position != self.position:
                if (hoik.position - self.position).length() < 50:
                    distance = distance - (hoik.position - self.position)
        return distance

    # Chase the boids in range
    def chase(self, boids: list[Boid]):
        closest_boid = None
        for boid in boids:
            distance = abs(self.position.distance_to(boid.position))
            if distance < self.radius:  # type: ignore
                closest_boid = boid
        if closest_boid:
            return (closest_boid.position - self.position).normalize()

        return Vector2(0, 0)

    #########################
    ### Screen wrap or bound
    #########################

    # Wrap the boids around the screen when they go off screen
    def wrap_position(self):
        if self.position.x > self.screen_width:
            self.position.x = 0
        if self.position.x < 0:
            self.position.x = self.screen_width
        if self.position.y > self.screen_height:
            self.position.y = 0
        if self.position.y < 0:
            self.position.y = self.screen_height

    # Bound the boids to the screen. Change the velocity when they reach margin
    def bound_position(self, margin: int = 100):
        if self.position.x > self.screen_width - margin:
            self.velocity += Vector2(-0.7, 0)  # type: ignore
        if self.position.x < margin:
            self.velocity += Vector2(0.7, 0)  # type: ignore
        if self.position.y > self.screen_height - margin:
            self.velocity += Vector2(0, -0.7)  # type: ignore
        if self.position.y < margin:
            self.velocity += Vector2(0, 0.7)  # type: ignore
